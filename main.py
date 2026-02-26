"""
BlueBox File Receiver Service
Receives files from Lovable Cloud edge functions and stores them
in tenant-isolated directories for the ingestion pipeline.
Also provides PDF-to-images conversion for page-by-page OCR.

Storage layout:
  /data/files/{tenant_slug}/{kb_id}/{doc_id}/{filename}

Every request must include X-Api-Key header.
Tenant isolation is enforced: X-Tenant header must match the path tenant_slug.
"""

import base64
import io
import json
import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from enum import Enum

# ── Configuration ──────────────────────────────────────────────

STORAGE_ROOT = os.getenv("STORAGE_ROOT", "/data/files")
API_KEY = os.getenv("FILE_RECEIVER_API_KEY", "change-me-in-production")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
PDF_DPI = int(os.getenv("PDF_DPI", "150"))

app = FastAPI(
    title="BlueBox File Receiver",
    version="2.0.0",
    description="Tenant-isolated file storage with per-document directories + PDF-to-images conversion",
)


# ── Models ─────────────────────────────────────────────────────

class FileStatus(str, Enum):
    received = "received"
    processing = "processing"
    ingested = "ingested"
    failed = "failed"


class PdfToImagesRequest(BaseModel):
    """Request body for PDF-to-images conversion."""
    pdf_base64: str
    dpi: int = 150
    quality: int = 85
    pages: Optional[str] = None  # "all", "1-5", "3"


# ── Auth & tenant guard ───────────────────────────────────────

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def verify_tenant(path_tenant: str, x_tenant: Optional[str] = None):
    """Ensure the X-Tenant header matches the path tenant_slug (if provided)."""
    if x_tenant and x_tenant != path_tenant:
        raise HTTPException(status_code=403, detail="Tenant mismatch")


# ── Helpers ────────────────────────────────────────────────────

def get_doc_path(tenant_slug: str, kb_id: str, doc_id: str) -> Path:
    """Returns and ensures the directory for a specific document."""
    path = Path(STORAGE_ROOT) / tenant_slug / kb_id / doc_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_kb_path(tenant_slug: str, kb_id: str) -> Path:
    path = Path(STORAGE_ROOT) / tenant_slug / kb_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_metadata(doc_path: Path, metadata: dict):
    """Write/update metadata.json in the document directory."""
    meta_file = doc_path / "metadata.json"
    existing = {}
    if meta_file.exists():
        try:
            existing = json.loads(meta_file.read_text())
        except Exception:
            pass
    existing.update(metadata)
    existing["updated_at"] = datetime.utcnow().isoformat() + "Z"
    meta_file.write_text(json.dumps(existing, ensure_ascii=False, indent=2))


def read_metadata(doc_path: Path) -> dict:
    meta_file = doc_path / "metadata.json"
    if meta_file.exists():
        try:
            return json.loads(meta_file.read_text())
        except Exception:
            pass
    return {}


def parse_pages(pages_str: Optional[str]) -> tuple | None:
    """Parse page range: None/'all' → None, '1-5' → (1, 5), '3' → (3, 3)."""
    if not pages_str or pages_str.strip().lower() == "all":
        return None
    parts = pages_str.strip().split("-")
    if len(parts) == 1:
        p = int(parts[0])
        return (p, p)
    return (int(parts[0]), int(parts[1]))


MIME_MAP = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".html": "text/html",
    ".xml": "application/xml",
    ".json": "application/json",
    ".md": "text/markdown",
    ".rtf": "application/rtf",
    ".odt": "application/vnd.oasis.opendocument.text",
}


# ── Endpoints ──────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Health check endpoint."""
    storage_ok = Path(STORAGE_ROOT).exists()
    poppler_ok = shutil.which("pdftoppm") is not None

    # Count tenants & files on disk
    root = Path(STORAGE_ROOT)
    tenant_count = sum(1 for d in root.iterdir() if d.is_dir()) if root.exists() else 0

    return {
        "status": "ok" if storage_ok else "degraded",
        "storage_root": STORAGE_ROOT,
        "storage_accessible": storage_ok,
        "poppler_available": poppler_ok,
        "tenant_count": tenant_count,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ── Upload ─────────────────────────────────────────────────────


@app.post("/upload/{tenant_slug}/{kb_id}/{doc_id}")
async def upload_file(
    tenant_slug: str,
    kb_id: str,
    doc_id: str,
    file: UploadFile = File(...),
    source: Optional[str] = Query(None, description="Source: onedrive, ezd, manual"),
    original_name: Optional[str] = Query(None, description="Original filename if different"),
    x_api_key: str = Header(...),
    x_tenant: Optional[str] = Header(None),
):
    """
    Upload a file for a specific document.
    Stored at: /{STORAGE_ROOT}/{tenant_slug}/{kb_id}/{doc_id}/{filename}
    """
    verify_api_key(x_api_key)
    verify_tenant(tenant_slug, x_tenant)

    contents = await file.read()
    size_bytes = len(contents)
    if size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max: {MAX_FILE_SIZE_MB}MB")

    filename = file.filename or "unnamed"
    doc_path = get_doc_path(tenant_slug, kb_id, doc_id)
    file_path = doc_path / filename
    file_path.write_bytes(contents)

    # Write metadata
    write_metadata(doc_path, {
        "doc_id": doc_id,
        "kb_id": kb_id,
        "tenant_slug": tenant_slug,
        "filename": filename,
        "original_name": original_name or filename,
        "size_bytes": size_bytes,
        "source": source,
        "status": "received",
        "created_at": datetime.utcnow().isoformat() + "Z",
    })

    return JSONResponse(status_code=201, content={
        "doc_id": doc_id,
        "filename": filename,
        "size_bytes": size_bytes,
        "path": f"{tenant_slug}/{kb_id}/{doc_id}/{filename}",
        "source": source,
    })


# ── Legacy upload (backward compat) ───────────────────────────


@app.post("/files/upload")
async def upload_file_legacy(
    file: UploadFile = File(...),
    tenant_slug: str = Query(...),
    knowledge_base_id: str = Query(...),
    source: Optional[str] = Query(None),
    x_api_key: str = Header(...),
):
    """Legacy upload endpoint — auto-generates doc_id."""
    verify_api_key(x_api_key)

    doc_id = uuid.uuid4().hex[:12]
    contents = await file.read()
    size_bytes = len(contents)
    if size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max: {MAX_FILE_SIZE_MB}MB")

    filename = file.filename or "unnamed"
    doc_path = get_doc_path(tenant_slug, knowledge_base_id, doc_id)
    file_path = doc_path / filename
    file_path.write_bytes(contents)

    write_metadata(doc_path, {
        "doc_id": doc_id,
        "kb_id": knowledge_base_id,
        "tenant_slug": tenant_slug,
        "filename": filename,
        "original_name": filename,
        "size_bytes": size_bytes,
        "source": source,
        "status": "received",
        "created_at": datetime.utcnow().isoformat() + "Z",
    })

    return JSONResponse(status_code=201, content={
        "doc_id": doc_id,
        "filename": filename,
        "size_bytes": size_bytes,
        "path": f"{tenant_slug}/{knowledge_base_id}/{doc_id}/{filename}",
        "source": source,
    })


# ── Download / serve file ─────────────────────────────────────


@app.get("/files/{tenant_slug}/{kb_id}/{doc_id}/{filename}")
async def download_file(
    tenant_slug: str,
    kb_id: str,
    doc_id: str,
    filename: str,
    x_api_key: str = Header(...),
    x_tenant: Optional[str] = Header(None),
    download: bool = Query(False, description="Force download instead of inline display"),
):
    """
    Serve/download a specific file.
    - download=false (default): opens inline in browser (Content-Disposition: inline)
    - download=true: forces download (Content-Disposition: attachment)
    """
    verify_api_key(x_api_key)
    verify_tenant(tenant_slug, x_tenant)

    file_path = Path(STORAGE_ROOT) / tenant_slug / kb_id / doc_id / filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Security: ensure resolved path is within STORAGE_ROOT
    if not str(file_path.resolve()).startswith(str(Path(STORAGE_ROOT).resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    ext = file_path.suffix.lower()
    media_type = MIME_MAP.get(ext, "application/octet-stream")

    disposition = "attachment" if download else "inline"

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename,
        headers={"Content-Disposition": f'{disposition}; filename="{filename}"'},
    )


# ── List files in a document ──────────────────────────────────


@app.get("/files/{tenant_slug}/{kb_id}/{doc_id}")
async def list_doc_files(
    tenant_slug: str,
    kb_id: str,
    doc_id: str,
    x_api_key: str = Header(...),
    x_tenant: Optional[str] = Header(None),
):
    """List all files in a specific document directory."""
    verify_api_key(x_api_key)
    verify_tenant(tenant_slug, x_tenant)

    doc_path = Path(STORAGE_ROOT) / tenant_slug / kb_id / doc_id
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    files = []
    for f in doc_path.iterdir():
        if f.is_file() and f.name != "metadata.json":
            files.append({
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "url": f"/files/{tenant_slug}/{kb_id}/{doc_id}/{f.name}",
            })

    metadata = read_metadata(doc_path)

    return {
        "doc_id": doc_id,
        "files": files,
        "metadata": metadata,
        "total": len(files),
    }


# ── List documents in a KB ────────────────────────────────────


@app.get("/files/{tenant_slug}/{kb_id}")
async def list_kb_documents(
    tenant_slug: str,
    kb_id: str,
    x_api_key: str = Header(...),
    x_tenant: Optional[str] = Header(None),
):
    """List all documents (directories) in a knowledge base."""
    verify_api_key(x_api_key)
    verify_tenant(tenant_slug, x_tenant)

    kb_path = Path(STORAGE_ROOT) / tenant_slug / kb_id
    if not kb_path.exists():
        return {"kb_id": kb_id, "documents": [], "total": 0}

    documents = []
    for d in kb_path.iterdir():
        if d.is_dir():
            metadata = read_metadata(d)
            file_count = sum(1 for f in d.iterdir() if f.is_file() and f.name != "metadata.json")
            total_size = sum(f.stat().st_size for f in d.iterdir() if f.is_file() and f.name != "metadata.json")
            documents.append({
                "doc_id": d.name,
                "file_count": file_count,
                "total_size_bytes": total_size,
                "metadata": metadata,
            })

    return {
        "kb_id": kb_id,
        "documents": documents,
        "total": len(documents),
    }


# ── Delete document ────────────────────────────────────────────


@app.delete("/files/{tenant_slug}/{kb_id}/{doc_id}")
async def delete_document(
    tenant_slug: str,
    kb_id: str,
    doc_id: str,
    x_api_key: str = Header(...),
    x_tenant: Optional[str] = Header(None),
):
    """Delete an entire document directory and all its files."""
    verify_api_key(x_api_key)
    verify_tenant(tenant_slug, x_tenant)

    doc_path = Path(STORAGE_ROOT) / tenant_slug / kb_id / doc_id
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    # Security check
    if not str(doc_path.resolve()).startswith(str(Path(STORAGE_ROOT).resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    shutil.rmtree(doc_path)

    return {"deleted": True, "doc_id": doc_id}


# ── Delete single file ────────────────────────────────────────


@app.delete("/files/{tenant_slug}/{kb_id}/{doc_id}/{filename}")
async def delete_single_file(
    tenant_slug: str,
    kb_id: str,
    doc_id: str,
    filename: str,
    x_api_key: str = Header(...),
    x_tenant: Optional[str] = Header(None),
):
    """Delete a single file from a document directory."""
    verify_api_key(x_api_key)
    verify_tenant(tenant_slug, x_tenant)

    file_path = Path(STORAGE_ROOT) / tenant_slug / kb_id / doc_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if not str(file_path.resolve()).startswith(str(Path(STORAGE_ROOT).resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    file_path.unlink()

    return {"deleted": True, "filename": filename, "doc_id": doc_id}


# ── Tenant stats ──────────────────────────────────────────────


@app.get("/tenants/{tenant_slug}/stats")
async def tenant_stats(tenant_slug: str, x_api_key: str = Header(...)):
    """Get storage statistics for a tenant."""
    verify_api_key(x_api_key)

    tenant_path = Path(STORAGE_ROOT) / tenant_slug
    if not tenant_path.exists():
        return {
            "tenant_slug": tenant_slug,
            "kb_count": 0,
            "doc_count": 0,
            "file_count": 0,
            "total_bytes": 0,
        }

    kb_count = 0
    doc_count = 0
    file_count = 0
    total_bytes = 0

    for kb_dir in tenant_path.iterdir():
        if kb_dir.is_dir():
            kb_count += 1
            for doc_dir in kb_dir.iterdir():
                if doc_dir.is_dir():
                    doc_count += 1
                    for f in doc_dir.iterdir():
                        if f.is_file() and f.name != "metadata.json":
                            file_count += 1
                            total_bytes += f.stat().st_size

    return {
        "tenant_slug": tenant_slug,
        "kb_count": kb_count,
        "doc_count": doc_count,
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


# ── PDF to images ─────────────────────────────────────────────


@app.post("/pdf-to-images")
async def pdf_to_images(req: PdfToImagesRequest, x_api_key: str = Header(...)):
    """
    Convert a PDF (base64) to an array of JPEG page images (base64).
    Uses poppler (pdf2image) for rendering.
    """
    verify_api_key(x_api_key)

    try:
        from pdf2image import convert_from_bytes
    except ImportError:
        raise HTTPException(status_code=500, detail="pdf2image not installed")

    try:
        pdf_bytes = base64.b64decode(req.pdf_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 PDF data")

    if len(pdf_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"PDF too large. Max: {MAX_FILE_SIZE_MB}MB")

    pages = parse_pages(req.pages)
    kwargs = {"dpi": req.dpi, "fmt": "jpeg"}
    if pages:
        kwargs["first_page"] = pages[0]
        kwargs["last_page"] = pages[1]

    try:
        images = convert_from_bytes(pdf_bytes, **kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {str(e)}")

    result_pages = []
    start_page = pages[0] if pages else 1
    for i, img in enumerate(images):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=req.quality)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        result_pages.append({
            "page": start_page + i,
            "image_base64": img_b64,
        })

    return {
        "pages": result_pages,
        "total_pages": len(result_pages),
        "dpi": req.dpi,
    }
