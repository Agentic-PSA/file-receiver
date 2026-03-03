"""
BlueBox File Receiver Service
Receives files from Lovable Cloud edge functions and stores them
in tenant-isolated directories for the ingestion pipeline.
Also provides PDF-to-images conversion for page-by-page OCR.
Includes background OCR Worker for large scanned documents.

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
import asyncio
import traceback
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from enum import Enum

import httpx

# ── Configuration ──────────────────────────────────────────────

STORAGE_ROOT = os.getenv("STORAGE_ROOT", "/data/files")
API_KEY = os.getenv("FILE_RECEIVER_API_KEY", "change-me-in-production")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
PDF_DPI = int(os.getenv("PDF_DPI", "150"))

# AI Gateway for OCR
LOVABLE_AI_GATEWAY = "https://ai.gateway.lovable.dev/v1/chat/completions"
QWEN_API_URL = "https://qwen.agentic.pl/v1/chat/completions"
QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# OCR Worker config
OCR_BATCH_SIZE = int(os.getenv("OCR_BATCH_SIZE", "10"))  # pages per Gemini call
OCR_PAGE_CONCURRENCY = int(os.getenv("OCR_PAGE_CONCURRENCY", "3"))  # parallel batches
OCR_HEARTBEAT_INTERVAL = int(os.getenv("OCR_HEARTBEAT_INTERVAL", "10"))  # pages between heartbeats

# Active OCR jobs tracking (in-memory)
ocr_jobs: Dict[str, Dict[str, Any]] = {}

app = FastAPI(
    title="BlueBox File Receiver",
    version="3.0.0",
    description="Tenant-isolated file storage + PDF-to-images + Background OCR Worker",
)


# ── Models ─────────────────────────────────────────────────────

class FileStatus(str, Enum):
    received = "received"
    processing = "processing"
    ingested = "ingested"
    failed = "failed"


class PdfToImagesRequest(BaseModel):
    """Request body for PDF-to-images conversion (base64)."""
    pdf_base64: str
    dpi: int = 150
    quality: int = 85
    pages: Optional[str] = None  # "all", "1-5", "3"


class OcrJobRequest(BaseModel):
    """Request body for submitting an OCR job."""
    # Source
    file_download_url: Optional[str] = None
    file_base64: Optional[str] = None
    file_name: str

    # DB context for progress updates and callback
    tenant_id: str
    ingestion_id: str
    file_index: int
    file_id: str
    knowledge_base_id: str
    file_path: Optional[str] = None
    file_source_type: Optional[str] = None
    file_user_id: Optional[str] = None

    # Processing settings
    settings: Dict[str, Any] = {}
    anonymize: bool = False
    estimated_pages: int = 0

    # Auth
    lovable_api_key: str
    qwen_api_key: Optional[str] = None
    supabase_url: str
    supabase_service_role_key: str
    supabase_anon_key: str
    auth_token: Optional[str] = None

    # Callback: pipeline URL to call when OCR is done
    callback_url: str


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


# ══════════════════════════════════════════════════════════════
# OCR Worker — Background Processing
# ══════════════════════════════════════════════════════════════


async def ocr_call_gemini(
    image_base64: str,
    api_key: str,
    anonymize: bool = False,
    qwen_api_key: Optional[str] = None,
    extract_images: bool = False,
    image_desc_tokens: int = 200,
) -> Dict[str, Any]:
    """Call Gemini/Qwen Vision API for a single page image OCR."""
    if anonymize and qwen_api_key:
        api_url = QWEN_API_URL
        api_key_val = qwen_api_key
        model = QWEN_MODEL
    else:
        api_url = LOVABLE_AI_GATEWAY
        api_key_val = api_key
        model = "google/gemini-2.5-flash"

    if extract_images:
        system_prompt = (
            "Jestes ekspertem od ekstrakcji tresci z dokumentow. Wykonaj DWA zadania:\n"
            "1. EKSTRAKCJA TEKSTU: Przepisz dokladnie CALY tekst w Markdown.\n"
            "2. DETEKCJA OBRAZOW: Zidentyfikuj elementy graficzne. "
            f"Dla kazdego podaj typ i opis (max {image_desc_tokens} tokenow). Pusta lista jesli brak.\n\n"
            'Odpowiedz WYLACZNIE JSON:\n{"text": "...", "images": [{"type": "...", "description": "..."}]}'
        )
    else:
        system_prompt = (
            "Przepisz dokladnie caly tekst widoczny na tym obrazie strony dokumentu. "
            "Zachowaj oryginalna strukture: naglowki, akapity, punkty, tabele w formacie Markdown.\n\n"
            'Odpowiedz WYLACZNIE JSON: {"text": "wyekstrahowany tekst..."}'
        )

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            api_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key_val}",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Przeanalizuj te strone dokumentu:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        ],
                    },
                ],
                "max_tokens": 8000,
                "temperature": 0,
            },
        )
        resp.raise_for_status()
        result = resp.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Try parse JSON
        try:
            cleaned = content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            parsed = json.loads(cleaned.strip())
            return parsed
        except json.JSONDecodeError:
            return {"text": content, "images": []}


async def update_heartbeat(
    supabase_url: str,
    service_role_key: str,
    tenant_id: str,
    ingestion_id: str,
    pages_done: int,
    total_pages: int,
) -> str:
    """Update heartbeat in DB and return current ingestion status."""
    async with httpx.AsyncClient(timeout=30) as client:
        # Get tenant slug
        resp = await client.get(
            f"{supabase_url}/rest/v1/tenants?id=eq.{tenant_id}&select=slug",
            headers={
                "apikey": service_role_key,
                "Authorization": f"Bearer {service_role_key}",
            },
        )
        tenants = resp.json()
        if not tenants:
            return "unknown"
        slug = tenants[0]["slug"]
        schema = f"tenant_{slug}"

        # Check status & update heartbeat
        resp = await client.post(
            f"{supabase_url}/rest/v1/rpc/execute_tenant_query",
            headers={
                "apikey": service_role_key,
                "Authorization": f"Bearer {service_role_key}",
                "Content-Type": "application/json",
            },
            json={
                "p_query": f"UPDATE {schema}.ingestions SET updated_at = now() WHERE id = '{ingestion_id}' RETURNING status"
            },
        )
        rows = resp.json()
        if isinstance(rows, list) and rows:
            return rows[0].get("status", "processing")
        return "processing"


async def ocr_worker_process(job: OcrJobRequest):
    """
    Background task: OCR a large PDF page by page.

    1. Download PDF
    2. Convert to page images (local pdf2image)
    3. OCR each page via Gemini/Qwen
    4. Update DB progress
    5. Callback pipeline with full text
    """
    job_id = f"{job.ingestion_id}_{job.file_index}"
    ocr_jobs[job_id] = {
        "status": "processing",
        "pages_done": 0,
        "total_pages": job.estimated_pages,
        "started_at": datetime.utcnow().isoformat(),
        "error": None,
    }

    try:
        from pdf2image import convert_from_bytes

        # Step 1: Get PDF bytes
        print(f"[ocr-worker] Starting job {job_id}: {job.file_name}")

        if job.file_download_url:
            print(f"[ocr-worker] Downloading from URL...")
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.get(job.file_download_url)
                resp.raise_for_status()
                pdf_bytes = resp.content
        elif job.file_base64:
            pdf_bytes = base64.b64decode(job.file_base64)
        else:
            raise ValueError("No file source provided")

        print(f"[ocr-worker] PDF size: {len(pdf_bytes)} bytes")

        # Step 2: Convert ALL pages to images
        print(f"[ocr-worker] Converting PDF to images (DPI={PDF_DPI})...")
        images = convert_from_bytes(pdf_bytes, dpi=PDF_DPI, fmt="jpeg")
        total_pages = len(images)
        ocr_jobs[job_id]["total_pages"] = total_pages
        print(f"[ocr-worker] Got {total_pages} page images")

        # Free PDF bytes from memory
        del pdf_bytes

        # Step 3: OCR each page in batches
        extract_images = job.settings.get("extractImages", False)
        image_desc_tokens = job.settings.get("imageDescTokens", 200)
        all_page_texts = [""] * total_pages
        all_images = []
        pages_done = 0

        # Process pages in small concurrent batches
        for batch_start in range(0, total_pages, OCR_PAGE_CONCURRENCY):
            batch_end = min(batch_start + OCR_PAGE_CONCURRENCY, total_pages)
            batch_indices = list(range(batch_start, batch_end))

            # Check cancellation
            if pages_done > 0 and pages_done % OCR_HEARTBEAT_INTERVAL == 0:
                try:
                    status = await update_heartbeat(
                        job.supabase_url, job.supabase_service_role_key,
                        job.tenant_id, job.ingestion_id,
                        pages_done, total_pages,
                    )
                    if status in ("cancelled", "paused"):
                        print(f"[ocr-worker] Job {job_id} {status} by user, stopping.")
                        ocr_jobs[job_id]["status"] = status
                        return
                except Exception as e:
                    print(f"[ocr-worker] Heartbeat error: {e}")

            # Convert batch images to base64 and OCR
            async def process_page(page_idx: int) -> Dict:
                img = images[page_idx]
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                buf.close()

                for attempt in range(3):
                    try:
                        result = await ocr_call_gemini(
                            img_b64, job.lovable_api_key,
                            anonymize=job.anonymize,
                            qwen_api_key=job.qwen_api_key,
                            extract_images=extract_images,
                            image_desc_tokens=image_desc_tokens,
                        )
                        return {"page_idx": page_idx, "result": result}
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 429:
                            wait = 10 * (attempt + 1)
                            print(f"[ocr-worker] Rate limited on page {page_idx + 1}, waiting {wait}s...")
                            await asyncio.sleep(wait)
                        else:
                            raise
                    except Exception as e:
                        if attempt < 2:
                            await asyncio.sleep(5)
                        else:
                            print(f"[ocr-worker] Page {page_idx + 1} failed after 3 attempts: {e}")
                            return {"page_idx": page_idx, "result": {"text": f"[OCR ERROR: {e}]", "images": []}}
                return {"page_idx": page_idx, "result": {"text": "[OCR ERROR]", "images": []}}

            tasks = [process_page(idx) for idx in batch_indices]
            results = await asyncio.gather(*tasks)

            for r in results:
                idx = r["page_idx"]
                text = r["result"].get("text", "")
                all_page_texts[idx] = text
                pages_done += 1

                # Collect images
                detected_images = r["result"].get("images", [])
                for img_info in detected_images:
                    all_images.append({
                        "description": f"[Obraz: {img_info.get('type', 'unknown')}, strona {idx + 1}] {img_info.get('description', '')}",
                        "section_context": text[:1000] if text else "",
                        "page": idx + 1,
                    })

            ocr_jobs[job_id]["pages_done"] = pages_done
            print(f"[ocr-worker] Progress: {pages_done}/{total_pages} pages")

            # Small delay between batches to respect rate limits
            if batch_end < total_pages:
                await asyncio.sleep(0.5)

        # Free images from memory
        del images

        # Step 4: Combine text
        full_text = "\n\n---\n\n".join(t for t in all_page_texts if t)
        print(f"[ocr-worker] OCR complete: {len(full_text)} chars, {len(all_images)} images from {total_pages} pages")

        # Step 5: Callback pipeline with extracted text
        print(f"[ocr-worker] Calling back pipeline...")
        callback_body = {
            "file_name": job.file_name,
            "file_id": job.file_id,
            "knowledge_base_id": job.knowledge_base_id,
            "settings": job.settings,
            "document_text": full_text,
            "tenant_id": job.tenant_id,
            "ingestion_id": job.ingestion_id,
            "file_index": job.file_index,
            "file_path": job.file_path,
            "file_source_type": job.file_source_type,
            "file_user_id": job.file_user_id,
            "auth_token": job.auth_token,
            # Pass image data so pipeline can create image chunks
            "_ocr_images": all_images,
            "_ocr_page_count": total_pages,
        }

        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                job.callback_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {job.auth_token or job.supabase_anon_key}",
                    "apikey": job.supabase_anon_key,
                },
                json=callback_body,
            )
            if resp.status_code >= 400:
                print(f"[ocr-worker] Callback failed: {resp.status_code} {resp.text[:500]}")
            else:
                print(f"[ocr-worker] Callback success: {resp.status_code}")

        ocr_jobs[job_id]["status"] = "completed"
        ocr_jobs[job_id]["pages_done"] = total_pages

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[ocr-worker] Job {job_id} failed: {error_msg}")
        print(traceback.format_exc())
        ocr_jobs[job_id]["status"] = "failed"
        ocr_jobs[job_id]["error"] = error_msg

        # Mark file as error in DB
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{job.supabase_url}/rest/v1/tenants?id=eq.{job.tenant_id}&select=slug",
                    headers={
                        "apikey": job.supabase_service_role_key,
                        "Authorization": f"Bearer {job.supabase_service_role_key}",
                    },
                )
                tenants = resp.json()
                if tenants:
                    slug = tenants[0]["slug"]
                    schema = f"tenant_{slug}"

                    # Get pending_files, mark this file as error
                    resp2 = await client.post(
                        f"{job.supabase_url}/rest/v1/rpc/execute_tenant_query",
                        headers={
                            "apikey": job.supabase_service_role_key,
                            "Authorization": f"Bearer {job.supabase_service_role_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "p_query": f"SELECT pending_files FROM {schema}.ingestions WHERE id = '{job.ingestion_id}'"
                        },
                    )
                    rows = resp2.json()
                    if isinstance(rows, list) and rows:
                        pf = rows[0].get("pending_files", [])
                        if job.file_index < len(pf):
                            pf[job.file_index]["status"] = "error"
                            pf[job.file_index]["error"] = error_msg[:500]
                            pc = sum(1 for f in pf if f.get("status") == "done")
                            ec = sum(1 for f in pf if f.get("status") == "error")
                            pf_json = json.dumps(pf).replace("'", "''")

                            await client.post(
                                f"{job.supabase_url}/rest/v1/rpc/execute_tenant_query",
                                headers={
                                    "apikey": job.supabase_service_role_key,
                                    "Authorization": f"Bearer {job.supabase_service_role_key}",
                                    "Content-Type": "application/json",
                                },
                                json={
                                    "p_query": f"UPDATE {schema}.ingestions SET pending_files = '{pf_json}'::jsonb, processed_count = {pc}, error_count = {ec}, updated_at = now() WHERE id = '{job.ingestion_id}' RETURNING id"
                                },
                            )

                            # Invoke worker for next file
                            remaining = sum(1 for f in pf if f.get("status") == "pending")
                            if remaining > 0:
                                await client.post(
                                    f"{job.supabase_url}/functions/v1/ingest-worker",
                                    headers={
                                        "Content-Type": "application/json",
                                        "Authorization": f"Bearer {job.supabase_anon_key}",
                                        "apikey": job.supabase_anon_key,
                                    },
                                    json={
                                        "ingestion_id": job.ingestion_id,
                                        "tenant_id": job.tenant_id,
                                        "auth_token": job.auth_token,
                                    },
                                )
                            else:
                                final_status = "failed" if ec == len(pf) else "completed"
                                await client.post(
                                    f"{job.supabase_url}/rest/v1/rpc/execute_tenant_query",
                                    headers={
                                        "apikey": job.supabase_service_role_key,
                                        "Authorization": f"Bearer {job.supabase_service_role_key}",
                                        "Content-Type": "application/json",
                                    },
                                    json={
                                        "p_query": f"UPDATE {schema}.ingestions SET status = '{final_status}', updated_at = now() WHERE id = '{job.ingestion_id}' RETURNING id"
                                    },
                                )
        except Exception as db_err:
            print(f"[ocr-worker] Error updating DB on failure: {db_err}")


# ── Endpoints ──────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Health check endpoint."""
    storage_ok = Path(STORAGE_ROOT).exists()
    poppler_ok = shutil.which("pdftoppm") is not None

    # Count tenants & files on disk
    root = Path(STORAGE_ROOT)
    tenant_count = sum(1 for d in root.iterdir() if d.is_dir()) if root.exists() else 0

    active_jobs = sum(1 for j in ocr_jobs.values() if j["status"] == "processing")

    return {
        "status": "ok" if storage_ok else "degraded",
        "storage_root": STORAGE_ROOT,
        "storage_accessible": storage_ok,
        "poppler_available": poppler_ok,
        "tenant_count": tenant_count,
        "active_ocr_jobs": active_jobs,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ── OCR Worker endpoints ──────────────────────────────────────


@app.post("/ocr-job")
async def submit_ocr_job(
    req: OcrJobRequest,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(...),
):
    """
    Submit a large PDF for background OCR processing.
    The worker will process pages, update DB progress, and callback pipeline when done.
    """
    verify_api_key(x_api_key)

    job_id = f"{req.ingestion_id}_{req.file_index}"

    # Check if job already running
    if job_id in ocr_jobs and ocr_jobs[job_id]["status"] == "processing":
        return JSONResponse(status_code=409, content={
            "error": "Job already running",
            "job_id": job_id,
            "status": ocr_jobs[job_id],
        })

    # Start background processing
    background_tasks.add_task(ocr_worker_process, req)

    return JSONResponse(status_code=202, content={
        "job_id": job_id,
        "message": f"OCR job accepted for {req.file_name} (~{req.estimated_pages} pages)",
        "status": "accepted",
    })


@app.get("/ocr-job/{ingestion_id}/{file_index}")
async def get_ocr_job_status(
    ingestion_id: str,
    file_index: int,
    x_api_key: str = Header(...),
):
    """Check status of an OCR job."""
    verify_api_key(x_api_key)

    job_id = f"{ingestion_id}_{file_index}"
    if job_id not in ocr_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return ocr_jobs[job_id]


@app.get("/ocr-jobs")
async def list_ocr_jobs(x_api_key: str = Header(...)):
    """List all OCR jobs."""
    verify_api_key(x_api_key)
    return {
        "jobs": ocr_jobs,
        "total": len(ocr_jobs),
        "active": sum(1 for j in ocr_jobs.values() if j["status"] == "processing"),
    }


# ── Upload ─────────────────────────────────────────────────────
# ... keep existing code (all upload, download, list, delete, stats endpoints)


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
    """Serve/download a specific file."""
    verify_api_key(x_api_key)
    verify_tenant(tenant_slug, x_tenant)

    file_path = Path(STORAGE_ROOT) / tenant_slug / kb_id / doc_id / filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

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

    if not str(doc_path.resolve()).startswith(str(Path(STORAGE_ROOT).resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    shutil.rmtree(doc_path)

    return {"deleted": True, "doc_id": doc_id}


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


@app.get("/files/{tenant_slug}")
async def list_tenant_kbs(
    tenant_slug: str,
    x_api_key: str = Header(...),
    x_tenant: Optional[str] = Header(None),
):
    """List all knowledge base directories for a tenant."""
    verify_api_key(x_api_key)
    verify_tenant(tenant_slug, x_tenant)

    tenant_path = Path(STORAGE_ROOT) / tenant_slug
    if not tenant_path.exists():
        return {"tenant_slug": tenant_slug, "knowledge_bases": [], "total": 0}

    kbs = []
    for kb_dir in sorted(tenant_path.iterdir()):
        if not kb_dir.is_dir():
            continue
        doc_count = 0
        file_count = 0
        total_bytes = 0
        for doc_dir in kb_dir.iterdir():
            if doc_dir.is_dir():
                doc_count += 1
                for f in doc_dir.iterdir():
                    if f.is_file() and f.name != "metadata.json":
                        file_count += 1
                        total_bytes += f.stat().st_size
        kbs.append({
            "kb_id": kb_dir.name,
            "doc_count": doc_count,
            "file_count": file_count,
            "total_size_bytes": total_bytes,
        })

    return {
        "tenant_slug": tenant_slug,
        "knowledge_bases": kbs,
        "total": len(kbs),
    }


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
