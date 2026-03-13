"""
BlueBox File Receiver Service
Receives files from Lovable Cloud edge functions and stores them
in tenant-isolated directories for the ingestion pipeline.
Also provides PDF-to-images conversion for page-by-page OCR,
EPUB text extraction, and a background OCR Worker for large scanned documents.

Storage layout:
  /data/files/{tenant_slug}/{kb_id}/{doc_id}/{filename}

Every request must include X-Api-Key header.
Tenant isolation is enforced: X-Tenant header must match the path tenant_slug.
"""

import base64
import functools
import io
import json
import os
import re
import uuid
import shutil
import asyncio
import traceback
import urllib.parse
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from enum import Enum

import httpx


class NonRetryableAPIError(Exception):
    """Raised when all AI providers fail with non-retryable errors (400, 401, 403).
    Outer retry loops should NOT retry on this — the request is fundamentally broken
    (bad key, invalid payload, auth failure) and retrying will only waste time."""
    pass


# ── Configuration ──────────────────────────────────────────────

STORAGE_ROOT = os.getenv("STORAGE_ROOT", "/data/files")
API_KEY = os.getenv("FILE_RECEIVER_API_KEY", "change-me-in-production")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
PDF_DPI = int(os.getenv("PDF_DPI", "150"))

# AI Gateway for OCR
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"
QWEN_API_URL = "https://172.16.10.119:8010/v1/chat/completions"
QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# OCR Worker config
# Number of pages rasterised from disk to RAM at once — keeps peak RAM ~130 MB
# regardless of total document length (e.g. 2000 pages @ 150 DPI ≈ 13 GB without chunking)
PDF_CHUNK_SIZE = int(os.getenv("PDF_CHUNK_SIZE", "20"))
OCR_PAGE_CONCURRENCY = int(os.getenv("OCR_PAGE_CONCURRENCY", "3"))    # parallel OCR API calls per batch
OCR_HEARTBEAT_INTERVAL = int(os.getenv("OCR_HEARTBEAT_INTERVAL", "10"))  # pages between DB heartbeats

# Active OCR jobs tracking (in-memory)
ocr_jobs: Dict[str, Dict[str, Any]] = {}

# Active EPUB extraction jobs tracking (in-memory)
epub_jobs: Dict[str, Dict[str, Any]] = {}

app = FastAPI(
    title="BlueBox File Receiver",
    version="4.7.0",
    description="Tenant-isolated file storage + PDF-to-images + Background OCR Worker + Background EPUB extraction",
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

class EpubExtractRequest(BaseModel):
    """Request body for submitting an EPUB extraction job."""
    # Source
    file_download_url: Optional[str] = None
    file_base64: Optional[str] = None
    file_name: str = "book.epub"

    # DB context for progress updates and callback
    tenant_id: str
    ingestion_id: str
    file_index: int
    file_id: str
    knowledge_base_id: str
    tenant_slug: str
    doc_id: str
    file_path: Optional[str] = None
    file_source_type: Optional[str] = None
    file_user_id: Optional[str] = None

    # Processing settings
    settings: Dict[str, Any] = {}
    image_desc_tokens: int = 200

    # Auth
    supabase_url: str
    supabase_service_role_key: str
    supabase_anon_key: str
    auth_token: Optional[str] = None

    # Callback: pipeline URL to call when extraction is done
    callback_url: str

# ── EPUB helpers ───────────────────────────────────────────────

class _HTMLToText(HTMLParser):
    """Minimal HTML→Markdown-like text converter preserving headers, lists, and image references."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._tag_stack: list[str] = []
        self._images: list[dict] = []  # collected image references

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        self._tag_stack.append(tag)
        if tag == "h1":
            self._parts.append("\n# ")
        elif tag == "h2":
            self._parts.append("\n## ")
        elif tag == "h3":
            self._parts.append("\n### ")
        elif tag in ("h4", "h5", "h6"):
            self._parts.append("\n#### ")
        elif tag == "li":
            self._parts.append("- ")
        elif tag == "br":
            self._parts.append("\n")
        elif tag == "img":
            attrs_dict = dict(attrs)
            src = attrs_dict.get("src", "")
            alt = attrs_dict.get("alt", "")
            if src:
                img_idx = len(self._images)
                self._images.append({"src": src, "alt": alt})
                self._parts.append(f"\n![{alt}](epub_image:{img_idx})\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()
        if tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"):
            self._parts.append("\n")

    def handle_data(self, data):
        self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

def _parse_epub_bytes(data: bytes) -> tuple[str, int, list[dict]]:
    """
    Parse EPUB ZIP bytes → Markdown text following OPF spine order.
    Returns (full_text, chapter_count, images_list).
    Images are extracted directly from the ZIP — no cropping needed
    because XHTML <img> tags define exact image boundaries.
    """
    zf = zipfile.ZipFile(io.BytesIO(data))

    # 1. container.xml → rootfile
    container = ET.fromstring(zf.read("META-INF/container.xml"))
    ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
    rootfile_el = container.find(".//c:rootfile", ns)
    if rootfile_el is None:
        rootfile_el = container.find(".//{*}rootfile")
    if rootfile_el is None:
        raise ValueError("Invalid EPUB: cannot find rootfile in container.xml")

    opf_path = rootfile_el.attrib["full-path"]
    opf_dir = os.path.dirname(opf_path)
    if opf_dir:
        opf_dir += "/"

    # 2. Parse OPF — manifest and spine
    opf = ET.fromstring(zf.read(opf_path))
    manifest = {}
    for item in opf.findall(".//{*}item"):
        manifest[item.attrib.get("id", "")] = item.attrib.get("href", "")
    spine_ids = [ref.attrib["idref"] for ref in opf.findall(".//{*}itemref")]

    # Build case-insensitive lookup for ZIP entries
    zip_names_lower = {n.lower(): n for n in zf.namelist()}
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}

    # 3. Extract chapters in spine order, collecting images
    chapters: list[str] = []
    all_images: list[dict] = []

    for chapter_idx, sid in enumerate(spine_ids):
        href = manifest.get(sid)
        if not href:
            continue
        full_path = opf_dir + href if opf_dir else href
        chapter_dir = os.path.dirname(full_path)
        if chapter_dir:
            chapter_dir += "/"

        matching = [n for n in zf.namelist() if n.lower() == full_path.lower()]
        if not matching:
            continue
        html_bytes = zf.read(matching[0])
        parser = _HTMLToText()
        parser.feed(html_bytes.decode("utf-8", errors="replace"))
        text = parser.get_text()

        # Resolve image references from this chapter's <img> tags
        for img_info in parser._images:
            src = img_info["src"]
            # Resolve relative path (e.g. ../images/fig1.jpg)
            if src.startswith("../") or not src.startswith("/"):
                resolved = os.path.normpath(os.path.join(os.path.dirname(full_path), src))
            else:
                resolved = src.lstrip("/")
            # Strip fragment/query
            resolved = resolved.split("#")[0].split("?")[0]

            zip_match = zip_names_lower.get(resolved.lower())
            if zip_match:
                ext = os.path.splitext(zip_match)[1].lower()
                if ext in image_extensions:
                    try:
                        raw_bytes = zf.read(zip_match)
                        all_images.append({
                            "alt": img_info.get("alt", ""),
                            "zip_path": zip_match,
                            "chapter_idx": chapter_idx,
                            "raw_bytes": raw_bytes,
                            "ext": ext,
                        })
                    except Exception as e:
                        print(f"[extract-epub] Failed to read image {zip_match}: {e}")

        if text.strip():
            chapters.append(text.strip())

    full_text = "\n\n---\n\n".join(chapters)
    return full_text, len(chapters), all_images

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
    ".epub": "application/epub+zip",
    ".mobi": "application/x-mobipocket-ebook",
}

def _resolve_local_path(download_url: str) -> Optional[Path]:
    """
    If download_url points to this file-receiver instance, return the local
    file path on disk instead of making an HTTP round-trip.

    Recognises:
      - FILE_RECEIVER_URL env var (e.g. https://file-receiver.agentic.pl)
      - http://127.0.0.1:8000, http://localhost:8000
      - http://file-receiver-service:8000 (k8s in-cluster)
    """
    file_receiver_url = os.getenv("FILE_RECEIVER_URL", "").rstrip("/")
    self_prefixes = [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://file-receiver-service:8000",
        "http://file-receiver-service.bbx331.svc.cluster.local:8000",
    ]
    if file_receiver_url:
        self_prefixes.append(file_receiver_url)

    for prefix in self_prefixes:
        if download_url.startswith(prefix):
            # Extract path after the prefix, e.g. /files/tenant/kb/doc/file.pdf
            url_path = download_url[len(prefix):]
            # Remove query string (?download=true etc.)
            url_path = url_path.split("?")[0]
            # URL path starts with /files/ — map to STORAGE_ROOT
            if url_path.startswith("/files/"):
                relative = url_path[len("/files/"):]
                # URL-decode the path components (e.g. %20 → space)
                relative = urllib.parse.unquote(relative)
                local = Path(STORAGE_ROOT) / relative
                if local.exists() and local.is_file():
                    return local
            break  # matched a self-prefix but file not found locally
    return None

def _rewrite_to_localhost(download_url: str) -> str:
    """
    Rewrite an external file-receiver URL to http://127.0.0.1:8000 so the
    OCR worker can reach the file without going through external DNS/ingress.
    """
    file_receiver_url = os.getenv("FILE_RECEIVER_URL", "").rstrip("/")
    if file_receiver_url and download_url.startswith(file_receiver_url):
        return download_url.replace(file_receiver_url, "http://127.0.0.1:8000", 1)
    return download_url

# ══════════════════════════════════════════════════════════════
# AI Vision call with provider chain (Gemini → OpenAI fallback)
# ══════════════════════════════════════════════════════════════

async def _call_vision_with_fallback(
    messages: list,
    max_tokens: int = 8000,
    temperature: float = 0,
    force_qwen: bool = False,
    qwen_api_key: Optional[str] = None,
) -> dict:
    """
    Call a vision-capable LLM with automatic fallback.
    Provider chain: Qwen (if forced) → Gemini → OpenAI.
    Returns the parsed JSON response from the API.
    Raises on total failure.
    """
    providers = []

    if force_qwen and qwen_api_key:
        providers.append({
            "url": QWEN_API_URL,
            "key": qwen_api_key,
            "model": QWEN_MODEL,
            "name": "Qwen",
            "max_retries": 2,
        })
    else:
        if GEMINI_API_KEY:
            providers.append({
                "url": GEMINI_API_URL,
                "key": GEMINI_API_KEY,
                "model": GEMINI_MODEL,
                "name": "Gemini",
                "max_retries": 3,
            })
        if OPENAI_API_KEY:
            providers.append({
                "url": OPENAI_API_URL,
                "key": OPENAI_API_KEY,
                "model": OPENAI_MODEL,
                "name": "OpenAI",
                "max_retries": 2,
            })

    if not providers:
        raise ValueError("No AI API keys configured (GEMINI_API_KEY or OPENAI_API_KEY required)")

    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
    NON_RETRYABLE_STATUS_CODES = {400, 401, 403}

    last_error = None
    all_non_retryable = True  # track if every failure was non-retryable

    for provider in providers:
        for attempt in range(provider["max_retries"] + 1):
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.post(
                        provider["url"],
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {provider['key']}",
                        },
                        json={
                            "model": provider["model"],
                            "messages": messages,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                    )

                    if resp.status_code == 200:
                        if provider["name"] != "Gemini":
                            print(f"[vision] Used {provider['name']} (attempt {attempt + 1})")
                        return resp.json()

                    # Non-retryable client errors — skip to next provider immediately
                    if resp.status_code in NON_RETRYABLE_STATUS_CODES:
                        await resp.aread()
                        last_error = f"{provider['name']} {resp.status_code}"
                        print(f"[vision] {provider['name']} returned {resp.status_code} (non-retryable), trying next provider...")
                        break  # next provider

                    # Retryable errors (429, 5xx) — retry within this provider
                    if resp.status_code in RETRYABLE_STATUS_CODES:
                        all_non_retryable = False
                        if attempt < provider["max_retries"]:
                            wait = min(10 * (attempt + 1), 30)
                            print(f"[vision] {provider['name']} {resp.status_code}, retry {attempt + 1}/{provider['max_retries']} in {wait}s")
                            await resp.aread()
                            await asyncio.sleep(wait)
                            continue
                        else:
                            print(f"[vision] {provider['name']} exhausted retries ({resp.status_code}), trying next provider...")
                            await resp.aread()
                            last_error = f"{provider['name']} {resp.status_code}"
                            break  # next provider

                    # Unknown error code — treat as non-retryable
                    await resp.aread()
                    last_error = f"{provider['name']} {resp.status_code}"
                    print(f"[vision] {provider['name']} returned unexpected {resp.status_code}, trying next provider...")
                    break  # next provider

            except Exception as e:
                all_non_retryable = False
                last_error = f"{provider['name']}: {e}"
                if attempt < provider["max_retries"]:
                    await asyncio.sleep(5)
                else:
                    break  # next provider

    # If all providers failed with non-retryable errors, raise NonRetryableAPIError
    # so outer retry loops know not to retry
    if all_non_retryable:
        raise NonRetryableAPIError(f"All AI providers failed with non-retryable errors. Last: {last_error}")
    raise RuntimeError(f"All AI providers failed. Last error: {last_error}")

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
    """Call Vision API for a single page image OCR with Gemini→OpenAI fallback."""
    if extract_images:
        system_prompt = (
            "Jestes ekspertem od ekstrakcji tresci z dokumentow. Wykonaj DWA zadania:\n"
            "1. EKSTRAKCJA TEKSTU: Przepisz dokladnie CALY tekst w Markdown.\n"
            "2. DETEKCJA OBRAZOW: Zidentyfikuj elementy graficzne (zdjecia, wykresy, diagramy, schematy, logo). "
            "NIE oznaczaj tabel, naglowkow, stopek ani dekoracji jako obrazy.\n"
            f"Dla kazdego obrazu podaj typ, opis (max {image_desc_tokens} tokenow) "
            "oraz wspolrzedne bbox jako [y_min, x_min, y_max, x_max] w skali 0-1000 "
            "(0,0 = lewy gorny rog, 1000,1000 = prawy dolny rog). "
            "Pusta lista jesli brak obrazow.\n\n"
            'Odpowiedz WYLACZNIE JSON:\n'
            '{"text": "...", "images": [{"type": "...", "description": "...", "bbox": [y_min, x_min, y_max, x_max]}]}'
        )
    else:
        system_prompt = (
            "Przepisz dokladnie caly tekst widoczny na tym obrazie strony dokumentu. "
            "Zachowaj oryginalna strukture: naglowki, akapity, punkty, tabele w formacie Markdown.\n\n"
            'Odpowiedz WYLACZNIE JSON: {"text": "wyekstrahowany tekst..."}'
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Przeanalizuj te strone dokumentu:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            ],
        },
    ]

    result = await _call_vision_with_fallback(
        messages=messages,
        max_tokens=8000,
        temperature=0,
        force_qwen=(anonymize and bool(qwen_api_key)),
        qwen_api_key=qwen_api_key,
    )

    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Attempt to parse JSON response; fall back to raw text
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

async def describe_epub_image(
    image_base64: str,
    image_desc_tokens: int = 200,
    mime_type: str = "image/jpeg",
) -> Dict[str, str]:
    """
    Call Vision API to describe a single image extracted from an EPUB.
    Returns {"type": "...", "description": "..."}.
    Uses Gemini → OpenAI fallback chain.
    """
    system_prompt = (
        "Jestes ekspertem od analizy obrazow w dokumentach. "
        "Opisz dokladnie co przedstawia ten obraz.\n\n"

        "Jesli obraz zawiera tabele:\n"
        "- Jezeli cala tabela mozna przepisac w limicie tokenow, przepisz jej zawartosc w formacie Markdown.\n"
        "- Jezeli tabela jest zbyt duza i przepisanie jej przekroczyloby limit tokenow, NIE przepisuj jej. "
        "Zamiast tego opisz jej strukture, glowne kolumny, typ danych oraz kontekst.\n\n"

        "Jesli to wykres, diagram, schemat lub zdjecie — opisz jego zawartosc i kontekst.\n\n"

        f"Odpowiedz WYLACZNIE JSON (maksymalnie {image_desc_tokens} tokenow opisu):\n"
        '{"type": "photo|chart|diagram|table|schema|logo|illustration|other", "description": "..."}'
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Opisz ten obraz z dokumentu EPUB:"},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}},
            ],
        },
    ]

    result = await _call_vision_with_fallback(
        messages=messages,
        max_tokens=2000,
        temperature=0,
    )

    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

    try:
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        parsed = json.loads(cleaned.strip())
        return {
            "type": parsed.get("type", "unknown"),
            "description": parsed.get("description", ""),
        }
    except json.JSONDecodeError:
        return {"type": "unknown", "description": content}

async def _describe_image_with_retry(
    img_b64: str,
    image_desc_tokens: int,
    mime_type: str,
    img_idx: int,
) -> Dict[str, str]:
    """Describe a single image with up to 3 retries and rate-limit back-off."""
    for attempt in range(3):
        try:
            return await describe_epub_image(img_b64, image_desc_tokens, mime_type)
        except NonRetryableAPIError as exc:
            print(f"[epub-worker] Image {img_idx} failed (non-retryable): {exc}")
            return {"type": "unknown", "description": f"[DESCRIPTION ERROR: {exc}]"}
        except Exception as exc:
            if attempt < 2:
                wait = 5 * (attempt + 1)
                print(f"[epub-worker] Image {img_idx} attempt {attempt + 1} failed: {exc}, retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"[epub-worker] Image {img_idx} description failed after 3 attempts: {exc}")
                return {"type": "unknown", "description": f"[DESCRIPTION ERROR: {exc}]"}

    return {"type": "unknown", "description": "[DESCRIPTION ERROR]"}

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
        # Resolve tenant slug
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

        # Touch updated_at and return current status in one query
        resp = await client.post(
            f"{supabase_url}/rest/v1/rpc/execute_tenant_query",
            headers={
                "apikey": service_role_key,
                "Authorization": f"Bearer {service_role_key}",
                "Content-Type": "application/json",
            },
            json={
                "p_query": (
                    f"UPDATE {schema}.ingestions "
                    f"SET updated_at = now() "
                    f"WHERE id = '{ingestion_id}' "
                    f"RETURNING status"
                )
            },
        )
        rows = resp.json()
        if isinstance(rows, list) and rows:
            return rows[0].get("status", "processing")
        return "processing"

async def _get_tenant_slug(job: OcrJobRequest) -> Optional[str]:
    """Resolve tenant_id to slug via Supabase REST."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{job.supabase_url}/rest/v1/tenants?id=eq.{job.tenant_id}&select=slug",
                headers={
                    "apikey": job.supabase_service_role_key,
                    "Authorization": f"Bearer {job.supabase_service_role_key}",
                },
            )
            tenants = resp.json()
            if tenants:
                return tenants[0]["slug"]
    except Exception as e:
        print(f"[ocr-worker] Failed to get tenant slug: {e}")
    return None

async def _ocr_page_with_retry(
    img_b64: str,
    job: OcrJobRequest,
    extract_images: bool,
    image_desc_tokens: int,
    page_num: int,
) -> Dict[str, Any]:
    """OCR a single page image with up to 3 retries and rate-limit back-off."""
    for attempt in range(3):
        try:
            return await ocr_call_gemini(
                img_b64,
                job.lovable_api_key,
                anonymize=job.anonymize,
                qwen_api_key=job.qwen_api_key,
                extract_images=extract_images,
                image_desc_tokens=image_desc_tokens,
            )
        except NonRetryableAPIError as exc:
            print(f"[ocr-worker] Page {page_num} failed (non-retryable): {exc}")
            return {"text": f"[OCR ERROR: {exc}]", "images": []}
        except Exception as exc:
            if attempt < 2:
                wait = 5 * (attempt + 1)
                print(f"[ocr-worker] Page {page_num} attempt {attempt + 1} failed: {exc}, retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"[ocr-worker] Page {page_num} failed after 3 attempts: {exc}")
                return {"text": f"[OCR ERROR: {exc}]", "images": []}

    return {"text": "[OCR ERROR]", "images": []}

async def ocr_worker_process(job: OcrJobRequest):
    """
    Background task: OCR a large PDF page by page without loading the whole
    document into RAM at once.

    Memory strategy
    ───────────────
    1. Stream-download (or decode) the PDF directly to a temp file on disk —
       never hold the full byte string in memory alongside the image data.
    2. Use pdfinfo_from_path to learn the page count cheaply (no rasterisation).
    3. Rasterise only PDF_CHUNK_SIZE pages at a time with convert_from_path,
       then discard those PIL Images before converting the next chunk.

    Peak RAM per chunk:  PDF_CHUNK_SIZE × DPI² × channels ≈ 20 × 6.5 MB ≈ 130 MB
    vs. naive approach:  2 000 pages × 6.5 MB                            ≈ 13 GB

    4. Both pdfinfo_from_path and convert_from_path run in a thread-pool executor
       via asyncio.get_running_loop().run_in_executor() so that the synchronous
       poppler subprocess calls never block the event loop. Multiple concurrent
       OCR jobs can therefore make progress during each other's rasterisation steps.
    5. functools.partial is used instead of lambda so that PyCharm's type checker
       does not raise "Parameter 'args' unfilled" on run_in_executor calls.
    6. OCR results are written incrementally to JSONL files on disk — no large
       list accumulation in RAM across thousands of pages.
    7. The callback sends only a lightweight URL reference (_ocr_result_url) instead
       of the full document text in the request body, avoiding Edge Function size limits.
    8. Temp PDF file is always removed in the `finally` block, even on cancellation
       or unhandled exceptions, preventing /tmp from filling up.
    """
    job_id = f"{job.ingestion_id}_{job.file_index}"
    ocr_jobs[job_id] = {
        "status": "processing",
        "pages_done": 0,
        "total_pages": job.estimated_pages,
        "started_at": datetime.utcnow().isoformat(),
        "error": None,
    }

    # Temp file path — declared before try so finally can always clean up
    tmp_path: Optional[str] = None

    try:
        # Import here so startup does not fail if pdf2image is somehow missing
        from pdf2image import convert_from_path, pdfinfo_from_path

        print(f"[ocr-worker] Starting job {job_id}: {job.file_name}")

        loop = asyncio.get_running_loop()

        # ── Step 1: Write PDF to disk ──────────────────────────────────────────
        tmp_path = f"/tmp/ocr_{job_id}_{uuid.uuid4().hex}.pdf"

        if job.file_download_url:
            local_path = _resolve_local_path(job.file_download_url)
            if local_path:
                print(f"[ocr-worker] Local file found → copying {local_path} → {tmp_path}")
                shutil.copy2(str(local_path), tmp_path)
            else:
                actual_url = _rewrite_to_localhost(job.file_download_url)
                if actual_url != job.file_download_url:
                    print(f"[ocr-worker] Rewritten URL: {job.file_download_url} → {actual_url}")

                print(f"[ocr-worker] Streaming download → {tmp_path}")
                async with httpx.AsyncClient(timeout=300) as client:
                    async with client.stream("GET", actual_url, headers={
                        "X-Api-Key": API_KEY,
                    }) as resp:
                        resp.raise_for_status()
                        with open(tmp_path, "wb") as fh:
                            async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                                fh.write(chunk)

        elif job.file_base64:
            print(f"[ocr-worker] Decoding base64 → {tmp_path}")
            pdf_bytes = base64.b64decode(job.file_base64)
            with open(tmp_path, "wb") as fh:
                fh.write(pdf_bytes)
            del pdf_bytes

        else:
            raise ValueError("No file source provided (file_download_url or file_base64 required)")

        file_size_mb = os.path.getsize(tmp_path) / 1024 / 1024
        print(f"[ocr-worker] PDF on disk: {file_size_mb:.1f} MB → {tmp_path}")

        # ── Step 2: Determine page count without loading any images ───────────
        info = await loop.run_in_executor(
            None,
            functools.partial(pdfinfo_from_path, tmp_path),
        )
        total_pages = info["Pages"]
        ocr_jobs[job_id]["total_pages"] = total_pages
        print(f"[ocr-worker] Total pages: {total_pages}")

        # ── Step 3: Prepare incremental JSONL output files on disk ────────────
        tenant_slug = await _get_tenant_slug(job)
        if not tenant_slug:
            raise ValueError("Could not resolve tenant slug")

        ocr_dir = Path(STORAGE_ROOT) / tenant_slug / job.knowledge_base_id / (job.file_id or "unknown")
        ocr_dir.mkdir(parents=True, exist_ok=True)

        texts_jsonl = ocr_dir / "_ocr_texts.jsonl"
        images_jsonl = ocr_dir / "_ocr_images.jsonl"

        for f in [texts_jsonl, images_jsonl]:
            if f.exists():
                f.unlink()

        extract_images = job.settings.get("extractImages", False)
        image_desc_tokens = job.settings.get("imageDescTokens", 200)
        pages_done = 0

        # ── Token usage estimation ────────────────────────────────────────
        # Each page OCR: ~765 input tokens (image) + ~200 (prompt) = ~965 input
        # Output: page text ~3000 chars ≈ 750 tokens
        # Image descriptions add ~200 tokens output each
        EST_INPUT_TOKENS_PER_PAGE = 965
        EST_OUTPUT_TOKENS_PER_PAGE = 750
        EST_OUTPUT_TOKENS_PER_IMAGE_DESC = 200
        total_input_tokens = 0
        total_output_tokens = 0

        pages_dir = ocr_dir / "_images"
        if extract_images:
            pages_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 4: Rasterise and OCR in chunks ────────────────────────────────
        for chunk_start_0 in range(0, total_pages, PDF_CHUNK_SIZE):
            chunk_end_0 = min(chunk_start_0 + PDF_CHUNK_SIZE, total_pages)

            first_page = chunk_start_0 + 1
            last_page = chunk_end_0

            print(
                f"[ocr-worker] Rasterising pages {first_page}–{last_page} "
                f"of {total_pages} (DPI={PDF_DPI})..."
            )

            chunk_images = await loop.run_in_executor(
                None,
                functools.partial(
                    convert_from_path,
                    tmp_path,
                    dpi=PDF_DPI,
                    fmt="jpeg",
                    first_page=first_page,
                    last_page=last_page,
                ),
            )

            for batch_start in range(0, len(chunk_images), OCR_PAGE_CONCURRENCY):
                batch_end = min(batch_start + OCR_PAGE_CONCURRENCY, len(chunk_images))

                if pages_done > 0 and pages_done % OCR_HEARTBEAT_INTERVAL == 0:
                    try:
                        status = await update_heartbeat(
                            job.supabase_url,
                            job.supabase_service_role_key,
                            job.tenant_id,
                            job.ingestion_id,
                            pages_done,
                            total_pages,
                        )
                        if status in ("cancelled", "paused"):
                            print(f"[ocr-worker] Job {job_id} {status} by user, stopping.")
                            ocr_jobs[job_id]["status"] = status
                            texts_jsonl.unlink(missing_ok=True)
                            images_jsonl.unlink(missing_ok=True)
                            return
                    except Exception as hb_err:
                        print(f"[ocr-worker] Heartbeat error: {hb_err}")

                async def process_page(local_idx: int) -> Dict:
                    """Encode one PIL Image to JPEG/base64, OCR it, crop detected images."""
                    global_page_num = chunk_start_0 + local_idx + 1
                    img = chunk_images[local_idx]
                    img_width, img_height = img.size

                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    jpeg_bytes = buf.getvalue()
                    img_b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
                    buf.close()
                    del jpeg_bytes

                    result = await _ocr_page_with_retry(
                        img_b64, job, extract_images, image_desc_tokens, global_page_num,
                    )

                    cropped_images = []
                    for img_idx, img_info in enumerate(result.get("images", [])):
                        bbox = img_info.get("bbox")
                        if not bbox or len(bbox) != 4:
                            cropped_images.append(None)
                            continue

                        try:
                            y_min, x_min, y_max, x_max = bbox

                            pad_x = int((x_max - x_min) * 0.03)
                            pad_y = int((y_max - y_min) * 0.03)
                            x_min = max(0, x_min - pad_x)
                            y_min = max(0, y_min - pad_y)
                            x_max = min(1000, x_max + pad_x)
                            y_max = min(1000, y_max + pad_y)

                            left = int(x_min / 1000 * img_width)
                            upper = int(y_min / 1000 * img_height)
                            right = int(x_max / 1000 * img_width)
                            lower = int(y_max / 1000 * img_height)

                            if (right - left) < 20 or (lower - upper) < 20:
                                print(f"[ocr-worker] Page {global_page_num} img {img_idx}: bbox too small, skipping crop")
                                cropped_images.append(None)
                                continue

                            cropped = img.crop((left, upper, right, lower))
                            crop_filename = f"page_{global_page_num:04d}_img_{img_idx:02d}.jpg"
                            crop_path = pages_dir / crop_filename
                            cropped.save(str(crop_path), format="JPEG", quality=90)
                            cropped_images.append(crop_filename)
                        except Exception as crop_err:
                            print(f"[ocr-worker] Page {global_page_num} img {img_idx}: crop failed: {crop_err}")
                            cropped_images.append(None)

                    return {
                        "page_num": global_page_num,
                        "result": result,
                        "cropped_images": cropped_images,
                    }

                tasks = [process_page(idx) for idx in range(batch_start, batch_end)]
                results = await asyncio.gather(*tasks)

                for r in results:
                    page_num = r["page_num"]
                    text = r["result"].get("text", "")
                    cropped = r.get("cropped_images", [])
                    pages_done += 1
                    # Accumulate estimated token usage
                    total_input_tokens += EST_INPUT_TOKENS_PER_PAGE
                    page_output_est = max(EST_OUTPUT_TOKENS_PER_PAGE, len(text) // 4)
                    num_images_detected = len(r["result"].get("images", []))
                    total_output_tokens += page_output_est + (num_images_detected * EST_OUTPUT_TOKENS_PER_IMAGE_DESC)

                    with open(texts_jsonl, "a", encoding="utf-8") as fh:
                        fh.write(
                            json.dumps({"page": page_num, "text": text}, ensure_ascii=False) + "\n"
                        )

                    for img_idx, img_info in enumerate(r["result"].get("images", [])):
                        crop_filename = cropped[img_idx] if img_idx < len(cropped) else None
                        img_entry = {
                            "description": (
                                f"[Image: {img_info.get('type', 'unknown')}, "
                                f"page {page_num}] {img_info.get('description', '')}"
                            ),
                            "section_context": text[:1000] if text else "",
                            "page": page_num,
                            "bbox": img_info.get("bbox"),
                        }
                        if crop_filename:
                            img_entry["image_url"] = (
                                f"/files/{tenant_slug}/{job.knowledge_base_id}/"
                                f"{job.file_id or 'unknown'}/_images/{crop_filename}"
                            )
                        with open(images_jsonl, "a", encoding="utf-8") as fh:
                            fh.write(
                                json.dumps(img_entry, ensure_ascii=False) + "\n"
                            )

                ocr_jobs[job_id]["pages_done"] = pages_done
                print(f"[ocr-worker] Progress: {pages_done}/{total_pages} pages")

                if batch_end < len(chunk_images):
                    await asyncio.sleep(0.5)

            del chunk_images

        # ── Step 5: Assemble final result from JSONL files ────────────────────
        print("[ocr-worker] Assembling OCR result from JSONL files...")

        page_texts: Dict[int, str] = {}
        with open(texts_jsonl, "r", encoding="utf-8") as fh:
            for line in fh:
                entry = json.loads(line)
                page_texts[entry["page"]] = entry["text"]

        sorted_texts = [page_texts.get(p, "") for p in sorted(page_texts.keys())]
        full_text = "\n\n---\n\n".join(t for t in sorted_texts if t)

        all_images = []
        if images_jsonl.exists():
            with open(images_jsonl, "r", encoding="utf-8") as fh:
                for line in fh:
                    all_images.append(json.loads(line))

        print(
            f"[ocr-worker] OCR complete: {len(full_text)} chars, "
            f"{len(all_images)} images from {total_pages} pages, "
            f"est. tokens: {total_input_tokens} in / {total_output_tokens} out"
        )

        cropped_count = 0
        if extract_images and pages_dir.exists():
            cropped_count = len(list(pages_dir.glob("page_*_img_*.jpg")))
            print(f"[ocr-worker] Saved {cropped_count} cropped images to {pages_dir}")

        ocr_result_file = ocr_dir / "_ocr_result.json"
        with open(ocr_result_file, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "document_text": full_text,
                    "_ocr_images": all_images,
                    "_ocr_page_count": total_pages,
                    "_cropped_images_count": cropped_count,
                },
                fh,
                ensure_ascii=False,
            )

        result_size_mb = ocr_result_file.stat().st_size / 1024 / 1024
        print(f"[ocr-worker] OCR result saved: {ocr_result_file} ({result_size_mb:.1f} MB)")

        texts_jsonl.unlink(missing_ok=True)
        images_jsonl.unlink(missing_ok=True)

        # ── Step 6: Lightweight callback — URL reference only ─────────────────
        file_receiver_base = os.getenv("FILE_RECEIVER_URL", "https://file-receiver.agentic.pl").rstrip("/")
        ocr_result_url = (
            f"{file_receiver_base}/files/{tenant_slug}/"
            f"{job.knowledge_base_id}/{job.file_id or 'unknown'}/_ocr_result.json"
            f"?download=true"
        )

        print(f"[ocr-worker] Calling back pipeline at {job.callback_url}")
        callback_body = {
            "file_name": job.file_name,
            "file_id": job.file_id,
            "knowledge_base_id": job.knowledge_base_id,
            "settings": job.settings,
            "tenant_id": job.tenant_id,
            "ingestion_id": job.ingestion_id,
            "file_index": job.file_index,
            "file_path": job.file_path,
            "file_source_type": job.file_source_type,
            "file_user_id": job.file_user_id,
            "auth_token": job.auth_token,
            "_ocr_result_url": ocr_result_url,
            "_ocr_page_count": total_pages,
            "_ocr_token_usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "model": GEMINI_MODEL,
            },
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

        # Best-effort: mark the file as errored in the DB and trigger next file
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

                    resp2 = await client.post(
                        f"{job.supabase_url}/rest/v1/rpc/execute_tenant_query",
                        headers={
                            "apikey": job.supabase_service_role_key,
                            "Authorization": f"Bearer {job.supabase_service_role_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "p_query": (
                                f"SELECT pending_files FROM {schema}.ingestions "
                                f"WHERE id = '{job.ingestion_id}'"
                            )
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
                                    "p_query": (
                                        f"UPDATE {schema}.ingestions "
                                        f"SET pending_files = '{pf_json}'::jsonb, "
                                        f"processed_count = {pc}, "
                                        f"error_count = {ec}, "
                                        f"updated_at = now() "
                                        f"WHERE id = '{job.ingestion_id}' RETURNING id"
                                    )
                                },
                            )

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
                                        "p_query": (
                                            f"UPDATE {schema}.ingestions "
                                            f"SET status = '{final_status}', "
                                            f"updated_at = now() "
                                            f"WHERE id = '{job.ingestion_id}' RETURNING id"
                                        )
                                    },
                                )
        except Exception as db_err:
            print(f"[ocr-worker] Error updating DB on failure: {db_err}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                print(f"[ocr-worker] Temp file removed: {tmp_path}")
            except Exception as cleanup_err:
                print(f"[ocr-worker] Failed to remove temp file {tmp_path}: {cleanup_err}")

# ── Endpoints ──────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    storage_ok = Path(STORAGE_ROOT).exists()
    poppler_ok = shutil.which("pdftoppm") is not None

    root = Path(STORAGE_ROOT)
    tenant_count = sum(1 for d in root.iterdir() if d.is_dir()) if root.exists() else 0

    active_jobs = sum(1 for j in ocr_jobs.values() if j["status"] == "processing")
    active_epub = sum(1 for j in epub_jobs.values() if j["status"] == "processing")

    return {
        "status": "ok" if storage_ok else "degraded",
        "storage_root": STORAGE_ROOT,
        "storage_accessible": storage_ok,
        "poppler_available": poppler_ok,
        "tenant_count": tenant_count,
        "active_ocr_jobs": active_jobs,
        "active_epub_jobs": active_epub,
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
    The worker streams the file to disk, processes pages in chunks to keep
    RAM bounded, updates DB progress, and calls back the pipeline when done.
    """
    verify_api_key(x_api_key)

    job_id = f"{req.ingestion_id}_{req.file_index}"

    if job_id in ocr_jobs and ocr_jobs[job_id]["status"] == "processing":
        return JSONResponse(status_code=409, content={
            "error": "Job already running",
            "job_id": job_id,
            "status": ocr_jobs[job_id],
        })

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

@app.get("/files/{tenant_slug}/{kb_id}/{doc_id}/{filename:path}")
async def download_file(
    tenant_slug: str,
    kb_id: str,
    doc_id: str,
    filename: str,
    x_api_key: str = Header(...),
    x_tenant: Optional[str] = Header(None),
    download: bool = Query(False, description="Force download instead of inline display"),
):
    """Serve/download a specific file. Supports nested paths (e.g. _images/page_0001_img_00.jpg)."""
    verify_api_key(x_api_key)
    verify_tenant(tenant_slug, x_tenant)

    file_path = Path(STORAGE_ROOT) / tenant_slug / kb_id / doc_id / filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if not str(file_path.resolve()).startswith(str(Path(STORAGE_ROOT).resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    display_name = Path(filename).name
    ext = file_path.suffix.lower()
    media_type = MIME_MAP.get(ext, "application/octet-stream")
    disposition = "attachment" if download else "inline"

    ascii_filename = display_name.encode("ascii", "ignore").decode("ascii").strip() or "download"
    utf8_filename = urllib.parse.quote(display_name)

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        headers={
            "Content-Disposition": (
                f'{disposition}; filename="{ascii_filename}"; '
                f"filename*=UTF-8''{utf8_filename}"
            )
        },
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

@app.delete("/files/{tenant_slug}/{kb_id}/{doc_id}/{filename:path}")
async def delete_single_file(
    tenant_slug: str,
    kb_id: str,
    doc_id: str,
    filename: str,
    x_api_key: str = Header(...),
    x_tenant: Optional[str] = Header(None),
):
    """Delete a single file from a document directory. Supports nested paths."""
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
    Note: intended for small/partial PDFs only — no chunking applied here.
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

# ── EPUB extraction (Background Worker) ───────────────────────

EPUB_IMAGE_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
}

async def epub_worker_process(job: EpubExtractRequest):
    """
    Background task: Extract text and images from an EPUB file.

    1. Download/decode EPUB to bytes
    2. Parse ZIP → extract text (HTMLParser) and raw images
    3. Save images to disk
    4. Send each image to Gemini/OpenAI for description (with concurrency + retries)
    5. Save _epub_result.json to disk
    6. Lightweight callback with URL reference (like OCR worker)
    """
    job_id = f"{job.ingestion_id}_{job.file_index}"
    epub_jobs[job_id] = {
        "status": "processing",
        "images_done": 0,
        "total_images": 0,
        "started_at": datetime.utcnow().isoformat(),
        "error": None,
    }

    try:
        print(f"[epub-worker] Starting job {job_id}: {job.file_name}")

        # ── Step 1: Get EPUB bytes ─────────────────────────────────────────
        epub_bytes: Optional[bytes] = None

        if job.file_download_url:
            local_path = _resolve_local_path(job.file_download_url)
            if local_path:
                print(f"[epub-worker] Local file found: {local_path}")
                epub_bytes = local_path.read_bytes()
            else:
                actual_url = _rewrite_to_localhost(job.file_download_url)
                if actual_url != job.file_download_url:
                    print(f"[epub-worker] Rewritten URL: {job.file_download_url} → {actual_url}")
                print(f"[epub-worker] Downloading: {actual_url}")
                async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.get(actual_url, headers={"X-Api-Key": API_KEY})
                    resp.raise_for_status()
                    epub_bytes = resp.content
        elif job.file_base64:
            epub_bytes = base64.b64decode(job.file_base64)
        else:
            raise ValueError("No file source provided (file_download_url or file_base64 required)")

        print(f"[epub-worker] EPUB size: {len(epub_bytes) / 1024:.1f} KB")

        # ── Step 2: Parse EPUB ─────────────────────────────────────────────
        text, chapter_count, extracted_images = _parse_epub_bytes(epub_bytes)
        del epub_bytes  # free RAM

        total_images = len(extracted_images)
        epub_jobs[job_id]["total_images"] = total_images
        print(f"[epub-worker] Parsed: {len(text)} chars, {chapter_count} chapters, {total_images} images")

        # ── Step 3: Save images to disk + describe with AI ─────────────────
        file_receiver_base = os.getenv(
            "FILE_RECEIVER_URL", "https://file-receiver.agentic.pl"
        ).rstrip("/")
        doc_path = get_doc_path(job.tenant_slug, job.knowledge_base_id, job.doc_id)
        images_jsonl = doc_path / "_epub_images.jsonl"
        if images_jsonl.exists():
            images_jsonl.unlink()

        image_desc_tokens = job.image_desc_tokens
        images_done = 0

        for batch_start in range(0, total_images, OCR_PAGE_CONCURRENCY):
            batch_end = min(batch_start + OCR_PAGE_CONCURRENCY, total_images)

            async def process_image(idx: int) -> dict:
                img = extracted_images[idx]
                raw_bytes = img["raw_bytes"]
                ext = img.get("ext", ".jpg")
                alt = img.get("alt", "")
                img_filename = f"epub_image_{idx}{ext}"
                mime_type = EPUB_IMAGE_MIME.get(ext, "image/jpeg")

                # Save image file to disk
                img_url = None
                img_path = doc_path / img_filename
                try:
                    img_path.write_bytes(raw_bytes)
                    img_url = (
                        f"{file_receiver_base}/files/{job.tenant_slug}/"
                        f"{job.knowledge_base_id}/{job.doc_id}/{img_filename}"
                    )
                except Exception as save_err:
                    print(f"[epub-worker] Failed to save image {img_filename}: {save_err}")

                # Describe with AI (Gemini → OpenAI fallback)
                img_b64 = base64.b64encode(raw_bytes).decode("ascii")
                desc = await _describe_image_with_retry(
                    img_b64, image_desc_tokens, mime_type, idx,
                )

                return {
                    "idx": idx,
                    "img_url": img_url,
                    "description": desc,
                    "alt": alt,
                    "zip_path": img.get("zip_path", ""),
                    "chapter_idx": img.get("chapter_idx", 0),
                }

            tasks = [process_image(i) for i in range(batch_start, batch_end)]
            results = await asyncio.gather(*tasks)

            for r in results:
                images_done += 1
                img_entry = {
                    "description": (
                        f"[Image: {r['description'].get('type', 'unknown')}, "
                        f"chapter {r['chapter_idx']}] "
                        f"{r['description'].get('description', r['alt'])}"
                    ),
                    "section_context": "",
                    "page": r["chapter_idx"],
                    "image_url": r["img_url"],
                }
                with open(images_jsonl, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(img_entry, ensure_ascii=False) + "\n")

                # ── Heartbeat: check if job was cancelled ──────────────
                if images_done > 0 and images_done % OCR_HEARTBEAT_INTERVAL == 0:
                    try:
                        status = await update_heartbeat(
                            job.supabase_url,
                            job.supabase_service_role_key,
                            job.tenant_id,
                            job.ingestion_id,
                            images_done,
                            total_images,
                        )
                        if status in ("cancelled", "paused"):
                            print(f"[epub-worker] Job {job_id} {status} by user, stopping.")
                            epub_jobs[job_id]["status"] = status
                            images_jsonl.unlink(missing_ok=True)
                            return
                    except Exception as hb_err:
                        print(f"[epub-worker] Heartbeat error: {hb_err}")

            epub_jobs[job_id]["images_done"] = images_done
            print(f"[epub-worker] Image progress: {images_done}/{total_images}")

            # Small back-off between batches to respect API rate limits
            if batch_end < total_images:
                await asyncio.sleep(0.5)

        # ── Step 4: Assemble and save result ───────────────────────────────
        all_images = []
        if images_jsonl.exists():
            with open(images_jsonl, "r", encoding="utf-8") as fh:
                for line in fh:
                    all_images.append(json.loads(line))
            images_jsonl.unlink(missing_ok=True)

        result_file = doc_path / "_epub_result.json"
        with open(result_file, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "document_text": text,
                    "_epub_images": all_images,
                    "_epub_chapter_count": chapter_count,
                    "_epub_image_count": len(all_images),
                },
                fh,
                ensure_ascii=False,
            )

        result_size_mb = result_file.stat().st_size / 1024 / 1024
        print(f"[epub-worker] Result saved: {result_file} ({result_size_mb:.1f} MB)")

        # ── Step 5: Lightweight callback ───────────────────────────────────
        epub_result_url = (
            f"{file_receiver_base}/files/{job.tenant_slug}/"
            f"{job.knowledge_base_id}/{job.doc_id}/_epub_result.json"
            f"?download=true"
        )

        print(f"[epub-worker] Calling back pipeline at {job.callback_url}")
        callback_body = {
            "file_name": job.file_name,
            "file_id": job.file_id,
            "knowledge_base_id": job.knowledge_base_id,
            "settings": job.settings,
            "tenant_id": job.tenant_id,
            "ingestion_id": job.ingestion_id,
            "file_index": job.file_index,
            "file_path": job.file_path,
            "file_source_type": job.file_source_type,
            "file_user_id": job.file_user_id,
            "auth_token": job.auth_token,
            "_epub_result_url": epub_result_url,
            "_epub_chapter_count": chapter_count,
            "_epub_token_usage": {
                "input_tokens": total_images * 965,  # ~965 input tokens per image description call
                "output_tokens": total_images * 200,  # ~200 output tokens per description
                "model": GEMINI_MODEL,
            },
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
                print(f"[epub-worker] Callback failed: {resp.status_code} {resp.text[:500]}")
            else:
                print(f"[epub-worker] Callback success: {resp.status_code}")

        epub_jobs[job_id]["status"] = "completed"
        epub_jobs[job_id]["images_done"] = total_images

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[epub-worker] Job {job_id} failed: {error_msg}")
        print(traceback.format_exc())
        epub_jobs[job_id]["status"] = "failed"
        epub_jobs[job_id]["error"] = error_msg

        # Best-effort: mark the file as errored in the DB and trigger next file
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

                    resp2 = await client.post(
                        f"{job.supabase_url}/rest/v1/rpc/execute_tenant_query",
                        headers={
                            "apikey": job.supabase_service_role_key,
                            "Authorization": f"Bearer {job.supabase_service_role_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "p_query": (
                                f"SELECT pending_files FROM {schema}.ingestions "
                                f"WHERE id = '{job.ingestion_id}'"
                            )
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
                                    "p_query": (
                                        f"UPDATE {schema}.ingestions "
                                        f"SET pending_files = '{pf_json}'::jsonb, "
                                        f"processed_count = {pc}, "
                                        f"error_count = {ec}, "
                                        f"updated_at = now() "
                                        f"WHERE id = '{job.ingestion_id}' RETURNING id"
                                    )
                                },
                            )

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
                                        "p_query": (
                                            f"UPDATE {schema}.ingestions "
                                            f"SET status = '{final_status}', "
                                            f"updated_at = now() "
                                            f"WHERE id = '{job.ingestion_id}' RETURNING id"
                                        )
                                    },
                                )
        except Exception as db_err:
            print(f"[epub-worker] Error updating DB on failure: {db_err}")

@app.post("/extract-epub")
async def submit_epub_job(
    req: EpubExtractRequest,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(...),
):
    """
    Submit an EPUB file for background extraction.
    The worker extracts text and images, describes each image with AI,
    saves results to disk, and calls back the pipeline when done.
    """
    verify_api_key(x_api_key)

    job_id = f"{req.ingestion_id}_{req.file_index}"

    if job_id in epub_jobs and epub_jobs[job_id]["status"] == "processing":
        return JSONResponse(status_code=409, content={
            "error": "Job already running",
            "job_id": job_id,
            "status": epub_jobs[job_id],
        })

    background_tasks.add_task(epub_worker_process, req)

    return JSONResponse(status_code=202, content={
        "job_id": job_id,
        "message": f"EPUB extraction job accepted for {req.file_name}",
        "status": "accepted",
    })

@app.get("/epub-job/{ingestion_id}/{file_index}")
async def get_epub_job_status(
    ingestion_id: str,
    file_index: int,
    x_api_key: str = Header(...),
):
    """Check status of an EPUB extraction job."""
    verify_api_key(x_api_key)

    job_id = f"{ingestion_id}_{file_index}"
    if job_id not in epub_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return epub_jobs[job_id]

@app.get("/epub-jobs")
async def list_epub_jobs(x_api_key: str = Header(...)):
    """List all EPUB extraction jobs."""
    verify_api_key(x_api_key)
    return {
        "jobs": epub_jobs,
        "total": len(epub_jobs),
        "active": sum(1 for j in epub_jobs.values() if j["status"] == "processing"),
    }