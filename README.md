# BlueBox File Receiver v2.0

Self-hosted, tenant-isolated file storage for the BlueBox platform.
Receives files from any source (OneDrive, EZD RP, manual upload) and serves them for chat knowledge panel.

## Architecture

```
OneDrive ─┐
EZD RP  ──┤──> Edge Function ──> File Receiver (this service) ──> Disk
Upload  ──┘                                                        │
                                                                   ↓
Chat "Wiedza" panel  ←── GET /files/...  ←── Neo4j (file_url) ←── Ingest Pipeline
```

## Storage Layout

```
/data/files/
  └── {tenant_slug}/
        └── {kb_id}/
              └── {doc_id}/
                    ├── original-document.pdf
                    └── metadata.json
```

Each tenant has fully isolated storage. Cross-tenant access is blocked.

## Quick Start

```bash
# 1. Copy env file
cp .env.example .env

# 2. Set your API key (must match Lovable Cloud secret FILE_RECEIVER_API_KEY)
nano .env

# 3. Create storage directory
sudo mkdir -p /srv/bluebox/files

# 4. Build & start
docker compose up -d --build

# 5. Verify
curl http://localhost:8000/health
```

## API Endpoints

All endpoints (except `/health`) require `X-Api-Key` header.
Optional `X-Tenant` header enforces tenant path matching.

### Upload

```bash
# Upload with explicit doc_id
curl -X POST "http://localhost:8000/upload/{tenant}/{kb_id}/{doc_id}?source=onedrive" \
  -H "X-Api-Key: YOUR_KEY" \
  -F "file=@document.pdf"
```

### Download / View

```bash
# Open inline in browser (PDF viewer, image viewer, etc.)
curl "http://localhost:8000/files/{tenant}/{kb_id}/{doc_id}/{filename}" \
  -H "X-Api-Key: YOUR_KEY"

# Force download
curl "http://localhost:8000/files/{tenant}/{kb_id}/{doc_id}/{filename}?download=true" \
  -H "X-Api-Key: YOUR_KEY" -o document.pdf
```

### List & Delete

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/files/{tenant}/{kb_id}` | List all documents in KB |
| `GET` | `/files/{tenant}/{kb_id}/{doc_id}` | List files in a document |
| `DELETE` | `/files/{tenant}/{kb_id}/{doc_id}` | Delete entire document |
| `DELETE` | `/files/{tenant}/{kb_id}/{doc_id}/{filename}` | Delete single file |
| `GET` | `/tenants/{tenant}/stats` | Tenant storage stats |

### PDF to Images

```bash
curl -X POST "http://localhost:8000/pdf-to-images" \
  -H "X-Api-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"pdf_base64": "...", "dpi": 200}'
```

## Security

- **API Key**: Every request validated against `FILE_RECEIVER_API_KEY`
- **Tenant isolation**: `X-Tenant` header cross-checked with URL path
- **Path traversal**: All file paths resolved and validated against `STORAGE_ROOT`
- **HTTPS**: Use nginx/caddy reverse proxy in production

## Production Notes

- Metadata stored as `metadata.json` per document (no external DB needed)
- Set up HTTPS via reverse proxy (nginx/caddy)
- Configure max file size via `MAX_FILE_SIZE_MB` (default: 500MB)
- PDF DPI for image conversion via `PDF_DPI` (default: 150)
