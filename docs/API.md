# API guide

Install and start the loopback service:

```bash
python -m pip install 'steganography-dfir[api]'
steganography serve --host 127.0.0.1 --port 8000
```

## Compatible v1

- `GET /healthz`
- `GET /v1/modules`
- `POST /v1/analyze` — one multipart upload, synchronous result
- `POST /v1/scans` — multipart batch, persistent background job
- `GET /v1/scans/{job_id}`
- `DELETE /v1/scans/{job_id}`

## Authenticated v2

- `POST /v2/session`, `DELETE /v2/session`, `GET /v2/health`
- `GET /v2/vault`, `POST /v2/vault/initialize|unlock|lock`
- `/v2/cases` and `/v2/cases/{case_id}/evidence`
- `/v2/cases/{case_id}/scans`
- `/v2/jobs/{id}` and `/v2/jobs/{id}/events`
- `/v2/reports/{id}` with JSON, HTML, or SARIF format
- `/v2/studio/capacity|embed|extract`
- `/v2/artifacts/{id}/download`
- `/v2/models` and `/v2/audit/verify`

On local UI startup, a mode-0600 API token is created at
`$STEGANO_STATE_DIR/v2-api.key`. Browser login exchanges it for an HttpOnly,
SameSite session; cookie-authenticated writes also require the CSRF token.
Bearer clients send `Authorization: Bearer <token>`.

Uploaded evidence is streamed into the encrypted vault; evidence endpoints do
not accept server-side paths. Temporary plaintext is bounded to the operation.
Jobs and result JSON live in SQLite. Defaults limit file size, batch size,
worker count, and retention; `.env.example` documents overrides.

Binding outside loopback is refused without `STEGANO_API_KEY`. CORS is disabled.
An internet-facing deployment still needs TLS termination, a hardened reverse
proxy, network controls, authentication policy, backups, and monitoring.

AI sends locally generated signal text only by default. Image upload requires
both request opt-in and `STEGANO_ALLOW_AI_FILE_UPLOAD=1` on the server.

Python callers that do not need HTTP should use the service layer directly:

```python
from pathlib import Path

from core.workspace import create_workspace

workspace = create_workspace(Path("./state"))
case = workspace.cases.create_case("Example")
```
