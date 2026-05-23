# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities to the repository maintainers via GitHub's private vulnerability reporting:
https://github.com/WYZAAACCC/textpatch-studio/security/advisories/new

## Deployment Security

### Default Safe Configuration

- The application listens on `127.0.0.1` (localhost) by default
- CORS is restricted to local origins
- Remote font download is disabled by default
- No API key is required when running locally

### Public Deployment

When deploying on a public network:

```bash
# Enable authentication
export TEXTPATCH_REQUIRE_AUTH=true
export TEXTPATCH_API_TOKEN=<your-random-token>

# Set allowed origins
export TEXTPATCH_ALLOWED_ORIGINS=https://your-domain.com

# Bind to all interfaces
export TEXTPATCH_HOST=0.0.0.0
```

### Image Upload Protection

- Maximum upload size is configurable (default 50 MB)
- Image dimensions and pixel counts are validated
- Corrupted or malformed images are rejected
- Path traversal is prevented in project/region IDs

### LLM API Key Protection

- API keys are loaded from environment variables or `apikey.txt`
- `apikey.txt` is excluded from git via `.gitignore`
- Logs never contain API keys
- Mock client is used when no key is configured (clearly marked as unavailable)

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Security Features

- Optional API token authentication for all write endpoints
- Chunked file upload with size limits
- Image decompression bomb protection
- Zip Slip prevention in font downloader
- Atomic file writes (write-to-temp + rename)
- CORS origin whitelist
- Prompt injection guard in LLM system prompts
- Log redaction for sensitive data
