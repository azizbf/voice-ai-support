from __future__ import annotations

import hashlib
import hmac
import time
from itsdangerous import BadSignature, URLSafeSerializer

from app.core.config import get_settings


def _serializer() -> URLSafeSerializer:
    return URLSafeSerializer(get_settings().demo_signing_secret, salt="tenant-demo")


def issue_tenant_token(tenant_id: str, expires_at_ts: float) -> str:
    return _serializer().dumps({"tid": tenant_id, "exp": expires_at_ts})


def verify_tenant_token(token: str, tenant_id: str) -> bool:
    try:
        data = _serializer().loads(token)
    except BadSignature:
        return False
    if data.get("tid") != tenant_id:
        return False
    if float(data.get("exp", 0)) < time.time():
        return False
    return True


def safe_filename(name: str) -> str:
    base = "".join(c if c.isalnum() or c in "._-" else "_" for c in name.strip())
    base = base.strip("._") or "document.pdf"
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    return base[:180]


def is_pdf_magic(header: bytes) -> bool:
    return header[:5] == b"%PDF-"


def hash_ip(ip: str) -> str:
    secret = get_settings().demo_signing_secret.encode()
    return hmac.new(secret, ip.encode(), hashlib.sha256).hexdigest()[:16]
