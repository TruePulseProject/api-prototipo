"""
Token de sessão emitido quando a biometria aprova o login.

JWT HS256 implementado com a biblioteca padrão (hmac + hashlib + base64) para
não adicionar dependência só por causa de ~40 linhas. Se o seu backend já usa
PyJWT ou python-jose, troque as duas funções abaixo pela sua implementação —
o resto da API não muda.

Este token diz apenas "esta pessoa passou pela verificação biométrica no
instante X". Ele não substitui o seu controle de sessão/refresh: o seu
serviço de usuários é que decide o que o portador pode fazer.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from app.config import settings


def _b64url(dados: bytes) -> str:
    return base64.urlsafe_b64encode(dados).rstrip(b"=").decode("ascii")


def _b64url_decode(texto: str) -> bytes:
    padding = "=" * (-len(texto) % 4)
    return base64.urlsafe_b64decode(texto + padding)


def _assinar(mensagem: str) -> str:
    assinatura = hmac.new(settings.jwt_secret.encode(), mensagem.encode(),
                          hashlib.sha256).digest()
    return _b64url(assinatura)


def emitir_token(user_id: str | None, job_id: str,
                 liveness_pct: float) -> tuple[str, int]:
    """Devolve (token, segundos_ate_expirar)."""
    agora = int(time.time())
    ttl = settings.jwt_ttl_minutes * 60
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": user_id or "anonimo",
        "iss": settings.jwt_issuer,
        "iat": agora,
        "exp": agora + ttl,
        "job_id": job_id,
        "auth_method": "biometria_rppg",
        "liveness_pct": round(liveness_pct, 2),
    }
    corpo = f"{_b64url(json.dumps(header, separators=(',', ':')).encode())}." \
            f"{_b64url(json.dumps(payload, separators=(',', ':')).encode())}"
    return f"{corpo}.{_assinar(corpo)}", ttl


def validar_token(token: str) -> dict | None:
    """Devolve o payload se o token for válido e não expirado; senão None."""
    try:
        header_b64, payload_b64, assinatura = token.split(".")
    except ValueError:
        return None

    corpo = f"{header_b64}.{payload_b64}"
    if not hmac.compare_digest(_assinar(corpo), assinatura):
        return None

    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except (ValueError, json.JSONDecodeError):
        return None

    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    return payload
