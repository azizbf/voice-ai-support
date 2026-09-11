from __future__ import annotations

import re
from dataclasses import dataclass

from pypdf import PdfReader

from app.core.config import get_settings
from app.core.security import is_pdf_magic
from app.models.schemas import PageText


class PdfValidationError(ValueError):
    pass


@dataclass
class ExtractedPdf:
    page_count: int
    pages: list[PageText]


def validate_pdf_bytes(data: bytes, filename: str) -> None:
    settings = get_settings()
    if not filename.lower().endswith(".pdf"):
        raise PdfValidationError("Seuls les fichiers PDF sont acceptés.")
    if len(data) == 0:
        raise PdfValidationError("Fichier vide.")
    if len(data) > settings.max_upload_bytes:
        raise PdfValidationError(
            f"Fichier trop volumineux (max {settings.max_upload_bytes // (1024 * 1024)} Mo)."
        )
    if not is_pdf_magic(data[:8]):
        raise PdfValidationError("Le fichier n'est pas un PDF valide.")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Drop very short noise lines that are often headers/footers
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if len(stripped) <= 2 and not stripped.isalnum():
            continue
        lines.append(stripped)
    return "\n".join(lines).strip()


def extract_pdf(data: bytes) -> ExtractedPdf:
    settings = get_settings()
    from io import BytesIO

    reader = PdfReader(BytesIO(data))
    if len(reader.pages) == 0:
        raise PdfValidationError("PDF sans pages extractibles.")
    if len(reader.pages) > settings.max_pdf_pages:
        raise PdfValidationError(f"Trop de pages (max {settings.max_pdf_pages}).")

    pages: list[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if cleaned:
            pages.append(PageText(page=i, text=cleaned))

    if not pages:
        raise PdfValidationError(
            "Impossible d'extraire du texte. Le PDF est peut-être scanné (image seule)."
        )

    return ExtractedPdf(page_count=len(reader.pages), pages=pages)
