"""Generate a sample FAQ PDF from the shared demo knowledge pages."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.rag.demo_knowledge import DEMO_PAGES  # noqa: E402


def build_pdf(pages: list[str]) -> bytes:
    """Minimal PDF with one text line per page (ASCII-safe for Helvetica)."""
    objs: list[bytes] = []

    def obj(n: int, body: bytes) -> None:
        objs.append(f"{n} 0 obj\n".encode() + body + b"\nendobj\n")

    obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    page_ids = list(range(3, 3 + len(pages) * 2, 2))
    kids = " ".join(f"{i} 0 R" for i in page_ids).encode()
    obj(2, b"<< /Type /Pages /Count %d /Kids [%s] >>" % (len(pages), kids))

    n = 3
    for text in pages:
        # Helvetica is Latin-1; fold accents for the optional PDF artifact
        folded = (
            text.replace("é", "e")
            .replace("è", "e")
            .replace("ê", "e")
            .replace("à", "a")
            .replace("â", "a")
            .replace("ù", "u")
            .replace("û", "u")
            .replace("ô", "o")
            .replace("î", "i")
            .replace("ï", "i")
            .replace("ç", "c")
            .replace("É", "E")
            .replace("À", "A")
            .replace("’", "'")
            .replace("–", "-")
            .replace("—", "-")
        )
        # Keep first ~900 chars so the content stream stays simple
        line = " ".join(folded.split())[:900]
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 10 Tf 40 720 Td ({safe}) Tj ET".encode("latin-1", errors="replace")
        obj(
            n,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents %d 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>"
            % (n + 1),
        )
        obj(n + 1, b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        n += 2

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for o in objs:
        offsets.append(len(out))
        out.extend(o)
    xref_pos = len(out)
    out.extend(f"xref\n0 {n}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode())
    out.extend(f"trailer\n<< /Size {n} /Root 1 0 R >>\n".encode())
    out.extend(f"startxref\n{xref_pos}\n%%EOF\n".encode())
    return bytes(out)


def main() -> None:
    out = ROOT / "samples" / "faq_internet.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(build_pdf(DEMO_PAGES))
    print(f"Wrote {out} ({len(DEMO_PAGES)} pages)")


if __name__ == "__main__":
    main()
