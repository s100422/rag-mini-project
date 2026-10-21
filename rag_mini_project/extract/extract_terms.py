"""약관·용어집처럼 글 위주인 PDF를 pymupdf4llm으로 Markdown으로 추출한다.

쪽마다 `<!-- page: N -->` 표시를 남겨 근거 쪽을 찾을 수 있게 한다.

사용법:
  python extract/extract_terms.py                      # 01_data/standard_terms/*.pdf → md/standard_terms/
  python extract/extract_terms.py --glossary           # 01_data/financial_glossary_readable.pdf → md/glossary/
  python extract/extract_terms.py a.pdf b.pdf -o md/x  # 파일과 출력 폴더를 직접 지정
  --force 를 주면 이미 있는 md도 다시 만든다
"""
import argparse
import sys
import time
from pathlib import Path

import pymupdf4llm

ROOT = Path(__file__).resolve().parent.parent
STANDARD_TERMS = (ROOT / "01_data" / "standard_terms", ROOT / "md" / "standard_terms")
GLOSSARY = (ROOT / "01_data" / "financial_glossary_readable.pdf", ROOT / "md" / "glossary")


def extract(pdf: Path) -> str:
    chunks = pymupdf4llm.to_markdown(str(pdf), page_chunks=True, show_progress=False)
    parts = [f"<!-- source: {pdf.name} -->"]
    for c in chunks:
        parts.append(f"<!-- page: {c['metadata']['page_number']} -->")
        parts.append(c["text"].strip())
    return "\n\n".join(parts) + "\n"


def promote_term_headings(md: str, pdf: Path) -> tuple[str, list[str]]:
    """용어집에서 제목(##)이 되지 못한 용어 제목 줄을 ##로 올린다.

    pymupdf4llm은 영문 괄호가 붙은 긴 제목('MMF(Money Market Fund, …)')을 본문으로 두는 경우가 있다.
    PDF에서 '##가 된 줄'의 글자 크기를 구해(가장 흔한 크기 = 용어 제목 크기), 같은 크기인데 제목이 아닌 줄을 올린다.
    """
    import re
    from collections import Counter

    import pymupdf

    key = lambda s: re.sub(r"[\s*#]", "", s)
    heads = {key(h) for h in re.findall(r"(?m)^## (.*)$", md)}
    lines = []
    for p in pymupdf.open(pdf):
        for b in p.get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                sp = [s for s in ln["spans"] if s["text"].strip()]
                if sp:
                    lines.append((round(sp[0]["size"], 1), "".join(s["text"] for s in sp).strip()))
    sizes = Counter(sz for sz, t in lines if key(t) in heads)
    if not sizes:
        return md, []
    title_size = sizes.most_common(1)[0][0]
    wanted = {key(t): t for sz, t in lines if sz == title_size and key(t) not in heads}
    promoted, out = [], []
    for ln in md.split("\n"):
        k = key(ln)
        if k in wanted and not ln.lstrip().startswith("#") and "|" not in ln:
            out.append("## " + ln.strip())
            promoted.append(wanted[k])
        else:
            out.append(ln)
    return "\n".join(out), promoted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="*", type=Path)
    ap.add_argument("-o", "--out-dir", type=Path)
    ap.add_argument("--glossary", action="store_true", help="금융용어집 PDF 한 개를 md/glossary/로")
    ap.add_argument("--force", action="store_true", help="이미 있는 md도 다시 만든다")
    args = ap.parse_args()

    if args.pdfs:
        pdfs, out_dir = args.pdfs, args.out_dir or STANDARD_TERMS[1]
    elif args.glossary:
        pdfs, out_dir = [GLOSSARY[0]], args.out_dir or GLOSSARY[1]
    else:
        pdfs, out_dir = sorted(STANDARD_TERMS[0].glob("*.pdf")), args.out_dir or STANDARD_TERMS[1]

    out_dir.mkdir(parents=True, exist_ok=True)
    for pdf in pdfs:
        out = out_dir / (pdf.stem + ".md")
        if out.exists() and not args.force:
            print(f"skip  {out.name}", flush=True)
            continue
        t = time.time()
        md = extract(pdf)
        if args.glossary:
            md, promoted = promote_term_headings(md, pdf)
            print(f"      제목으로 올린 용어 {len(promoted)}개: {promoted}", flush=True)
            import re
            md, n_page_no = re.subn(r"(?m)^[ \t]*\d{1,3}[ \t]*\n(?:[ \t]*\n)?", "", md)   # 쪽 번호만 있는 줄(쪽 표시가 따로 있다)
            print(f"      쪽 번호 줄 {n_page_no}개 제거", flush=True)
        out.write_text(md, encoding="utf-8")
        print(f"done  {out.name} ({time.time() - t:.1f}s)", flush=True)
    print(f"{len(pdfs)} files -> {out_dir}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
