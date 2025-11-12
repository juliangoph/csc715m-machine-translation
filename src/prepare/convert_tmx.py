"""
Convert a bilingual .tmx file into plain parallel text files.

Usage:
    python src/prepare/convert_tmx.py \
        --tmx data/raw/ceb-tl.tmx \
        --src_lang ceb \
        --tgt_lang tl \
        --out_dir data/raw

This writes:
    data/raw/source/source.txt
    data/raw/target/target.txt
"""

"""
Robust TMX -> parallel text converter.
- Tolerates broken XML via lxml recover
- Handles namespaces and xml:lang
- Optional pre-clean pass for control chars and stray ampersands
"""

import argparse, re, io
from pathlib import Path

# Try lxml first (best). If not available, fallback to stdlib.
try:
    from lxml import etree as LET
    HAVE_LXML = True
except Exception:
    import xml.etree.ElementTree as ET
    HAVE_LXML = False

CTRL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

AMP_FIX = re.compile(r"&(?!#\d+;|#x[0-9A-Fa-f]+;|[A-Za-z][A-Za-z0-9]*;)")

def clean_xml_bytes(raw_bytes: bytes) -> bytes:
    """
    Minimal cleaning:
    - decode/encode to normalize UTF-8
    - strip control chars
    - escape stray & to &amp;
    """
    text = raw_bytes.decode("utf-8", errors="replace")
    text = CTRL_CHARS.sub("", text)
    text = AMP_FIX.sub("&amp;", text)
    return text.encode("utf-8")

def parse_with_lxml(path: Path):
    parser = LET.XMLParser(recover=True, huge_tree=True, encoding="utf-8")
    return LET.parse(str(path), parser)

def parse_with_stdlib(path: Path):
    # stdlib is strict; often fails on OPUS TMX
    return ET.parse(str(path))

def iter_tu_nodes(tree):
    """Namespace-agnostic search for TU nodes."""
    if HAVE_LXML:
        return tree.findall(".//{*}tu")
    else:
        return tree.findall(".//tu")

def findall(node, tag):
    """Namespace-agnostic findall."""
    if HAVE_LXML:
        return node.findall(f".//{{*}}{tag}")
    else:
        return node.findall(tag)

def get_lang(tuv):
    # xml:lang may be namespaced
    if HAVE_LXML:
        lang = tuv.get("{http://www.w3.org/XML/1998/namespace}lang")
        if not lang:
            lang = tuv.get("lang")
        return (lang or "").lower()
    else:
        return (tuv.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "")
                or tuv.attrib.get("lang", "")).lower()

def get_seg_text(tuv):
    segs = findall(tuv, "seg")
    if not segs:
        return None
    # Prefer direct text; join children text if needed
    seg = segs[0]
    if HAVE_LXML:
        return "".join(seg.itertext()).strip() if seg is not None else None
    else:
        return (seg.text or "").strip() if seg is not None else None

def parse_tmx_pairs(tmx_path: Path, src_pref: str, tgt_pref: str):
    # 1) Try robust lxml parse
    src_lines, tgt_lines = [], []
    try:
        if HAVE_LXML:
            tree = parse_with_lxml(tmx_path)
        else:
            tree = parse_with_stdlib(tmx_path)
    except Exception:
        # 2) Clean then re-parse
        raw = tmx_path.read_bytes()
        cleaned = clean_xml_bytes(raw)
        tmp = tmx_path.with_suffix(".cleaned.tmx")
        tmp.write_bytes(cleaned)
        if HAVE_LXML:
            tree = parse_with_lxml(tmp)
        else:
            tree = parse_with_stdlib(tmp)

    tus = iter_tu_nodes(tree)
    count = 0
    for tu in tus:
        tuvs = findall(tu, "tuv")
        if len(tuvs) < 2:
            continue

        # Collect (lang, text) for all tuv
        pairs = []
        for tuv in tuvs:
            lang = get_lang(tuv)
            text = get_seg_text(tuv)
            if not text:
                continue
            pairs.append((lang, text))

        if not pairs:
            continue

        # Find one src and one tgt by prefix match
        src_text = None
        tgt_text = None

        # normalize preferences like 'ceb', 'tl', 'tgl'
        src_try = [src_pref.lower()]
        tgt_try = [tgt_pref.lower(), ("tgl" if tgt_pref.lower()=="tl" else "tl")]

        for lang, text in pairs:
            if src_text is None and any(lang.startswith(p) for p in src_try):
                src_text = text
            if tgt_text is None and any(lang.startswith(p) for p in tgt_try):
                tgt_text = text

        # If still missing (some TMX flip order), try best-effort pick of two distinct langs
        if src_text is None or tgt_text is None:
            # Heuristic: pick first two distinct langs
            seen = {}
            for lang, text in pairs:
                if lang not in seen:
                    seen[lang] = text
                if len(seen) >= 2:
                    break
            if src_text is None and seen:
                # pick any as src if it matches src_pref loosely
                for lang, text in seen.items():
                    if lang.startswith(src_pref.lower()):
                        src_text = text
                        break
            if tgt_text is None and seen:
                for lang, text in seen.items():
                    if lang.startswith(tgt_pref.lower()) or lang.startswith("tgl") or lang=="tl":
                        tgt_text = text
                        break

        if src_text and tgt_text:
            src_lines.append(src_text)
            tgt_lines.append(tgt_text)
            count += 1

    return src_lines, tgt_lines

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmx", required=True, help="Path to TMX file (e.g., data/raw/ceb-tl.tmx)")
    ap.add_argument("--src_lang", required=True, help="Source language code (e.g., ceb)")
    ap.add_argument("--tgt_lang", required=True, help="Target language code (e.g., tl or tgl)")
    ap.add_argument("--out_dir", default="data/raw", help="Output directory for source/target files")
    args = ap.parse_args()

    tmx_path = Path(args.tmx)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_lines, tgt_lines = parse_tmx_pairs(tmx_path, args.src_lang, args.tgt_lang)
    print(f"Extracted {len(src_lines):,} aligned pairs")

    # Write outputs
    src_out = out_dir / "source" / "source.txt"
    tgt_out = out_dir / "target" / "target.txt"
    src_out.parent.mkdir(parents=True, exist_ok=True)
    tgt_out.parent.mkdir(parents=True, exist_ok=True)

    with io.open(src_out, "w", encoding="utf-8") as fs, io.open(tgt_out, "w", encoding="utf-8") as ft:
        for s, t in zip(src_lines, tgt_lines):
            fs.write(s.replace("\n", " ").strip() + "\n")
            ft.write(t.replace("\n", " ").strip() + "\n")

    print(f"Wrote:\n  {src_out}\n  {tgt_out}")

if __name__ == "__main__":
    main()
