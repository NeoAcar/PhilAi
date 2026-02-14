from collections import Counter
from pathlib import Path

from .config import CONTENT_DIR
from .retriever import load_index, parse_date_string


def _read_url_from_file(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("URL:"):
                    return line.split(":", 1)[1].strip()
                if line.strip() == "-----":
                    break
    except Exception:
        return ""
    return ""


def _read_date_from_file(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("DATE:"):
                    return line.split(":", 1)[1].strip()
                if line.strip() == "-----":
                    break
    except Exception:
        return ""
    return ""


def build_doctor_report() -> dict:
    raw_files = list(CONTENT_DIR.rglob("*.txt"))
    raw_url_counter = Counter()
    raw_cat_counter = Counter()
    raw_date_ok = 0
    raw_date_total = 0

    for file_path in raw_files:
        raw_cat_counter[file_path.parent.name] += 1
        url = _read_url_from_file(file_path)
        if url:
            raw_url_counter[url] += 1
        raw_date = _read_date_from_file(file_path)
        if raw_date:
            raw_date_total += 1
            if parse_date_string(raw_date):
                raw_date_ok += 1

    index, chunks, metadatas, _ = load_index()
    idx_url_counter = Counter((m.get("url") or "").strip() for m in metadatas if (m.get("url") or "").strip())
    idx_cat_counter = Counter((m.get("category") or "").strip() for m in metadatas if (m.get("category") or "").strip())

    idx_date_total = 0
    idx_date_ok = 0
    for m in metadatas:
        d = (m.get("date") or "").strip()
        if not d:
            continue
        idx_date_total += 1
        if parse_date_string(d):
            idx_date_ok += 1

    raw_unique_urls = len(raw_url_counter)
    idx_unique_urls = len(idx_url_counter)
    raw_files_n = len(raw_files)
    idx_chunks_n = len(chunks)

    raw_dup_ratio = 1.0 - (raw_unique_urls / raw_files_n) if raw_files_n else 0.0
    idx_dup_ratio = 1.0 - (idx_unique_urls / idx_chunks_n) if idx_chunks_n else 0.0

    report = {
        "raw": {
            "files": raw_files_n,
            "unique_urls": raw_unique_urls,
            "dup_ratio": raw_dup_ratio,
            "categories": len(raw_cat_counter),
            "small_categories_le_2": sum(1 for _, c in raw_cat_counter.items() if c <= 2),
            "date_parse_rate": (raw_date_ok / raw_date_total) if raw_date_total else 0.0,
            "top_duplicate_urls": raw_url_counter.most_common(5),
        },
        "index": {
            "chunks": idx_chunks_n,
            "unique_urls": idx_unique_urls,
            "dup_ratio": idx_dup_ratio,
            "categories": len(idx_cat_counter),
            "date_parse_rate": (idx_date_ok / idx_date_total) if idx_date_total else 0.0,
            "top_duplicate_urls": idx_url_counter.most_common(5),
            "index_dim": index.d,
        },
        "coverage": {
            "raw_urls_missing_in_index": len(set(raw_url_counter) - set(idx_url_counter)),
            "index_urls_not_in_raw": len(set(idx_url_counter) - set(raw_url_counter)),
        },
    }
    return report


def _print_section(title: str):
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def run_doctor() -> dict:
    report = build_doctor_report()
    raw = report["raw"]
    idx = report["index"]
    cov = report["coverage"]

    _print_section("RAG Doctor Report")
    print(f"Raw files           : {raw['files']}")
    print(f"Raw unique URLs     : {raw['unique_urls']}")
    print(f"Raw duplicate ratio : {raw['dup_ratio']:.3f}")
    print(f"Raw categories      : {raw['categories']}")
    print(f"Small cats (<=2)    : {raw['small_categories_le_2']}")
    print(f"Raw date parse rate : {raw['date_parse_rate']:.3f}")

    _print_section("Index Health")
    print(f"Index chunks        : {idx['chunks']}")
    print(f"Index unique URLs   : {idx['unique_urls']}")
    print(f"Index duplicate ratio: {idx['dup_ratio']:.3f}")
    print(f"Index categories    : {idx['categories']}")
    print(f"Index date parse rate: {idx['date_parse_rate']:.3f}")
    print(f"Index dim           : {idx['index_dim']}")

    _print_section("Coverage")
    print(f"Raw URLs missing in index: {cov['raw_urls_missing_in_index']}")
    print(f"Index URLs not in raw    : {cov['index_urls_not_in_raw']}")

    _print_section("Top Duplicate URLs (Raw)")
    for url, cnt in raw["top_duplicate_urls"]:
        print(f"{cnt:4d}  {url}")

    _print_section("Top Duplicate URLs (Index Chunks)")
    for url, cnt in idx["top_duplicate_urls"]:
        print(f"{cnt:4d}  {url}")

    _print_section("Recommendations")
    if raw["small_categories_le_2"] > 10:
        print("- Noisy category labels yüksek: scraper kategori filtrelerini daha da sıkılaştırın.")
    if idx["dup_ratio"] > 0.8:
        print("- Indexte aynı URL'den çok chunk var: hybrid+MMR açık tutun, gerekirse max chunk per URL sınırı ekleyin.")
    if cov["raw_urls_missing_in_index"] > 0:
        print("- Bazı raw URL'ler indexte yok: `python main.py index --full` ile temiz rebuild önerilir.")
    if raw["date_parse_rate"] < 0.8:
        print("- DATE metadata formatlarını normalize edin (ISO önerilir) ki tarih filtresi daha etkili olsun.")
    if (
        raw["small_categories_le_2"] <= 10
        and idx["dup_ratio"] <= 0.8
        and cov["raw_urls_missing_in_index"] == 0
        and raw["date_parse_rate"] >= 0.8
    ):
        print("- Genel sağlık iyi görünüyor. Mevcut yapı yeni retrieval özelliklerini destekliyor.")

    return report


if __name__ == "__main__":
    run_doctor()
