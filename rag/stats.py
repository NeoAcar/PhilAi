import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import CONTENT_DIR
from .retriever import parse_date_string


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ts_to_iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _read_headers(path: Path) -> dict[str, str]:
    headers = {
        "title": "",
        "url": "",
        "date": "",
        "author": "",
        "categories": "",
    }
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("TITLE:"):
                    headers["title"] = line.split(":", 1)[1].strip()
                elif line.startswith("URL:"):
                    headers["url"] = line.split(":", 1)[1].strip()
                elif line.startswith("DATE:"):
                    headers["date"] = line.split(":", 1)[1].strip()
                elif line.startswith("AUTHOR:"):
                    headers["author"] = line.split(":", 1)[1].strip()
                elif line.startswith("CATEGORIES:"):
                    headers["categories"] = line.split(":", 1)[1].strip()
                elif line.strip() == "-----":
                    break
    except Exception:
        pass
    return headers


def _split_header_categories(raw: str) -> list[str]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(">")]
    return [p for p in parts if p]


def _safe_stat_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except Exception:
        return None


def build_stats_report(content_dir: Path = CONTENT_DIR) -> dict:
    txt_files = list(content_dir.rglob("*.txt"))
    category_file_counts = Counter()
    header_category_mentions = Counter()
    url_counter = Counter()
    category_unique_urls: dict[str, set[str]] = defaultdict(set)

    date_total = 0
    date_ok = 0
    parsed_dates = []

    newest_file_ts = None
    oldest_file_ts = None

    for path in txt_files:
        folder = path.parent.name
        category_file_counts[folder] += 1

        mtime = _safe_stat_mtime(path)
        if mtime is not None:
            newest_file_ts = mtime if newest_file_ts is None else max(newest_file_ts, mtime)
            oldest_file_ts = mtime if oldest_file_ts is None else min(oldest_file_ts, mtime)

        headers = _read_headers(path)
        url = headers.get("url", "")
        if url:
            url_counter[url] += 1
            category_unique_urls[folder].add(url)

        for cat in _split_header_categories(headers.get("categories", "")):
            header_category_mentions[cat] += 1

        raw_date = headers.get("date", "")
        if raw_date:
            date_total += 1
            parsed = parse_date_string(raw_date)
            if parsed:
                date_ok += 1
                parsed_dates.append(parsed)

    unique_urls = len(url_counter)
    file_count = len(txt_files)
    duplicate_file_ratio = 1.0 - (unique_urls / file_count) if file_count else 0.0

    categories_manifest = content_dir / "_categories.json"
    manifest_ts = _safe_stat_mtime(categories_manifest) if categories_manifest.exists() else None

    last_scrape_ts = None
    for ts in (newest_file_ts, manifest_ts):
        if ts is None:
            continue
        last_scrape_ts = ts if last_scrape_ts is None else max(last_scrape_ts, ts)

    categories = []
    for category, files in category_file_counts.most_common():
        categories.append(
            {
                "category": category,
                "files": files,
                "unique_urls": len(category_unique_urls.get(category, set())),
            }
        )

    report = {
        "generated_at_utc": _utc_now_iso(),
        "content_dir": str(content_dir),
        "summary": {
            "papers_total_files": file_count,
            "papers_unique_urls": unique_urls,
            "categories_total": len(category_file_counts),
            "duplicate_file_ratio": duplicate_file_ratio,
        },
        "scrape": {
            "categories_manifest_path": str(categories_manifest),
            "categories_manifest_updated_utc": _ts_to_iso(manifest_ts),
            "oldest_file_written_utc": _ts_to_iso(oldest_file_ts),
            "latest_file_written_utc": _ts_to_iso(newest_file_ts),
            "estimated_last_scrape_utc": _ts_to_iso(last_scrape_ts),
        },
        "dates": {
            "date_metadata_parse_rate": (date_ok / date_total) if date_total else 0.0,
            "earliest_article_date": min(parsed_dates).isoformat() if parsed_dates else None,
            "latest_article_date": max(parsed_dates).isoformat() if parsed_dates else None,
        },
        "categories": {
            "by_folder": categories,
            "header_category_mentions": [
                {"category": cat, "mentions": count}
                for cat, count in header_category_mentions.most_common()
            ],
        },
        "quality": {
            "top_duplicate_urls": [
                {"url": url, "count": count}
                for url, count in url_counter.most_common(10)
                if count > 1
            ]
        },
    }
    return report


def write_stats_report(output_path: Path | None = None, content_dir: Path = CONTENT_DIR) -> Path:
    report = build_stats_report(content_dir=content_dir)
    path = output_path or (content_dir / "_stats.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_stats(output_path: str | None = None) -> dict:
    target_path = Path(output_path) if output_path else (CONTENT_DIR / "_stats.json")
    report = build_stats_report()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 64)
    print("Corpus Stats")
    print("=" * 64)
    print(f"Output file          : {target_path}")
    print(f"Papers (files)       : {report['summary']['papers_total_files']}")
    print(f"Papers (unique URLs) : {report['summary']['papers_unique_urls']}")
    print(f"Categories           : {report['summary']['categories_total']}")
    print(f"Last scrape (est.)   : {report['scrape']['estimated_last_scrape_utc']}")
    print("=" * 64)
    return report


if __name__ == "__main__":
    run_stats()
