import argparse
import json
import random
from pathlib import Path

from .config import BASE_DIR, TOP_K
from .retriever import load_index, search, suggest_categories

DEFAULT_EVAL_PATH = BASE_DIR / "rag" / "eval_dataset.jsonl"


def _unique_records_from_index() -> list[dict]:
    _, _, metadatas, _ = load_index()
    seen = set()
    records = []
    for m in metadatas:
        url = (m.get("url") or "").strip()
        title = (m.get("title") or "").strip()
        category = (m.get("category") or "").strip()
        if not url or not title:
            continue
        if url in seen:
            continue
        seen.add(url)
        records.append(
            {
                "query": title,
                "expected_urls": [url],
                "category": category,
                "date_from": None,
                "date_to": None,
            }
        )
    return records


def create_eval_dataset(path: Path = DEFAULT_EVAL_PATH, sample_size: int = 120, seed: int = 42) -> Path:
    records = _unique_records_from_index()
    rng = random.Random(seed)
    if sample_size > 0 and len(records) > sample_size:
        records = rng.sample(records, sample_size)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def load_eval_dataset(path: Path = DEFAULT_EVAL_PATH) -> list[dict]:
    items = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            query = (item.get("query") or "").strip()
            expected = item.get("expected_urls") or []
            if query and expected:
                items.append(item)
    return items


def evaluate_retrieval(
    dataset: list[dict],
    top_k: int = TOP_K,
    use_category_filter: bool = False,
) -> dict:
    if not dataset:
        return {
            "count": 0,
            "hit_at_k": 0.0,
            "mrr": 0.0,
            "category_top1_acc": 0.0,
            "top_k": top_k,
        }

    hits = 0
    mrr_sum = 0.0
    cat_hits = 0
    cat_total = 0

    for item in dataset:
        query = item["query"]
        expected_urls = set(item.get("expected_urls") or [])
        category = (item.get("category") or "").strip() if use_category_filter else None
        date_from = item.get("date_from")
        date_to = item.get("date_to")

        docs = search(
            query=query,
            top_k=top_k,
            category=category or None,
            date_from=date_from,
            date_to=date_to,
        )
        found_rank = None
        for i, d in enumerate(docs, 1):
            url = (d.get("metadata", {}).get("url") or "").strip()
            if url in expected_urls:
                found_rank = i
                break

        if found_rank is not None:
            hits += 1
            mrr_sum += 1.0 / found_rank

        # Semantic category suggestion quality
        true_cat = (item.get("category") or "").strip()
        if true_cat:
            cat_total += 1
            suggested = suggest_categories(query, top_n=1)
            if suggested and suggested[0]["category"] == true_cat:
                cat_hits += 1

    n = len(dataset)
    return {
        "count": n,
        "hit_at_k": hits / n,
        "mrr": mrr_sum / n,
        "category_top1_acc": (cat_hits / cat_total) if cat_total else 0.0,
        "top_k": top_k,
        "use_category_filter": use_category_filter,
    }


def run_eval(
    dataset_path: Path = DEFAULT_EVAL_PATH,
    create_if_missing: bool = True,
    sample_size: int = 120,
    seed: int = 42,
    top_k: int = TOP_K,
    use_category_filter: bool = False,
) -> dict:
    if create_if_missing and not dataset_path.exists():
        created = create_eval_dataset(dataset_path, sample_size=sample_size, seed=seed)
        print(f"[i] Eval dataset oluşturuldu: {created}")

    dataset = load_eval_dataset(dataset_path)
    print(f"[i] Eval örnek sayısı: {len(dataset)}")

    metrics = evaluate_retrieval(
        dataset=dataset,
        top_k=top_k,
        use_category_filter=use_category_filter,
    )

    print()
    print("=" * 64)
    print("RAG Eval")
    print("=" * 64)
    print(f"Dataset path        : {dataset_path}")
    print(f"Samples             : {metrics['count']}")
    print(f"Top-K               : {metrics['top_k']}")
    print(f"Use category filter : {metrics['use_category_filter']}")
    print(f"Hit@K               : {metrics['hit_at_k']:.3f}")
    print(f"MRR                 : {metrics['mrr']:.3f}")
    print(f"Category Top1 Acc   : {metrics['category_top1_acc']:.3f}")
    print("=" * 64)

    return metrics


def cli(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="RAG retrieval değerlendirme")
    parser.add_argument("--dataset", default=str(DEFAULT_EVAL_PATH), help="JSONL eval dataset path")
    parser.add_argument("--build", action="store_true", help="Dataseti yeniden oluştur")
    parser.add_argument("--sample", type=int, default=120, help="Auto dataset örnek sayısı")
    parser.add_argument("--seed", type=int, default=42, help="Rastgele seed")
    parser.add_argument("--k", type=int, default=TOP_K, help="Top-K")
    parser.add_argument("--kategori", action="store_true", help="Aramada category filtresi uygula")
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    if args.build:
        create_eval_dataset(dataset_path, sample_size=args.sample, seed=args.seed)
        print(f"[i] Dataset rebuild tamam: {dataset_path}")

    return run_eval(
        dataset_path=dataset_path,
        create_if_missing=True,
        sample_size=args.sample,
        seed=args.seed,
        top_k=args.k,
        use_category_filter=args.kategori,
    )


if __name__ == "__main__":
    cli()
