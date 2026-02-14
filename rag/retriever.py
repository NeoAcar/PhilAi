# Retriever - FAISS'den arama
import json
import pickle
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .config import (
    EMBEDDING_PROVIDER,
    CATEGORY_DESCRIPTIONS,
    LOCAL_EMBEDDING_MODEL,
    MMR_LAMBDA,
    OPENAI_EMBEDDING_MODEL,
    RERANK_TOP_N,
    RERANK_WEIGHT,
    RERANKER_MODEL,
    SEMANTIC_CATEGORY_MIN_CHUNKS,
    TOP_K,
    USE_GPU,
    USE_INSTRUCT_FORMAT,
    USE_MMR,
    USE_RERANKER,
    get_index_path,
    INSTRUCT_TASK,
)

# Lazy imports
_openai_client = None
_local_model = None
_reranker_model = None
_index_cache: dict[str, Any] | None = None
_category_index_cache: dict[str, tuple[faiss.Index, np.ndarray]] = {}

# Retrieval tuning
DEFAULT_CANDIDATE_MULTIPLIER = 8
DEFAULT_MIN_CANDIDATES = 30
MAX_MULTI_QUERY_PER_QUERY_K = 14
MAX_SUBSET_VECTOR_SEARCH = 12000
CONTEXT_MAX_CHARS = 12000
CONTEXT_MAX_CHARS_PER_DOC = 1500

TR_MONTHS = {
    "ocak": 1,
    "şubat": 2,
    "subat": 2,
    "mart": 3,
    "nisan": 4,
    "mayıs": 5,
    "mayis": 5,
    "haziran": 6,
    "temmuz": 7,
    "ağustos": 8,
    "agustos": 8,
    "eylül": 9,
    "eylul": 9,
    "ekim": 10,
    "kasım": 11,
    "kasim": 11,
    "aralık": 12,
    "aralik": 12,
}


# Description embeddings cache
_desc_embedding_cache = {}

def _get_description_embeddings(config):
    """Kategori açıklamalarını önceden vektörleştir (lazy load)."""
    global _desc_embedding_cache
    if _desc_embedding_cache:
        return _desc_embedding_cache

    provider = config.get("embedding_provider", "local")
    
    # OpenAI için batch embed gerekebilir, şimdilik local odaklı yapıyoruz
    # Veya tek tek embed ediyoruz (az kategori var zaten)
    
    cats = {}
    
    for cat, desc in CATEGORY_DESCRIPTIONS.items():
        # Açıklamayı bir "doküman" gibi değil, bir "query" gibi de düşünebiliriz
        # ama en doğrusu: Query -> Description eşleşmesi.
        # Basitlik için _resolve_query_embedding kullanıyoruz.
        vec = _resolve_query_embedding(desc, config)
        if vec is not None:
             cats[cat] = vec
             
    _desc_embedding_cache = cats
    return cats


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI()
    return _openai_client


def get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
        _local_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL, device=device)
    return _local_model


def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder

        _reranker_model = CrossEncoder(RERANKER_MODEL)
    return _reranker_model


def _cache_fits(index_path: Path) -> bool:
    return _index_cache is not None and _index_cache.get("index_path") == str(index_path)


def clear_cache():
    """Retriever cache'ini temizle (uzun processlerde yenileme için)."""
    global _index_cache, _category_index_cache, _reranker_model
    _index_cache = None
    _category_index_cache = {}
    _reranker_model = None


def _clean_query(query: str) -> str:
    return " ".join((query or "").split())


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def parse_date_string(raw: str) -> date | None:
    """Metadata veya kullanıcı filtresinden tarih parse et."""
    text = (raw or "").strip()
    if not text:
        return None

    # ISO tarzı: 2024-03-18 veya 2024-03-18T...
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        parsed = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if parsed:
            return parsed

    # DD.MM.YYYY veya DD/MM/YYYY
    m = re.search(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b", text)
    if m:
        parsed = _safe_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if parsed:
            return parsed

    # "18 Mart 2021"
    m = re.search(r"\b(\d{1,2})\s+([A-Za-zÇĞİÖŞÜçğıöşü]+)\s+(\d{4})\b", text)
    if m:
        day = int(m.group(1))
        month_name = m.group(2).lower()
        month = TR_MONTHS.get(month_name)
        year = int(m.group(3))
        if month:
            parsed = _safe_date(year, month, day)
            if parsed:
                return parsed

    # Sadece yıl
    m = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if m:
        return date(int(m.group(1)), 1, 1)

    return None


def _to_ordinal_or_none(raw: str) -> int | None:
    parsed = parse_date_string(raw)
    return parsed.toordinal() if parsed else None


def _doc_key(metadata: dict, content: str) -> str:
    url = (metadata.get("url") or "").strip()
    if url:
        return f"url::{url}"
    filename = (metadata.get("filename") or "").strip()
    if filename:
        return f"file::{filename}"
    return f"text::{content[:120]}"


def _candidate_k(top_k: int, total: int) -> int:
    if total <= 0:
        return 0
    return min(total, max(top_k * DEFAULT_CANDIDATE_MULTIPLIER, top_k + DEFAULT_MIN_CANDIDATES))


def _build_metadata_caches(metadatas: list[dict]) -> tuple[list[int | None], dict[str, np.ndarray]]:
    date_ordinals: list[int | None] = []
    category_to_indices: dict[str, list[int]] = defaultdict(list)

    for i, m in enumerate(metadatas):
        date_ordinals.append(_to_ordinal_or_none(m.get("date", "")))
        cat = (m.get("category") or "").strip().lower()
        if cat:
            category_to_indices[cat].append(i)

    category_arrays = {k: np.array(v, dtype=np.int64) for k, v in category_to_indices.items()}
    return date_ordinals, category_arrays


def load_index(force_reload: bool = False):
    """FAISS index ve verileri yükle (process içi cache'li)."""
    global _index_cache, _category_index_cache
    index_path = get_index_path()

    if not index_path.exists():
        raise FileNotFoundError(f"Index bulunamadı: {index_path}\nÖnce 'python main.py index' çalıştırın.")

    if not force_reload and _cache_fits(index_path):
        return (
            _index_cache["index"],
            _index_cache["chunks"],
            _index_cache["metadatas"],
            _index_cache["config"],
        )

    index = faiss.read_index(str(index_path / "index.faiss"))

    with open(index_path / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open(index_path / "metadatas.pkl", "rb") as f:
        metadatas = pickle.load(f)

    config_path = index_path / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    date_ordinals, category_to_indices = _build_metadata_caches(metadatas)

    _index_cache = {
        "index_path": str(index_path),
        "index": index,
        "chunks": chunks,
        "metadatas": metadatas,
        "config": config,
        "date_ordinals": date_ordinals,
        "category_to_indices": category_to_indices,

        "category_centroids": None,
    }
    _category_index_cache = {}
    return index, chunks, metadatas, config


def _resolve_query_embedding(query: str, config: dict) -> np.ndarray:
    provider = config.get("embedding_provider", EMBEDDING_PROVIDER)
    if provider == "openai":
        client = get_openai_client()
        response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=query)
        emb = np.array([response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(emb)
        return emb

    model = get_local_model()
    if USE_INSTRUCT_FORMAT:
        processed = f"Instruct: {INSTRUCT_TASK}\nQuery: {query}"
    else:
        processed = f"query: {query}"
    emb = model.encode([processed], convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype(np.float32)


def _get_allowed_indices(
    metadatas: list[dict],
    category: str | None,
    date_from: str | None,
    date_to: str | None,
) -> np.ndarray | None:
    if _index_cache is None:
        load_index()

    date_ordinals = _index_cache["date_ordinals"]
    category_to_indices = _index_cache["category_to_indices"]

    allowed: np.ndarray | None = None

    if category:
        cat_key = category.strip().lower()
        allowed = category_to_indices.get(cat_key)
        if allowed is None:
            return np.array([], dtype=np.int64)

    from_ord = _to_ordinal_or_none(date_from or "")
    to_ord = _to_ordinal_or_none(date_to or "")

    if from_ord is None and to_ord is None:
        return allowed

    if allowed is None:
        base_indices = np.arange(len(metadatas), dtype=np.int64)
    else:
        base_indices = allowed

    filtered = []
    for idx in base_indices:
        d_ord = date_ordinals[int(idx)]
        if d_ord is None:
            continue
        if from_ord is not None and d_ord < from_ord:
            continue
        if to_ord is not None and d_ord > to_ord:
            continue
        filtered.append(int(idx))
    return np.array(filtered, dtype=np.int64)


def _vector_candidates(
    index,
    query_embedding: np.ndarray,
    top_n: int,
    allowed_indices: np.ndarray | None = None,
) -> dict[int, float]:
    if top_n <= 0:
        return {}

    total = index.ntotal
    if total <= 0:
        return {}

    # Filtre yoksa düz arama
    if allowed_indices is None:
        k = min(total, top_n)
        dists, idxs = index.search(query_embedding, k)
        return {int(idx): float(dists[0][i]) for i, idx in enumerate(idxs[0]) if int(idx) >= 0}

    if allowed_indices.size == 0:
        return {}

    # Küçük filtrede alt-index daha stabil
    if int(allowed_indices.size) <= MAX_SUBSET_VECTOR_SEARCH:
        vectors = np.vstack([index.reconstruct(int(i)) for i in allowed_indices]).astype(np.float32)
        sub_index = faiss.IndexFlatIP(vectors.shape[1])
        sub_index.add(vectors)
        k = min(int(allowed_indices.size), top_n)
        dists, local_idxs = sub_index.search(query_embedding, k)
        out = {}
        for i, loc in enumerate(local_idxs[0]):
            if int(loc) < 0:
                continue
            real_idx = int(allowed_indices[int(loc)])
            out[real_idx] = float(dists[0][i])
        return out

    # Büyük filtrelerde global arayıp sonra süz
    allowed_set = set(int(x) for x in allowed_indices)
    k = min(total, max(top_n * 4, 200))
    candidates: dict[int, float] = {}

    while True:
        dists, idxs = index.search(query_embedding, k)
        for i, idx in enumerate(idxs[0]):
            idx_int = int(idx)
            if idx_int < 0:
                continue
            if idx_int in allowed_set:
                candidates[idx_int] = float(dists[0][i])

        if len(candidates) >= top_n or k >= total:
            break
        k = min(total, int(k * 1.8))

    sorted_items = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return dict(sorted_items)


def _apply_reranker(query: str, candidates: list[dict], chunks: list[str]) -> list[dict]:
    if not candidates:
        return candidates
    try:
        model = get_reranker_model()
    except Exception:
        return candidates

    top_n = min(RERANK_TOP_N, len(candidates))
    top_slice = candidates[:top_n]
    pairs = [(query, chunks[item["idx"]][:1200]) for item in top_slice]

    try:
        raw_scores = model.predict(pairs)
    except Exception:
        return candidates

    # Normalize reranker scores to [0,1]
    raw_arr = np.array(raw_scores, dtype=np.float32)
    min_v, max_v = float(raw_arr.min()), float(raw_arr.max())
    if max_v - min_v < 1e-9:
        rerank_norm = {item["idx"]: 1.0 for item in top_slice}
    else:
        rerank_norm = {item["idx"]: float((raw_scores[i] - min_v) / (max_v - min_v)) for i, item in enumerate(top_slice)}

    rerank_map = {item["idx"]: float(raw_scores[i]) for i, item in enumerate(top_slice)}

    updated = []
    for item in candidates:
        idx = item["idx"]
        if idx in rerank_norm:
            item = dict(item)
            item["rerank_score"] = float(rerank_map[idx])
            item["score"] = float((1.0 - RERANK_WEIGHT) * item["score"] + RERANK_WEIGHT * rerank_norm[idx])
        updated.append(item)

    updated.sort(key=lambda x: x["score"], reverse=True)
    return updated


def _apply_mmr(index, candidates: list[dict], top_k: int, lambda_mult: float) -> list[dict]:
    if top_k <= 0 or len(candidates) <= top_k:
        return candidates[:top_k]

    pool_size = min(len(candidates), max(top_k * 8, 40))
    pool = candidates[:pool_size]
    vecs = np.vstack([index.reconstruct(item["idx"]) for item in pool]).astype(np.float32)
    faiss.normalize_L2(vecs)
    rel = np.array([item["score"] for item in pool], dtype=np.float32)

    selected = []
    selected_mask = np.zeros(pool_size, dtype=bool)

    first = int(np.argmax(rel))
    selected.append(first)
    selected_mask[first] = True

    while len(selected) < min(top_k, pool_size):
        best_idx = -1
        best_score = -1e9
        for i in range(pool_size):
            if selected_mask[i]:
                continue
            if not selected:
                diversity_penalty = 0.0
            else:
                sims = vecs[i : i + 1] @ vecs[selected].T
                diversity_penalty = float(np.max(sims))
            mmr_score = float(lambda_mult * rel[i] - (1.0 - lambda_mult) * diversity_penalty)
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
        selected_mask[best_idx] = True

    selected_items = [pool[i] for i in selected]
    selected_items.sort(key=lambda x: x["score"], reverse=True)

    # Havuz dışındakileri sadece gerekirse ekle
    if len(selected_items) < top_k:
        extras = pool_size
        while len(selected_items) < top_k and extras < len(candidates):
            selected_items.append(candidates[extras])
            extras += 1

    return selected_items[:top_k]


def _dedupe_by_source(candidates: list[dict], chunks: list[str], metadatas: list[dict], top_k: int, diversify_by_url: bool) -> list[dict]:
    if not diversify_by_url:
        return candidates[:top_k]

    out = []
    seen = set()
    for item in candidates:
        idx = item["idx"]
        key = _doc_key(metadatas[idx], chunks[idx])
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= top_k:
            break
    return out


def _build_category_index(index, metadatas: list[dict], category: str) -> tuple[faiss.Index, np.ndarray] | None:
    cat_key = category.lower().strip()
    if not cat_key:
        return None
    cached = _category_index_cache.get(cat_key)
    if cached is not None:
        return cached

    cat_indices = np.array(
        [i for i, m in enumerate(metadatas) if m.get("category", "").lower() == cat_key],
        dtype=np.int64,
    )
    if cat_indices.size == 0:
        return None

    vectors = np.vstack([index.reconstruct(int(i)) for i in cat_indices]).astype(np.float32)
    cat_index = faiss.IndexFlatIP(vectors.shape[1])
    cat_index.add(vectors)
    _category_index_cache[cat_key] = (cat_index, cat_indices)
    return cat_index, cat_indices


def search(
    query: str,
    top_k: int = TOP_K,
    category: str = None,
    diversify_by_url: bool = True,
    date_from: str = None,
    date_to: str = None,
    use_mmr: bool = USE_MMR,
    mmr_lambda: float = MMR_LAMBDA,
    use_reranker: bool = USE_RERANKER,
) -> list[dict]:
    """Sorguya en benzer dokümanları getir (vector + opsiyonel reranker + MMR)."""
    clean_query = _clean_query(query)
    if not clean_query or top_k <= 0:
        return []

    index, chunks, metadatas, config = load_index()
    query_embedding = _resolve_query_embedding(clean_query, config)

    allowed_indices = _get_allowed_indices(metadatas, category, date_from, date_to)
    if allowed_indices is not None and allowed_indices.size == 0:
        return []

    candidate_n = _candidate_k(top_k, int(allowed_indices.size) if allowed_indices is not None else len(chunks))
    if candidate_n <= 0:
        return []

    vector_scores = _vector_candidates(index, query_embedding, candidate_n, allowed_indices)
    if not vector_scores:
        return []

    # Vector scores -> ranked list
    ranked = [{"idx": idx, "score": score} for idx, score in vector_scores.items()]
    ranked.sort(key=lambda x: x["score"], reverse=True)

    if use_reranker:
        ranked = _apply_reranker(clean_query, ranked, chunks)

    if use_mmr:
        ranked = _apply_mmr(index, ranked, max(top_k * 2, top_k), mmr_lambda)

    ranked = _dedupe_by_source(ranked, chunks, metadatas, top_k, diversify_by_url)

    docs = []
    for item in ranked[:top_k]:
        idx = item["idx"]
        docs.append(
            {
                "content": chunks[idx],
                "metadata": metadatas[idx],
                "score": float(item["score"]),
                "rerank_score": float(item.get("rerank_score", 0.0)),
            }
        )
    return docs


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        key = _clean_query(item).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(_clean_query(item))
    return out


def multi_search(
    queries: list[str],
    top_k: int = TOP_K,
    category: str = None,
    date_from: str = None,
    date_to: str = None,
    use_mmr: bool = USE_MMR,
    use_reranker: bool = USE_RERANKER,
) -> list[dict]:
    """Birden fazla sorgu ile arama yap, sonuçları birleştir."""
    if top_k <= 0:
        return []

    unique_queries = _unique_preserve_order(queries or [])
    if not unique_queries:
        return []

    all_docs: dict[str, dict] = {}
    hit_counts: dict[str, int] = {}

    per_query_k = max(4, min(MAX_MULTI_QUERY_PER_QUERY_K, top_k // len(unique_queries) + 3))

    for query in unique_queries:
        docs = search(
            query=query,
            top_k=per_query_k,
            category=category,
            diversify_by_url=True,
            date_from=date_from,
            date_to=date_to,
            use_mmr=use_mmr,
            use_reranker=use_reranker,
        )
        for doc in docs:
            key = _doc_key(doc["metadata"], doc["content"])
            hit_counts[key] = hit_counts.get(key, 0) + 1
            if key not in all_docs or doc["score"] > all_docs[key]["score"]:
                all_docs[key] = doc

    ranked = []
    for key, doc in all_docs.items():
        combined_score = doc["score"] + 0.03 * (hit_counts.get(key, 1) - 1)
        ranked.append((combined_score, doc))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def format_context(
    docs: list[dict],
    max_total_chars: int = CONTEXT_MAX_CHARS,
    max_chars_per_doc: int = CONTEXT_MAX_CHARS_PER_DOC,
) -> str:
    """Dokümanları context string'e çevir (token taşmasını azaltmak için kırpar)."""
    if not docs:
        return ""

    parts = []
    total_chars = 0

    for i, doc in enumerate(docs, 1):
        md = doc.get("metadata", {})
        title = md.get("title", "Bilinmiyor")
        author = md.get("author", "")
        category = md.get("category", "")
        date_text = md.get("date", "")
        content = (doc.get("content") or "").strip()
        if len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc].rsplit(" ", 1)[0].rstrip() + "..."

        header = f"[Kaynak {i}] {title}"
        if author:
            header += f" - {author}"
        if category:
            header += f" ({category})"
        if date_text:
            header += f" [{date_text}]"

        part = f"{header}\n{content}"
        projected = total_chars + len(part)
        if projected > max_total_chars:
            break

        parts.append(part)
        total_chars = projected

    return "\n\n---\n\n".join(parts)


def build_evidence_snippets(
    docs: list[dict],
    cited_indices: list[int] | None = None,
    max_items: int = 3,
    snippet_chars: int = 220,
) -> str:
    """Kaynaklara dayalı kısa kanıt parçaları üret."""
    if not docs:
        return ""

    if cited_indices:
        selected = []
        for n in cited_indices:
            if 1 <= n <= len(docs):
                selected.append((n, docs[n - 1]))
        if not selected:
            selected = [(i + 1, d) for i, d in enumerate(docs[:max_items])]
    else:
        selected = [(i + 1, d) for i, d in enumerate(docs[:max_items])]

    lines = ["\nKanıtlar:"]
    for n, doc in selected[:max_items]:
        text = (doc.get("content") or "").replace("\n", " ").strip()
        if len(text) > snippet_chars:
            text = text[:snippet_chars].rsplit(" ", 1)[0].rstrip() + "..."
        title = doc.get("metadata", {}).get("title", "Bilinmiyor")
        lines.append(f"- [Kaynak {n}] {title}: {text}")
    return "\n".join(lines)


def suggest_categories(query: str, top_n: int = 3, min_chunks: int = 1) -> list[dict]:
    """Sorguya semantik olarak en yakın kategorileri öner (Açıklama bazlı)."""
    clean_q = _clean_query(query)
    if not clean_q:
        return []

    # Config'i yükle
    _, _, _, config = load_index()
    
    # 1. Sorgu vektörü
    q_vec = _resolve_query_embedding(clean_q, config)
    if q_vec is None:
        return []
    
    # 2. Kategori açıklama vektörleri
    desc_vecs = _get_description_embeddings(config)
    if not desc_vecs:
        return []
        
    scores = []
    for cat, d_vec in desc_vecs.items():
        # Cosine similarity (vektörler zaten normalize geliyor _resolve_query_embedding'den)
        # q_vec: (1, dim), d_vec: (1, dim)
        score = float(np.dot(q_vec[0], d_vec[0]))
        scores.append((score, cat))
        
    # Sırala
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Formatla
    suggestions = []
    # Index'ten chunk sayılarını alalım (sadece bilgi amaçlı)
    existing_cats = get_categories(min_chunks=0)
    
    for score, cat in scores[:top_n]:
        suggestions.append({
            "category": cat,
            "score": score,
            "chunks": existing_cats.get(cat, 0) # Eğer indexte yoksa 0 (ama açıklama var)
        })
        
    return suggestions


def get_categories(min_chunks: int = SEMANTIC_CATEGORY_MIN_CHUNKS) -> dict[str, int]:
    """Mevcut kategorileri ve chunk sayılarını döndür (Sadece tanımlı olanlar)."""
    index, _, metadatas, _ = load_index()
    cats = Counter()

    for m in metadatas:
        cat_str = m.get("category")
        if not cat_str:
            continue
            
        # Birden fazla kategori olabilir
        parts = [c.strip() for c in cat_str.replace("/", ",").split(",") if c.strip()]
        for c in parts:
            if c in CATEGORY_DESCRIPTIONS:
                cats[c] += 1

    return {k: v for k, v in sorted(cats.items(), key=lambda x: -x[1]) if v >= min_chunks}


if __name__ == "__main__":
    results = search("Epistemoloji nedir?")
    for doc in results:
        title = doc["metadata"].get("title", "N/A")
        print(f"[{doc['score']:.3f}] {title}")
