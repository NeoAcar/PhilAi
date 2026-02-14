# Indexer - Dokümanları FAISS'e indexle
import os
import json
import pickle
import re
from pathlib import Path
from tqdm import tqdm

import faiss
import numpy as np

from .config import (
    CONTENT_DIR, get_index_path,
    EMBEDDING_PROVIDER,
    OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_DIM,
    LOCAL_EMBEDDING_MODEL, LOCAL_EMBEDDING_DIM,
    USE_GPU, USE_INSTRUCT_FORMAT, INSTRUCT_TASK,
    CHUNK_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP,
    MIN_PARAGRAPH_LENGTH, MAX_PARAGRAPH_LENGTH
)

# Lazy imports
_openai_client = None
_local_model = None


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
        print(f"[i] Loading local embedding model on: {device.upper()}")
        if device == "cuda":
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
        _local_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL, device=device)
    return _local_model


def load_documents(content_dir: Path = CONTENT_DIR) -> list[dict]:
    """Tüm .txt dosyalarını yükle."""
    documents = []
    
    for txt_file in content_dir.rglob("*.txt"):
        try:
            content = txt_file.read_text(encoding="utf-8")
            
            # Metadata parse et
            lines = content.split("\n")
            metadata = {}
            body_start = 0
            
            for i, line in enumerate(lines):
                if line.startswith("TITLE:"):
                    metadata["title"] = line[6:].strip()
                elif line.startswith("URL:"):
                    metadata["url"] = line[4:].strip()
                elif line.startswith("DATE:"):
                    metadata["date"] = line[5:].strip()
                elif line.startswith("AUTHOR:"):
                    metadata["author"] = line[7:].strip()
                elif line.startswith("CATEGORIES:"):
                    metadata["categories"] = line[11:].strip()
                elif line.strip() == "-----":
                    body_start = i + 1
                    break
            
            body = "\n".join(lines[body_start:]).strip()
            metadata["category"] = txt_file.parent.name
            metadata["filename"] = txt_file.name
            metadata["relative_path"] = str(txt_file.relative_to(content_dir))
            
            documents.append({
                "content": body,
                "metadata": metadata
            })
        except Exception as e:
            print(f"[!] Error loading {txt_file}: {e}")
    
    return documents


# =============== CHUNKING STRATEGIES ===============

def chunk_character(text: str) -> list[str]:
    """Karakter bazlı chunking."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + CHUNK_SIZE
        
        if end < len(text):
            last_sentence_end = max(
                text.rfind(".", start, end),
                text.rfind("?", start, end),
                text.rfind("!", start, end)
            )
            if last_sentence_end > start + CHUNK_SIZE // 2:
                end = last_sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - CHUNK_OVERLAP
        if start >= len(text):
            break
    
    return chunks


def chunk_paragraph(text: str) -> list[str]:
    """Paragraf bazlı chunking."""
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not raw_paragraphs:
        return []

    def _split_long_paragraph(paragraph: str) -> list[str]:
        if len(paragraph) <= MAX_PARAGRAPH_LENGTH:
            return [paragraph]

        parts = []
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
        if not sentences:
            return [paragraph[i:i + MAX_PARAGRAPH_LENGTH] for i in range(0, len(paragraph), MAX_PARAGRAPH_LENGTH)]

        current = ""
        for sent in sentences:
            if not current:
                current = sent
                continue

            if len(current) + 1 + len(sent) <= MAX_PARAGRAPH_LENGTH:
                current += " " + sent
            else:
                parts.append(current.strip())
                current = sent

        if current:
            parts.append(current.strip())
        return parts

    segments = []
    for para in raw_paragraphs:
        segments.extend(_split_long_paragraph(para))

    chunks = []
    current_chunk = ""
    for segment in segments:
        if not current_chunk:
            current_chunk = segment
            continue

        projected_len = len(current_chunk) + 2 + len(segment)
        if projected_len <= MAX_PARAGRAPH_LENGTH:
            current_chunk += "\n\n" + segment
        else:
            chunks.append(current_chunk.strip())
            current_chunk = segment

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Çok kısa chunk'ları bir öncekiyle birleştirerek bağlam kaybını azalt.
    merged = []
    for chunk in chunks:
        if merged and len(chunk) < MIN_PARAGRAPH_LENGTH and (len(merged[-1]) + 2 + len(chunk) <= MAX_PARAGRAPH_LENGTH):
            merged[-1] = f"{merged[-1]}\n\n{chunk}"
        else:
            merged.append(chunk)

    return [c for c in merged if c.strip()]


def chunk_document(text: str) -> list[str]:
    """Tüm doküman tek chunk."""
    text = text.strip()
    if text:
        return [text]
    return []


def chunk_text(text: str) -> list[str]:
    """Seçilen stratejiye göre chunk'la."""
    if CHUNK_STRATEGY == "character":
        return chunk_character(text)
    elif CHUNK_STRATEGY == "paragraph":
        return chunk_paragraph(text)
    elif CHUNK_STRATEGY == "document":
        return chunk_document(text)
    else:
        print(f"[!] Unknown chunk strategy: {CHUNK_STRATEGY}, using character")
        return chunk_character(text)


# =============== EMBEDDING ===============

def get_embeddings_openai(texts: list[str]) -> np.ndarray:
    """OpenAI API ile embedding al."""
    client = get_openai_client()
    
    processed_texts = []
    for t in texts:
        t = t.strip()
        if not t:
            t = " "
        if len(t) > 8000:
            t = t[:8000]
        processed_texts.append(t)
    
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=processed_texts
    )
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings, dtype=np.float32)


def get_embeddings_local(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Yerel model ile embedding al."""
    model = get_local_model()
    
    processed_texts = []
    for t in texts:
        t = t.strip() if t.strip() else " "
        
        if USE_INSTRUCT_FORMAT:
            # Turkish E5 instruct format
            if is_query:
                processed_texts.append(f"Instruct: {INSTRUCT_TASK}\nQuery: {t}")
            else:
                # Passage için sadece metin (instruct yok)
                processed_texts.append(t)
        else:
            # Standart E5 format
            prefix = "query: " if is_query else "passage: "
            processed_texts.append(f"{prefix}{t}")
    
    embeddings = model.encode(
        processed_texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.astype(np.float32)


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Seçilen provider ile embedding al."""
    if EMBEDDING_PROVIDER == "openai":
        return get_embeddings_openai(texts)
    elif EMBEDDING_PROVIDER == "local":
        return get_embeddings_local(texts)
    else:
        raise ValueError(f"Unknown embedding provider: {EMBEDDING_PROVIDER}")


def get_embedding_dim() -> int:
    """Aktif embedding boyutunu döndür."""
    if EMBEDDING_PROVIDER == "openai":
        return OPENAI_EMBEDDING_DIM
    else:
        return LOCAL_EMBEDDING_DIM


def _doc_identity(metadata: dict) -> str:
    """Incremental indexing için belge kimliği."""
    url = (metadata.get("url") or "").strip()
    category = (metadata.get("category") or "").strip().lower()
    if url:
        return f"url::{url}::cat::{category}"

    rel = (metadata.get("relative_path") or "").strip()
    if rel:
        return f"path::{rel}"

    filename = (metadata.get("filename") or "").strip()
    return f"file::{filename}::cat::{category}"


# =============== INDEXING ===============

def index_documents(documents: list[dict] = None, batch_size: int = 32):
    """Dokümanları FAISS'e indexle (tam rebuild)."""
    
    if documents is None:
        print("[i] Loading documents...")
        documents = load_documents()
    
    print(f"[i] Loaded {len(documents)} documents")
    print(f"[i] Embedding provider: {EMBEDDING_PROVIDER}")
    print(f"[i] Chunk strategy: {CHUNK_STRATEGY}")
    
    # Chunk'la
    all_chunks = []
    all_metadatas = []
    
    for doc_idx, doc in enumerate(tqdm(documents, desc="Chunking")):
        chunks = chunk_text(doc["content"])
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                **doc["metadata"],
                "chunk_idx": chunk_idx,
                "doc_idx": doc_idx
            })
    
    print(f"[i] Created {len(all_chunks)} chunks")
    if not all_chunks:
        print("[!] Indexlenecek chunk yok.")
        return 0
    
    # Tüm embedding'leri topla
    all_embeddings = []
    
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding"):
        batch_chunks = all_chunks[i:i + batch_size]
        embeddings = get_embeddings(batch_chunks)
        all_embeddings.append(embeddings)
    
    # Birleştir
    embeddings_matrix = np.vstack(all_embeddings)
    
    # FAISS index oluştur
    dim = get_embedding_dim()
    index = faiss.IndexFlatIP(dim)
    
    # Local zaten normalize, OpenAI için normalize et
    if EMBEDDING_PROVIDER == "openai":
        faiss.normalize_L2(embeddings_matrix)
    
    index.add(embeddings_matrix)
    
    # Kaydet
    _save_index(index, all_chunks, all_metadatas)
    
    print(f"[✓] Indexed {len(all_chunks)} chunks to: {get_index_path()}")
    return len(all_chunks)


def update_index(batch_size: int = 32):
    """Sadece yeni dokümanları mevcut index'e ekle."""
    index_path = get_index_path()
    
    # Mevcut index var mı?
    if not (index_path / "index.faiss").exists():
        print("[i] Mevcut index yok, tam index oluşturuluyor...")
        return index_documents()
    
    # Mevcut indexed dosyaları bul
    with open(index_path / "metadatas.pkl", "rb") as f:
        existing_metadatas = pickle.load(f)
    
    indexed_docs = set()
    for m in existing_metadatas:
        indexed_docs.add(_doc_identity(m))
    
    # Tüm dokümanları yükle ve yeni olanları filtrele
    print("[i] Loading documents...")
    all_docs = load_documents()
    new_docs = [d for d in all_docs if _doc_identity(d["metadata"]) not in indexed_docs]
    
    if not new_docs:
        print(f"[✓] Index güncel! ({len(indexed_docs)} doküman zaten indexte)")
        return 0
    
    print(f"[i] {len(new_docs)} new documents to index (skipping {len(indexed_docs)} existing)")
    print(f"[i] Embedding provider: {EMBEDDING_PROVIDER}")
    print(f"[i] Chunk strategy: {CHUNK_STRATEGY}")
    
    # Yeni dokümanları chunk'la
    new_chunks = []
    new_metadatas = []
    base_doc_idx = len(indexed_docs)
    
    for doc_idx, doc in enumerate(tqdm(new_docs, desc="Chunking")):
        chunks = chunk_text(doc["content"])
        for chunk_idx, chunk in enumerate(chunks):
            new_chunks.append(chunk)
            new_metadatas.append({
                **doc["metadata"],
                "chunk_idx": chunk_idx,
                "doc_idx": base_doc_idx + doc_idx
            })
    
    print(f"[i] Created {len(new_chunks)} new chunks")
    if not new_chunks:
        print("[i] Yeni chunk oluşmadı, index değişmedi.")
        return 0
    
    # Embedding'leri al
    new_embeddings = []
    for i in tqdm(range(0, len(new_chunks), batch_size), desc="Embedding"):
        batch = new_chunks[i:i + batch_size]
        embeddings = get_embeddings(batch)
        new_embeddings.append(embeddings)
    
    new_embeddings_matrix = np.vstack(new_embeddings)
    
    if EMBEDDING_PROVIDER == "openai":
        faiss.normalize_L2(new_embeddings_matrix)
    
    # Mevcut index'i yükle ve genişlet
    index = faiss.read_index(str(index_path / "index.faiss"))
    
    with open(index_path / "chunks.pkl", "rb") as f:
        existing_chunks = pickle.load(f)
    
    index.add(new_embeddings_matrix)
    all_chunks = existing_chunks + new_chunks
    all_metadatas = existing_metadatas + new_metadatas
    
    # Kaydet
    _save_index(index, all_chunks, all_metadatas)
    
    print(f"[✓] Added {len(new_chunks)} chunks (total: {len(all_chunks)}) to: {index_path}")
    return len(new_chunks)


def _save_index(index, chunks, metadatas):
    """Index ve verileri diske kaydet."""
    index_path = get_index_path()
    index_path.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(index_path / "index.faiss"))
    
    with open(index_path / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    with open(index_path / "metadatas.pkl", "wb") as f:
        pickle.dump(metadatas, f)
    
    # Config kaydet
    with open(index_path / "config.json", "w") as f:
        json.dump({
            "embedding_provider": EMBEDDING_PROVIDER,
            "embedding_model": LOCAL_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "local" else OPENAI_EMBEDDING_MODEL,
            "embedding_dim": get_embedding_dim(),
            "chunk_strategy": CHUNK_STRATEGY,
            "num_chunks": len(chunks)
        }, f, indent=2)


if __name__ == "__main__":
    index_documents()
