# RAG Configuration
import os
from pathlib import Path
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
CONTENT_DIR = BASE_DIR / "oncul_dump"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"

# =============== EMBEDDING SETTINGS ===============
# Provider: "openai" veya "local"
EMBEDDING_PROVIDER = "local"

# OpenAI embedding (EMBEDDING_PROVIDER="openai" için)
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIM = 1536

# Local embedding (EMBEDDING_PROVIDER="local" için)
# Seçenekler:
#   - "intfloat/multilingual-e5-large" (genel multilingual)
#   - "ytu-ce-cosmos/turkish-e5-large" (Türkçe fine-tuned)
LOCAL_EMBEDDING_MODEL = "ytu-ce-cosmos/turkish-e5-large"
LOCAL_EMBEDDING_DIM = 1024

# Instruct format (turkish-e5-large için gerekli)
# True: Instruct format kullan (turkish-e5 için önerilen)
# False: Sadece passage:/query: prefix kullan (multilingual-e5 için)
USE_INSTRUCT_FORMAT = True
INSTRUCT_TASK = "Given a Turkish search query, retrieve relevant passages written in Turkish that best answer the query"

# GPU kullan (True) veya CPU (False)
USE_GPU = True


def get_index_path() -> Path:
    """Aktif provider/model için index klasörü."""
    if EMBEDDING_PROVIDER == "openai":
        name = OPENAI_EMBEDDING_MODEL.replace("/", "_")
    else:
        name = LOCAL_EMBEDDING_MODEL.replace("/", "_")
    return FAISS_INDEX_DIR / f"{EMBEDDING_PROVIDER}_{name}"


# =============== LLM SETTINGS ===============
CHAT_MODEL = "gpt-5.2"

# =============== CHUNKING SETTINGS ===============
# Options: "character", "paragraph", "document"
CHUNK_STRATEGY = "paragraph"

# Character chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Paragraph chunking
MIN_PARAGRAPH_LENGTH = 100
MAX_PARAGRAPH_LENGTH = 3000

# =============== RETRIEVAL SETTINGS ===============
TOP_K = 5

# MMR çeşitlendirme
USE_MMR = True
MMR_LAMBDA = 0.72

# Reranker (opsiyonel, yavaş ama daha isabetli)
USE_RERANKER = False
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 30
RERANK_WEIGHT = 0.25

# Semantic kategori öneri
SEMANTIC_CATEGORY_MIN_CHUNKS = 10

# Otomatik kategori tespiti için açıklamalar (Zero-shot routing)
CATEGORY_DESCRIPTIONS = {
    "Zihin_Felsefesi": "Bilinç, qualia, yapay zeka, benlik, fizikalizm, dualizm, zihin-beden problemi, psikoloji felsefesi.",
    "Din_Felsefesi": "Tanrı'nın varlığı, ateizm, teizm, deizm, kötülük problemi, inanç, din dili, mucizeler, teoloji.",
    "Etik": "Ahlak, iyi ve kötü, erdem, faydacılık, deontoloji, meta-etik, uygulamalı etik, kürtaj, ötanazi, hayvan hakları.",
    "Epistemoloji": "Bilgi felsefesi, inanç, gerekçelendirme, şüphecilik, doğruluk, algı, bilgi kaynakları, gettier problemi.",
    "Metafizik": "Varlık, töz, zaman, mekan, özgür irade, determinizm, nedensellik, mümkün dünyalar, ontoloji.",
    "Siyaset_Felsefesi": "Devlet, adalet, özgürlük, haklar, liberalizm, sosyalizm, demokrasi, toplumsal sözleşme, otorite.",
    "Bilim_Felsefesi": "Bilimsel yöntem, paradigma, yanlışlanabilirlik, bilimsel gerçekçilik, sözde bilim, doğa yasaları.",
    "Sanat_Felsefesi": "Estetik, güzellik, sanatın tanımı, beğeni yargıları, sanat eleştirisi.",
    "Hukuk_Felsefesi": "Yasa, adalet, ceza, hukuk devleti, doğal hukuk, hukuki pozitivizm.",
    "İyi_Oluş_&_Hayatın_Anlamı": "Mutluluk, yaşamın anlamı, ölüm, eudaimonia, yaşam sanatı, varoluşsal sorunlar.",
    "Mantık": "Akıl yürütme, safsatalar, sembolik mantık, önermeler, çıkarım kuralları, paradokslar.",
    "Felsefe_Tarihi": "Antik felsefe, modern felsefe, filozoflar tarihi, felsefi akımların gelişimi.",
}
