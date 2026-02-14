# Agents - LLM-as-planner for agentic RAG
import json
import re
from openai import OpenAI
from .config import CHAT_MODEL

_planner_client = None


def get_planner_client():
    global _planner_client
    if _planner_client is None:
        _planner_client = OpenAI()
    return _planner_client


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _is_small_talk(query: str) -> bool:
    q = _normalize_text(query).lower()
    if not q:
        return True

    small_talk_tokens = {
        "selam",
        "merhaba",
        "günaydın",
        "iyi akşamlar",
        "iyi geceler",
        "teşekkürler",
        "teşekkür ederim",
        "sağ ol",
        "sağol",
        "tamam",
        "ok",
        "oke",
        "görüşürüz",
        "bye",
        "hoşçakal",
        "nasılsın",
    }
    if q in small_talk_tokens:
        return True

    if len(q.split()) <= 3 and re.search(r"\b(teşekkür|sağ ol|tamam|ok|bye|görüşürüz)\b", q):
        return True

    return False


def _extract_json_array(text: str) -> list[str] | None:
    raw = (text or "").strip()
    if not raw:
        return None

    # ```json ... ``` gibi blokları temizle
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass

    # Model açıklama + array döndürdüyse ilk array'i yakala
    match = re.search(r"\[[\s\S]*\]", raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        return None
    return None


def _dedupe_queries(queries: list[str], max_items: int = 4) -> list[str]:
    seen = set()
    out = []
    for q in queries:
        clean_q = _normalize_text(q)
        key = clean_q.lower()
        if not clean_q or key in seen:
            continue
        seen.add(key)
        out.append(clean_q)
        if len(out) >= max_items:
            break
    return out


def _quick_llm(prompt: str, max_tokens: int = 200) -> str:
    """Hızlı LLM çağrısı (routing/planning için)."""
    client = get_planner_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_tokens,
        temperature=0
    )
    return response.choices[0].message.content.strip()


def should_use_rag(query: str) -> bool:
    """Sorgu için RAG gerekli mi?"""
    clean_query = _normalize_text(query)
    if _is_small_talk(clean_query):
        return False

    prompt = f"""Aşağıdaki mesaj felsefe, akademik bilgi veya kaynak gerektiren bir soru mu?
Yoksa gündelik sohbet mi (selamlama, teşekkür, vedalaşma, kısa yorum)?

Mesaj: "{clean_query}"

Sadece "RAG" veya "CHAT" yaz, başka bir şey yazma."""

    try:
        result = _quick_llm(prompt, max_tokens=10)
        return "RAG" in result.upper()
    except Exception:
        # Planner arızasında bilgi kaybını azaltmak için RAG'e dön.
        return True


def expand_query(query: str) -> list[str]:
    """Tek sorguyu birden fazla arama sorgusuna genişlet."""
    clean_query = _normalize_text(query)
    if not clean_query:
        return []

    # Çok kısa sorgularda genişletme çoğu zaman gürültü üretir.
    if len(clean_query.split()) <= 3:
        return [clean_query]

    prompt = f"""Aşağıdaki felsefe sorusunu daha iyi yanıtlayabilmek için 3 farklı arama sorgusu üret.
Her sorgu farklı bir açıdan araştırmalı.

Soru: "{clean_query}"

JSON array formatında döndür, sadece array yaz:
["sorgu1", "sorgu2", "sorgu3"]"""

    try:
        result = _quick_llm(prompt, max_tokens=220)
    except Exception:
        return [clean_query]

    queries = _extract_json_array(result)
    try:
        if queries:
            return _dedupe_queries([clean_query] + queries, max_items=4)
    except Exception:
        pass

    return [clean_query]  # fallback: sadece orijinal sorgu


def analyze_argument(argument: str) -> list[str]:
    """Kullanıcının argümanını analiz et ve karşıt arama sorguları üret."""
    clean_arg = _normalize_text(argument)
    if not clean_arg:
        return []

    prompt = f"""Sen bir felsefe tartışmacısısın. Aşağıdaki argümanı analiz et ve en zayıf noktalarını bul.
Bu zayıf noktalara karşı argüman bulmak için 3 arama sorgusu üret.

Argüman: "{clean_arg}"

JSON array formatında döndür, sadece array yaz:
["karşıt_sorgu1", "karşıt_sorgu2", "karşıt_sorgu3"]"""

    try:
        result = _quick_llm(prompt, max_tokens=220)
    except Exception:
        return [clean_arg]

    queries = _extract_json_array(result)
    try:
        if queries:
            return _dedupe_queries(queries, max_items=4)
    except Exception:
        pass

    return [clean_arg]  # fallback


def extract_date_range(query: str) -> tuple[str | None, str | None]:
    """Sorgudan YYYY / tarih aralığı filtreleri çıkar."""
    text = _normalize_text(query).lower()
    if not text:
        return None, None

    # Açık ISO aralığı: 2020-01-01 ... 2022-12-31
    iso = re.findall(r"(19\d{2}|20\d{2})-(\d{2})-(\d{2})", text)
    if len(iso) >= 2:
        d1 = f"{iso[0][0]}-{iso[0][1]}-{iso[0][2]}"
        d2 = f"{iso[1][0]}-{iso[1][1]}-{iso[1][2]}"
        return (d1, d2) if d1 <= d2 else (d2, d1)

    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]
    if not years:
        return None, None

    # "2010-2018", "2010 ile 2018 arası"
    if len(years) >= 2 and re.search(r"(arası|arasi|ile|-)", text):
        y1, y2 = min(years[0], years[1]), max(years[0], years[1])
        return f"{y1:04d}-01-01", f"{y2:04d}-12-31"

    y = years[0]

    # "2010 sonrası", "2010'dan sonra"
    if re.search(r"(sonra|sonrası|sonrasi)", text):
        return f"{y:04d}-01-01", None

    # "2010 öncesi", "2010'dan önce"
    if re.search(r"(önce|once|öncesi|oncesi)", text):
        return None, f"{y:04d}-12-31"

    # Tek yıl -> o yıl
    return f"{y:04d}-01-01", f"{y:04d}-12-31"


def extract_claims(argument: str, max_claims: int = 3) -> list[str]:
    """Argümanı kısa doğrulanabilir iddialara böl."""
    clean_arg = _normalize_text(argument)
    if not clean_arg:
        return []

    # Önce hızlı heuristik
    candidates = [p.strip() for p in re.split(r"(?<=[.!?])\s+", clean_arg) if p.strip()]
    claims = []
    for sent in candidates:
        if len(sent.split()) < 4:
            continue
        claims.append(sent)
        if len(claims) >= max_claims:
            return claims

    # Heuristikten çıkmazsa LLM fallback
    prompt = f"""Aşağıdaki argümandan doğrulanabilir en fazla {max_claims} temel iddia çıkar.
Kısa cümleler üret.

Argüman: "{clean_arg}"

JSON array formatında döndür, sadece array yaz:
["iddia1", "iddia2"]"""

    try:
        result = _quick_llm(prompt, max_tokens=180)
        parsed = _extract_json_array(result)
        if parsed:
            return _dedupe_queries(parsed, max_items=max_claims)
    except Exception:
        pass

    return claims[:max_claims] if claims else [clean_arg]


def find_contradictions(argument: str, context: str, max_items: int = 3) -> list[str]:
    """Argüman ile kaynak bağlam arasındaki çelişki noktalarını çıkar."""
    clean_arg = _normalize_text(argument)
    if not clean_arg or not (context or "").strip():
        return []

    prompt = f"""Kullanıcının argümanı ile kaynaklar arasında olası çelişkileri bul.
Sadece argümandaki iddialarla ilgili kal.
En fazla {max_items} kısa madde üret.

Argüman:
\"\"\"{clean_arg}\"\"\"

Kaynak Bağlamı:
\"\"\"{context[:6000]}\"\"\"

JSON array formatında döndür, sadece array yaz:
["çelişki1", "çelişki2"]"""

    try:
        result = _quick_llm(prompt, max_tokens=220)
        parsed = _extract_json_array(result)
        if parsed:
            return _dedupe_queries(parsed, max_items=max_items)
    except Exception:
        pass

    return []
