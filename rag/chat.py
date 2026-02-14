# Chat - Agentic RAG sohbet arayüzü
from openai import OpenAI

import re

from .config import CHAT_MODEL, TOP_K
from .retriever import (
    search,
    multi_search,
    format_context,
    get_categories,
    build_evidence_snippets,
    suggest_categories,
)
from .agents import (
    should_use_rag,
    expand_query,
    analyze_argument,
    extract_date_range,
    extract_claims,
    find_contradictions,
)

# Kategori modunda daha fazla sonuç
CATEGORY_TOP_K = 15
MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_CHARS = 9000
AUTO_CATEGORY_SCORE_THRESHOLD = 0.33
APPEND_EVIDENCE_SNIPPETS = True

_chat_client = None


def get_chat_client():
    global _chat_client
    if _chat_client is None:
        _chat_client = OpenAI()
    return _chat_client


SYSTEM_PROMPT = """Sen Türkçe felsefe alanında uzman bir asistansın.
Yanıtlarını aşağıdaki kaynaklardan üret.

Kurallar:
1. Önce kaynak bağlamını kullan. Spekülasyon yapma.
2. İddialarını metindeki kanıtlara dayandır.
3. Yanıtın sonunda en az 1 kaynak etiketi ver: [Kaynak 1], [Kaynak 2]...
4. Kaynakta net bilgi yoksa bunu açıkça söyle ve ardından genel bilgiyle kısa destek ver.
5. Türkçe, net ve akademik bir üslup kullan.

Kaynak dokümanlar:
{context}
"""

SYSTEM_PROMPT_NO_CONTEXT = """Sen Türkçe felsefe alanında uzman bir asistansın.
Kaynak bağlamı bulunamadığında bunu açıkça belirt ve genel felsefe bilgisiyle yardımcı ol.
Türkçe ve net yaz."""

SYSTEM_PROMPT_NO_RAG = """Sen Türkçe felsefe konularında uzman bir asistansın.
Doğal ve samimi şekilde sohbet et. Türkçe yanıt ver."""

DEBATER_PROMPT = """Sen keskin bir felsefe tartışmacısısın.

Kurallar:
1. Sadece kullanıcının verdiği argümana yanıt ver.
2. Her yanıtta tek bir zayıf noktaya odaklan.
3. 3-4 cümleyi geçme.
4. Mümkünse kaynaklardan kanıt kullan ve [Kaynak n] ile belirt.
5. Kişiye değil argümana saldır.
6. Kullanıcı pes ederse tartışmayı kısa ve nazik şekilde kapat.

Kaynaklar:
{context}
"""

DEBATER_PROMPT_NO_CONTEXT = """Sen keskin bir felsefe tartışmacısısın.
Kaynak yoksa bunu bir cümleyle belirt, sonra genel felsefe mantığıyla çürütme yap.
3-4 cümle sınırını koru."""

ARENA_PROMPT_A = """Sen {position} pozisyonunu savunan bir felsefe tartışmacısısın.
Adın: 🔴 KIRMIZI

Kurallar:
1. Pozisyonunu güçlü argümanlarla savun
2. Rakibinin argümanlarını çürüt
3. Kısa ve keskin ol - maksimum 3-4 cümle
4. Kaynaklardaki bilgileri kullan
5. Rakip pes ederse zafer ilan et

Kaynaklar:
{context}
"""

ARENA_PROMPT_B = """Sen {position} pozisyonunu savunan bir felsefe tartışmacısısın.
Adın: 🔵 MAVİ

Kurallar:
1. Pozisyonunu güçlü argümanlarla savun
2. Rakibinin argümanlarını çürüt
3. Kısa ve keskin ol - maksimum 3-4 cümle
4. Kaynaklardaki bilgileri kullan
5. Rakip pes ederse zafer ilan et

Kaynaklar:
{context}
"""


def _trim_history(history: list, max_messages: int = MAX_HISTORY_MESSAGES, max_chars: int = MAX_HISTORY_CHARS) -> list:
    """Geçmişi token taşmasını azaltmak için son mesajlara kırp."""
    if not history:
        return []

    trimmed = history[-max_messages:]
    total = sum(len(msg.get("content", "")) for msg in trimmed)
    while trimmed and total > max_chars:
        removed = trimmed.pop(0)
        total -= len(removed.get("content", ""))
    return trimmed


def _stream_response(client, messages, model=CHAT_MODEL) -> str:
    """Stream a chat completion and return the full response."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )
    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content
    print()
    return full_response


def _extract_cited_sources(text: str) -> list[int]:
    matches = re.findall(r"\[Kaynak\s+(\d+)\]", text or "", flags=re.IGNORECASE)
    nums = []
    seen = set()
    for m in matches:
        n = int(m)
        if n not in seen:
            nums.append(n)
            seen.add(n)
    return nums


def _append_evidence_if_any(response: str, docs: list[dict], stream: bool) -> str:
    if not docs or not APPEND_EVIDENCE_SNIPPETS:
        return response
    cited = _extract_cited_sources(response)
    evidence = build_evidence_snippets(docs, cited_indices=cited, max_items=3)
    if not evidence:
        return response
    if stream:
        print(evidence)
    return f"{response}\n{evidence}"


def chat(
    query: str,
    history: list = None,
    top_k: int = TOP_K,
    stream: bool = True,
    mode: str = "chat",
    category: str = None,
    auto_category: bool = False,
    date_from: str = None,
    date_to: str = None,
) -> str:
    """Agentic RAG chat - akıllı routing + multi-query + kategori filtresi + hafıza."""
    client = get_chat_client()
    docs: list[dict] = []

    if auto_category and not category:
        suggestions = suggest_categories(query, top_n=1)
        if suggestions and suggestions[0]["score"] >= AUTO_CATEGORY_SCORE_THRESHOLD:
            category = suggestions[0]["category"]
            print(f"  🧭 Otomatik kategori: {category}", flush=True)

    inferred_from, inferred_to = extract_date_range(query)
    final_date_from = date_from or inferred_from
    final_date_to = date_to or inferred_to
    if final_date_from or final_date_to:
        print(f"  📅 Tarih filtresi: {final_date_from or '...'} -> {final_date_to or '...'}", flush=True)

    effective_top_k = CATEGORY_TOP_K if category else top_k
    if history is None:
        history = []
    
    if mode == "debate":
        # === AGENTIC DEBATER ===
        print("  🔍 Argüman analiz ediliyor...", flush=True)
        claims = extract_claims(query, max_claims=3)
        counter_queries = analyze_argument(query)
        all_queries = counter_queries + claims
        print(f"  🎯 {len(all_queries)} karşıt arama yapılıyor...", flush=True)
        
        docs = multi_search(
            all_queries,
            top_k=effective_top_k,
            category=category,
            date_from=final_date_from,
            date_to=final_date_to,
        )
        context = format_context(docs)
        if context:
            system_prompt = DEBATER_PROMPT.format(context=context)
            contradictions = find_contradictions(query, context, max_items=3)
            notes = []
            if claims:
                notes.append("Temel iddialar:\n- " + "\n- ".join(claims))
            if contradictions:
                notes.append("Çelişki kontrol notları:\n- " + "\n- ".join(contradictions))
            if notes:
                system_prompt += "\n\nEk notlar:\n" + "\n\n".join(notes)
        else:
            system_prompt = DEBATER_PROMPT_NO_CONTEXT
    
    elif mode == "chat":
        # === SMART ROUTING ===
        use_rag = should_use_rag(query)
        
        if not use_rag:
            print("  💬 Sohbet modu", flush=True)
            system_prompt = SYSTEM_PROMPT_NO_RAG
        else:
            # === MULTI-QUERY RAG ===
            print("  🔍 Sorgular genişletiliyor...", flush=True)
            queries = expand_query(query)
            cat_label = f" [{category}]" if category else ""
            print(f"  📚 {len(queries)} farklı araştırma yapılıyor...{cat_label}", flush=True)
            
            docs = multi_search(
                queries,
                top_k=effective_top_k,
                category=category,
                date_from=final_date_from,
                date_to=final_date_to,
            )
            context = format_context(docs)
            if context:
                system_prompt = SYSTEM_PROMPT.format(context=context)
            else:
                system_prompt = SYSTEM_PROMPT_NO_CONTEXT
    
    else:
        docs = search(
            query,
            top_k=effective_top_k,
            category=category,
            date_from=final_date_from,
            date_to=final_date_to,
        )
        context = format_context(docs)
        if context:
            system_prompt = SYSTEM_PROMPT.format(context=context)
        else:
            system_prompt = SYSTEM_PROMPT_NO_CONTEXT
    
    # Mesajları oluştur: system + geçmiş + yeni soru
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_trim_history(history))
    messages.append({"role": "user", "content": query})
    
    if stream:
        response = _stream_response(client, messages)
    else:
        resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
        response = resp.choices[0].message.content

    return _append_evidence_if_any(response, docs, stream=stream)


def arena_response(messages: list, system_prompt: str, stream: bool = True) -> str:
    """Arena için tek yanıt üret."""
    client = get_chat_client()
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    
    if stream:
        return _stream_response(client, full_messages)
    else:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=full_messages
        )
        return response.choices[0].message.content


def _arena_build_prompt(argument: str, position: str, name: str, color_emoji: str, topic: str) -> str:
    """Arena turunda agentic RAG ile prompt oluştur."""
    print(f"  🔍 {name} argüman analiz ediyor...", flush=True)
    claims = extract_claims(argument, max_claims=3)
    counter_queries = analyze_argument(argument)
    all_queries = counter_queries + claims
    print(f"  🎯 {len(all_queries)} karşıt arama yapılıyor...", flush=True)

    docs = multi_search(all_queries, top_k=TOP_K)
    context = format_context(docs)

    if context:
        prompt = f"""Sen {position} pozisyonunu savunan bir felsefe tartışmacısısın.
Adın: {color_emoji} {name}

Kurallar:
1. Pozisyonunu güçlü argümanlarla savun
2. Rakibinin argümanlarını çürüt
3. Kısa ve keskin ol - maksimum 3-4 cümle
4. Kaynaklardaki bilgileri kullan ve [Kaynak n] ile belirt
5. Rakip pes ederse zafer ilan et

Kaynaklar:
{context}"""
        contradictions = find_contradictions(argument, context, max_items=3)
        if claims:
            prompt += "\n\nRakibin temel iddiaları:\n- " + "\n- ".join(claims)
        if contradictions:
            prompt += "\n\nÇelişki notları:\n- " + "\n- ".join(contradictions)
    else:
        prompt = f"Sen {position} pozisyonunu savun. Kısa, net, felsefi olarak tutarlı yaz."

    return prompt


def arena_loop():
    """İki AI'ın birbiriyle tartıştığı arena modu (agentic RAG destekli)."""
    print("=" * 60)
    print("⚔️  ARENA MODU - Agentic AI Tartışması ⚔️")
    print("=" * 60)
    print()
    
    topic = input("Tartışma konusu: ").strip()
    if not topic:
        print("Konu gerekli!")
        return
    
    print("\nPozisyonları belirle:")
    pos_a = input("🔴 KIRMIZI'nın pozisyonu: ").strip()
    pos_b = input("🔵 MAVİ'nin pozisyonu: ").strip()
    
    if not pos_a or not pos_b:
        print("Her iki pozisyon da gerekli!")
        return
    
    # İlk tur: konu üzerinden RAG
    print("\n  📚 Konu için araştırma yapılıyor...", flush=True)
    queries = expand_query(topic)
    docs = multi_search(queries, top_k=TOP_K)
    initial_context = format_context(docs)

    if initial_context:
        initial_prompt_a = ARENA_PROMPT_A.format(position=pos_a, context=initial_context)
    else:
        initial_prompt_a = f"Sen {pos_a} pozisyonunu savun. Kısa, net, felsefi olarak tutarlı yaz."
    
    history = []
    
    print("\n" + "=" * 60)
    print(f"🔴 KIRMIZI: {pos_a}")
    print(f"🔵 MAVİ: {pos_b}")
    print("=" * 60)
    print("\n[Enter] = sonraki tur, [q] = çıkış\n")
    
    print("🔴 KIRMIZI: ", end="")
    opening = arena_response(
        [{"role": "user", "content": f"Tartışmaya başla. Konu: {topic}. Senin pozisyonun: {pos_a}"}],
        initial_prompt_a
    )
    history.append({"role": "assistant", "content": f"[KIRMIZI] {opening}"})
    
    turn = "blue"
    
    while True:
        try:
            cmd = input("\n[Enter devam, q çık]: ").strip().lower()
            if cmd == "q":
                print("\nARENA BİTTİ!")
                break
            
            print()
            last_msg = history[-1]["content"]
            
            if turn == "blue":
                # MAVİ rakibin (KIRMIZI) argümanını analiz edip karşıt kaynak arıyor
                prompt_b = _arena_build_prompt(last_msg, pos_b, "MAVİ", "🔵", topic)
                print("🔵 MAVİ: ", end="")
                response = arena_response(
                    [{"role": "user", "content": f"Rakibin (KIRMIZI) şunu söyledi: {last_msg}\n\nÇürüt ve kendi pozisyonunu savun."}],
                    prompt_b
                )
                history.append({"role": "assistant", "content": f"[MAVİ] {response}"})
                turn = "red"
            else:
                # KIRMIZI rakibin (MAVİ) argümanını analiz edip karşıt kaynak arıyor
                prompt_a = _arena_build_prompt(last_msg, pos_a, "KIRMIZI", "🔴", topic)
                print("🔴 KIRMIZI: ", end="")
                response = arena_response(
                    [{"role": "user", "content": f"Rakibin (MAVİ) şunu söyledi: {last_msg}\n\nÇürüt ve kendi pozisyonunu savun."}],
                    prompt_a
                )
                history.append({"role": "assistant", "content": f"[KIRMIZI] {response}"})
                turn = "blue"
                
        except KeyboardInterrupt:
            print("\n\nARENA BİTTİ!")
            break
        except Exception as e:
            print(f"\n[!] Hata: {e}")


def chat_loop(
    mode: str = "chat",
    category: str = None,
    auto_category: bool = False,
    date_from: str = None,
    date_to: str = None,
):
    """İnteraktif chat döngüsü (konuşma hafızalı)."""
    print("=" * 60)
    if mode == "debate":
        print("🔥 DEBATER MODU - Agentic Sokrates 🔥")
        print("Görüşlerini söyle, argümanını analiz edip çürüteceğim.")
    else:
        print("🧠 Öncül Analitik Felsefe - Agentic RAG Chat")
        print("Akıllı routing + multi-query araştırma")
    
    if category:
        print(f"📂 Kategori: {category} (top_k={CATEGORY_TOP_K})")
    elif auto_category:
        print("🧭 Otomatik kategori: aktif")

    if date_from or date_to:
        print(f"📅 Sabit tarih filtresi: {date_from or '...'} -> {date_to or '...'}")
    
    print("💾 Konuşma hafızası aktif")
    print("Çıkmak için 'q' veya 'exit' yazın")
    print("=" * 60)
    print()
    
    history = []  # Konuşma geçmişi
    
    while True:
        try:
            prompt = "Görüşün: " if mode == "debate" else "Soru: "
            query = input(prompt).strip()
            if not query:
                continue
            if query.lower() in ("q", "exit", "quit", "çıkış"):
                print("Güle güle!")
                break
            
            print("\nYanıt: ", end="")
            response = chat(
                query,
                history=history,
                mode=mode,
                category=category,
                auto_category=auto_category,
                date_from=date_from,
                date_to=date_to,
            )
            
            # Geçmişe ekle
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})
            
            print()
        except KeyboardInterrupt:
            print("\nGüle güle!")
            break
        except Exception as e:
            print(f"\n[!] Hata: {e}\n")


def select_category() -> str | None:
    """Kategori seçim menüsü."""
    cats = get_categories()
    if not cats:
        print("[!] Kategori bulunamadı")
        return None
    
    print("\n📂 Kategoriler:")
    cat_list = list(cats.items())
    for i, (name, count) in enumerate(cat_list, 1):
        display_name = name.replace("_", " ")
        print(f"  {i:2d}. {display_name} ({count} chunk)")
    print(f"   0. Tüm kategoriler (filtresiz)")
    
    try:
        choice = input("\nKategori seç (numara): ").strip()
        if not choice or choice == "0":
            return None
        idx = int(choice) - 1
        if 0 <= idx < len(cat_list):
            return cat_list[idx][0]
    except (ValueError, IndexError):
        pass
    
    print("Geçersiz seçim, filtresiz devam ediliyor.")
    return None


if __name__ == "__main__":
    chat_loop()
