#!/usr/bin/env python3
"""
Öncül Analitik Felsefe RAG CLI

Kullanım:
    python main.py sync      # Yeni makaleleri indir + indexle (tek adım)
    python main.py scrape    # Yeni makaleleri indir (incremental)
    python main.py index     # Dokümanları indexle
    python main.py chat      # Agentic RAG sohbet
    python main.py debate    # Agentic debater (seni çürütür)
    python main.py arena     # İki AI birbirine tartışır
    python main.py ask "..." # Tek soru sor
    python main.py categories "..."  # Semantik kategori öner
    python main.py doctor    # Veri/index sağlık raporu
    python main.py eval      # Retrieval değerlendirme
"""
import sys
import logging

# Gereksiz kütüphane loglarını sustur
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)


def _parse_shared_flags(args: list[str]) -> tuple[dict, list[str]]:
    """chat/debate/ask için ortak flag parser."""
    opts = {
        "date_from": None,
        "date_to": None,
        "category": None,
        "category_picker": False,
        "auto_category": False,
    }
    rest = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "--from" and i + 1 < len(args):
            opts["date_from"] = args[i + 1]
            i += 2
        elif tok == "--to" and i + 1 < len(args):
            opts["date_to"] = args[i + 1]
            i += 2
        elif tok == "--kategori":
            # Eski davranış: seçim menüsü
            opts["category_picker"] = True
            i += 1
        elif tok == "--category" and i + 1 < len(args):
            opts["category"] = args[i + 1]
            i += 2
        elif tok == "--otokategori":
            opts["auto_category"] = True
            i += 1
        else:
            rest.append(tok)
            i += 1
    return opts, rest


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "sync":
        import asyncio
        from scrape import main as scrape_main
        print("=" * 40)
        print("📡 Adım 1/2: Yeni makaleleri indirme...")
        print("=" * 40)
        asyncio.run(scrape_main(update_only=True))
        print()
        print("=" * 40)
        print("📊 Adım 2/2: Yeni makaleleri indexleme...")
        print("=" * 40)
        from rag.indexer import update_index
        update_index()
        print("\n[✓] Sync tamamlandı!")
    
    elif command == "scrape":
        import asyncio
        from scrape import main as scrape_main
        full = "--full" in sys.argv
        asyncio.run(scrape_main(update_only=not full))
    
    elif command == "index":
        if "--full" in sys.argv:
            from rag.indexer import index_documents
            index_documents()
        else:
            from rag.indexer import update_index
            update_index()
    
    elif command == "chat":
        from rag.chat import chat_loop, select_category
        opts, _ = _parse_shared_flags(sys.argv[2:])
        category = opts["category"]
        if opts["category_picker"]:
            category = select_category()
        chat_loop(
            mode="chat",
            category=category,
            auto_category=opts["auto_category"],
            date_from=opts["date_from"],
            date_to=opts["date_to"],
        )
    
    elif command == "debate":
        from rag.chat import chat_loop, select_category
        opts, _ = _parse_shared_flags(sys.argv[2:])
        category = opts["category"]
        if opts["category_picker"]:
            category = select_category()
        chat_loop(
            mode="debate",
            category=category,
            auto_category=opts["auto_category"],
            date_from=opts["date_from"],
            date_to=opts["date_to"],
        )
    
    elif command == "arena":
        from rag.chat import arena_loop
        arena_loop()
    
    elif command == "ask":
        opts, rest = _parse_shared_flags(sys.argv[2:])
        if not rest:
            print("Kullanım: python main.py ask \"Sorunuz\"")
            sys.exit(1)
        query = " ".join(rest)
        from rag.chat import chat
        print(f"\nSoru: {query}\n")
        print("Yanıt: ", end="")
        chat(
            query=query,
            mode="chat",
            category=opts["category"],
            auto_category=opts["auto_category"],
            date_from=opts["date_from"],
            date_to=opts["date_to"],
        )

    elif command == "categories":
        if len(sys.argv) < 3:
            print('Kullanım: python main.py categories "sorgu"')
            sys.exit(1)
        query = " ".join(sys.argv[2:])
        from rag.retriever import suggest_categories

        suggestions = suggest_categories(query, top_n=5)
        if not suggestions:
            print("Kategori önerisi bulunamadı.")
            return
        print("\nSemantik kategori önerileri:")
        for i, s in enumerate(suggestions, 1):
            print(f"  {i}. {s['category']} (score={s['score']:.3f}, chunks={s['chunks']})")

    elif command == "doctor":
        from rag.doctor import run_doctor

        run_doctor()

    elif command == "eval":
        from rag.eval import cli as eval_cli

        eval_cli(sys.argv[2:])
    
    else:
        print(f"Bilinmeyen komut: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
