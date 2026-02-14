# scrape_onculanalitikfelsefe.py - ASYNC VERSION
# pip install aiohttp beautifulsoup4 lxml tqdm aiofiles

import os
import re
import json
import asyncio
import unicodedata
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm

BASE = "https://onculanalitikfelsefe.com/"
DEFAULT_SEED = "https://onculanalitikfelsefe.com/kategori/etik/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
}

TIMEOUT = 15
RETRIES = 4
CONCURRENT_LIMIT = 5  # daha nazik
MAX_PAGES_PER_CATEGORY = 50  # kategori başına max sayfa
DELAY_BETWEEN_REQUESTS = 0.3  # istek arası bekleme


def safe_name(s: str, max_len: int = 120) -> str:
    """Filesystem-safe folder/file name."""
    s = s.strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if ch not in r'<>:"/\|?*')
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ", "_")
    s = s.strip("._")
    if not s:
        s = "untitled"
    return s[:max_len]


def normalize_url(url: str) -> str:
    p = urlparse(url)
    return p._replace(fragment="").geturl()


def _extract_slug_from_category_url(url: str) -> str | None:
    """Kategori URL'inden slug çıkar."""
    try:
        path = urlparse(url).path.strip("/")
    except Exception:
        return None
    parts = path.split("/")
    if len(parts) >= 2 and parts[0] == "kategori" and parts[1]:
        return parts[1].strip().lower()
    return None


def _extract_post_categories(article: BeautifulSoup, base_url: str = BASE) -> list[str]:
    """Post metadata bölümünden kategori isimlerini çıkar."""
    containers = article.select(
        ".cat-links, .entry-categories, .post-categories, .entry-meta, .post-meta"
    )
    search_roots = containers if containers else [article]

    by_slug: dict[str, str] = {}
    for root in search_roots:
        for a in root.select('a[href*="/kategori/"]'):
            name = a.get_text(" ", strip=True)
            href = normalize_url(urljoin(base_url, a.get("href", "")))
            slug = _extract_slug_from_category_url(href)
            if not slug:
                continue

            # Çok uzun metinler çoğunlukla kategori değil, yazı içi link oluyor.
            if not name or len(name) > 80:
                continue
            if len(name.split()) > 8:
                continue

            if slug not in by_slug:
                by_slug[slug] = name

    return list(by_slug.values())


async def fetch(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        last_err = None
        for attempt in range(RETRIES):
            try:
                await asyncio.sleep(DELAY_BETWEEN_REQUESTS)  # nazik bekleme
                async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as r:
                    if r.status == 403:
                        raise RuntimeError(f"403 Forbidden: {url}")
                    if r.status == 500:
                        raise RuntimeError(f"500 Server Error: {url}")
                    r.raise_for_status()
                    return await r.text()
            except Exception as e:
                last_err = e
                wait_time = (attempt + 1) * 1.0  # exponential backoff
                await asyncio.sleep(wait_time)
        raise RuntimeError(f"Fetch failed for {url}: {last_err}")


def extract_sidebar_categories(html: str, base_url: str = BASE):
    soup = BeautifulSoup(html, "lxml")
    header = soup.find(
        lambda t: t.name in {"h1", "h2", "h3", "h4", "h5"}
        and t.get_text(strip=True).lower() == "kategoriler"
    )

    cat_links = []
    if header:
        ul = header.find_next("ul")
        if ul:
            for a in ul.select("a[href]"):
                name = a.get_text(" ", strip=True)
                href = urljoin(base_url, a["href"])
                if "/kategori/" in href:
                    cat_links.append((name, normalize_url(href)))

    if not cat_links:
        for a in soup.select('a[href*="/kategori/"]'):
            name = a.get_text(" ", strip=True)
            href = urljoin(base_url, a["href"])
            cat_links.append((name, normalize_url(href)))

    seen = set()
    out = []
    for name, href in cat_links:
        if href not in seen:
            seen.add(href)
            out.append((name, href))
    return out


def extract_archive_posts_and_next(html: str, base_url: str = BASE):
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for a in soup.select("h2 a[href], h3 a[href]"):
        href = urljoin(base_url, a.get("href", ""))
        href = normalize_url(href)
        if not href.startswith(base_url):
            continue
        if "/kategori/" in href or "/tag/" in href or "/etiket/" in href:
            continue
        links.add(href)

    next_url = None
    a_next = soup.select_one('a[rel="next"], a.next, a.next.page-numbers')
    if not a_next:
        a_next = soup.find("a", string=re.compile(r"Sonraki", re.IGNORECASE))
    if a_next and a_next.get("href"):
        next_url = normalize_url(urljoin(base_url, a_next["href"]))

    return sorted(links), next_url


@dataclass
class PostData:
    title: str
    author: str
    date: str
    categories: list
    content: str


def parse_post(html: str, url: str, base_url: str = BASE) -> PostData:
    soup = BeautifulSoup(html, "lxml")
    article = soup.find("article") or soup

    h1 = article.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else "Untitled"

    author = ""
    a_author = article.select_one('a[rel="author"], span.author a, a.author, .author a')
    if a_author:
        author = a_author.get_text(" ", strip=True)

    date = ""
    t = article.find("time")
    if t:
        date = t.get("datetime", "") or t.get_text(" ", strip=True)
    else:
        m = re.search(r"\b(\d{1,2}\s+[A-Za-zÇĞİÖŞÜçğıöşü]+\s+\d{4})\b", article.get_text(" ", strip=True))
        if m:
            date = m.group(1)

    categories = _extract_post_categories(article, base_url=base_url)

    content_div = article.find(class_=re.compile(r"(entry-content|post-content|content)", re.I))
    if not content_div:
        content_div = soup.find("main") or article

    for tag in content_div.select("script, style, noscript"):
        tag.decompose()

    content = content_div.get_text("\n", strip=True)

    return PostData(title=title, author=author, date=date, categories=categories, content=content)


async def save_post(root: str, post: PostData, url: str):
    """Her kategori için ayrı klasör oluşturup dosyayı kopyala."""
    categories = post.categories[:] if post.categories else ["Uncategorized"]

    date_part = safe_name(post.date) if post.date else "no_date"
    title_part = safe_name(post.title)
    filename = f"{date_part}__{title_part}.txt"

    header = [
        f"TITLE: {post.title}",
        f"URL: {url}",
        f"DATE: {post.date}",
        f"AUTHOR: {post.author}",
        f"CATEGORIES: {' > '.join(post.categories) if post.categories else ''}",
        "",
        "-----",
        "",
    ]
    content = "\n".join(header) + post.content + "\n"

    saved_paths = []
    for cat in categories:
        out_dir = os.path.join(root, safe_name(cat))
        os.makedirs(out_dir, exist_ok=True)

        path = os.path.join(out_dir, filename)

        if os.path.exists(path):
            base, ext = os.path.splitext(path)
            k = 2
            while os.path.exists(f"{base}__{k}{ext}"):
                k += 1
            path = f"{base}__{k}{ext}"

        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(content)

        saved_paths.append(path)

    return saved_paths


async def collect_all_post_urls(session: aiohttp.ClientSession, categories: list, semaphore: asyncio.Semaphore, max_pages: int | None = None):
    """Tüm kategorilerdeki post URL'lerini topla."""
    all_posts = set()
    pages_limit = max_pages or MAX_PAGES_PER_CATEGORY

    async def crawl_category(cat_name: str, cat_url: str):
        posts = []
        page_url = cat_url
        page_count = 0
        while page_url and page_count < pages_limit:
            try:
                html = await fetch(session, page_url, semaphore)
                post_links, next_url = extract_archive_posts_and_next(html)
                posts.extend(post_links)
                page_count += 1
                print(f"    [{cat_name}] page {page_count}: {len(post_links)} posts")
                page_url = next_url
            except Exception as e:
                print(f"[!] Category page error {page_url}: {e}")
                break
        print(f"  [+] {cat_name}: {len(posts)} posts from {page_count} pages")
        return posts

    tasks = [crawl_category(name, url) for name, url in categories]
    print(f"[i] Crawling {len(categories)} categories (max {pages_limit} pages each)...")
    results = await asyncio.gather(*tasks)

    for post_list in results:
        all_posts.update(post_list)

    return list(all_posts)


async def process_post(session: aiohttp.ClientSession, post_url: str, out_root: str, semaphore: asyncio.Semaphore):
    """Tek bir postu indir ve kaydet."""
    try:
        post_html = await fetch(session, post_url, semaphore)
        post = parse_post(post_html, post_url)
        await save_post(out_root, post, post_url)
        return True
    except Exception as e:
        print(f"[!] Failed post {post_url}: {e}")
        return False


async def main(
    out_root: str = "./oncul_dump",
    seed_url: str = DEFAULT_SEED,
    max_categories: int | None = None,
    max_posts_total: int | None = None,
    update_only: bool = False,
):
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)

    async with aiohttp.ClientSession() as session:
        # 1. Kategorileri al
        seed_html = await fetch(session, seed_url, semaphore)
        categories = extract_sidebar_categories(seed_html)

        if max_categories is not None:
            categories = categories[:max_categories]

        os.makedirs(out_root, exist_ok=True)

        print(f"[i] Found {len(categories)} categories from seed: {seed_url}")
        meta_path = os.path.join(out_root, "_categories.json")
        async with aiofiles.open(meta_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(categories, ensure_ascii=False, indent=2))

        # 2. Post URL'lerini topla (update modunda sadece ilk sayfalar)
        crawl_pages = 3 if update_only else None
        all_post_urls = await collect_all_post_urls(session, categories, semaphore, max_pages=crawl_pages)
        print(f"[i] Found {len(all_post_urls)} unique posts on site")

        # 3. Update modunda mevcut URL'leri çıkar ve filtrele
        if update_only:
            existing = get_existing_urls(out_root)
            print(f"[i] Already have {len(existing)} posts locally")
            new_urls = [u for u in all_post_urls if u not in existing]
            print(f"[i] {len(new_urls)} new posts to download")
            if not new_urls:
                print("[✓] Already up to date!")
                return
            all_post_urls = new_urls

        if max_posts_total is not None:
            all_post_urls = all_post_urls[:max_posts_total]

        # 4. Postları paralel indir
        tasks = [process_post(session, url, out_root, semaphore) for url in all_post_urls]
        results = await tqdm.gather(*tasks, desc="Downloading posts")

        success = sum(1 for r in results if r)
        print(f"[✓] Downloaded {success}/{len(all_post_urls)} posts")


def get_existing_urls(root: str) -> set:
    """Mevcut dosyalardaki URL: satırlarını oku."""
    urls = set()
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.endswith(".txt"):
                continue
            filepath = os.path.join(dirpath, fname)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("URL:"):
                            url = line[4:].strip()
                            if url:
                                urls.add(url)
                            break
                        if line == "-----":
                            break
            except Exception:
                pass
    return urls


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Öncül Analitik Felsefe Scraper")
    parser.add_argument("--update", action="store_true", help="Sadece yeni makaleleri indir")
    parser.add_argument("--full", action="store_true", help="Tüm makaleleri baştan indir")
    parser.add_argument("--out", default="./oncul_dump", help="Çıkış klasörü")
    args = parser.parse_args()

    if args.full:
        asyncio.run(main(out_root=args.out, update_only=False))
    else:
        # Varsayılan: update modu
        asyncio.run(main(out_root=args.out, update_only=True))
