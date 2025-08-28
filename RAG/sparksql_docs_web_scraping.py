# This aims to get the Spark SQL API documentation and parse it into a csv file
# This code is used to scrape the Spark SQL API documentation, clean it up,
# and save it in a structured format for further processing or analysis.

import json
import re
import time
from html import unescape as _html_unescape
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, NavigableString, SoupStrainer
from ftfy import fix_text


# Define helper functions for cleaning and normalizing the HTML content
def _strip_tags_outside_code(text: str) -> str:
    """
    Remove any HTML tags only outside fenced code blocks (```...```).
    Also unescapes HTML entities outside code.
    """
    parts = text.split("```")
    for i in range(0, len(parts), 2):  # even indices = outside code
        chunk = _html_unescape(parts[i])
        chunk = re.compile(r"</?[^>]+?>", re.DOTALL).sub("", chunk)
        parts[i] = chunk
    return "```".join(parts)


def _tidy_sig(text: str) -> str:
    """Normalize signature/name strings."""
    text = re.sub(r"\s+([()\[\],])", r"\1", text)  # no space before ()[],,
    text = re.sub(r",\s*", ", ", text)  # space after commas
    text = re.sub(r"^(class)(\S)", r"\1 \2", text)  # 'className' -> 'class Name'
    return text


def _flatten_text(tag) -> str:
    """Single-line text from a tag (no embedded newlines / double spaces)."""
    return re.sub(r"\s+", " ", " ".join(tag.stripped_strings)).strip()


def _lines_from_container(container) -> list[str]:
    """
    Extract '- <head> — <desc>' lines from:
    - <dl><dt>/<dd></dl>
    - <ul>/<ol><li>...</li>
    - <table class="autosummary"><tr><td>head</td><td>desc</td></tr>...</table>
    """
    lines = []
    if container.name == "dl":  # <dl> with <dt> and <dd>
        for dt in container.find_all("dt", recursive=False):  # direct children only
            sig = dt.select_one("dt.sig, span.sig")
            head = _tidy_sig(
                "".join(sig.stripped_strings) if sig else _flatten_text(dt)
            )
            dd = dt.find_next_sibling("dd")
            desc = _flatten_text(dd) if dd else ""
            lines.append(f"- {head} — {desc}" if desc else f"- {head}")
    elif container.name in ("ul", "ol"):  # <ul>/<ol> with <li>
        for li in container.find_all("li", recursive=False):
            item = _flatten_text(li)
            lines.append(f"- {item}")
    elif container.name == "table" and "autosummary" in " ".join(
        container.get("class", [])
    ):  # <table class="autosummary">
        tbody = container.find("tbody") or container
        for tr in tbody.find_all("tr", recursive=False):
            tds = tr.find_all("td", recursive=False)
            if not tds:  # skip empty rows
                continue
            head = _tidy_sig(_flatten_text(tds[0]))
            desc = _flatten_text(tds[1]) if len(tds) > 1 else ""
            lines.append(f"- {head} — {desc}" if desc else f"- {head}")
    return lines


def _rewrite_rubric_sections(
    article: BeautifulSoup, names=("methods", "attributes")
) -> None:
    """Normalize rubric sections like 'Methods' and 'Attributes' into bullet lists."""
    for rub in article.find_all(
        lambda t: t.name in ("p", "div", "h2", "h3")
        and "rubric" in (t.get("class") or [])
        and t.get_text(strip=True).lower() in names
    ):
        cont = rub.find_next_sibling()
        while cont and cont.name not in ("dl", "ul", "ol", "table"):
            cont = cont.find_next_sibling()
        if not cont:
            continue
        lines = _lines_from_container(cont)
        if lines:
            cont.clear()
            cont.append(NavigableString("\n".join(lines)))


def _rewrite_orphan_autosummaries(article: BeautifulSoup) -> None:
    """
    Handle autosummary tables that appear without a rubric/heading
    (e.g., spark_session.html). Leaves ones already rewritten alone.
    """
    for tbl in article.select("table.autosummary"):
        # If a table was already rewritten, it won't have rows anymore.
        if not tbl.find("tr"):
            continue
        lines = _lines_from_container(tbl)
        if not lines:
            continue
        # Infer heading from <caption> or nearest previous rubric/header
        heading = None
        cap = tbl.find("caption")
        if cap:
            heading = cap.get_text(strip=True)
        else:
            prev = tbl.previous_sibling
            while prev and (
                getattr(prev, "name", None) is None or not prev.get_text(strip=True)
            ):
                prev = prev.previous_sibling
            if prev and (
                (prev.name in ("p", "div") and "rubric" in (prev.get("class") or []))
                or prev.name in ("h2", "h3", "h4")
            ):
                heading = prev.get_text(strip=True)
        block = "\n".join(lines) if not heading else f"{heading}\n" + "\n".join(lines)
        tbl.clear()
        tbl.append(NavigableString(block))


# Main function to clean the article content
def clean_article(article: BeautifulSoup) -> str:
    """
    Clean up the HTML article content from Spark SQL docs:
        - remove anchor icons (#) and '[source]' links
        - collapse signatures to one line
        - flatten paragraphs/version notes (no '\n' inside sentences)
        - expand span.classifier -> ' : <type>'
        - rewrite 'Parameters' as one-line bullets
        - rewrite 'Returns' as '<type> — <desc>'
        - rewrite 'Methods' and 'Attributes' to bullets
        - rewrite orphan autosummary tables to bullets
        - keep code blocks as fenced ```python
    """

    # 1) drop '#' anchor icons and '[source]' links
    for a in article.select("a.headerlink"):
        a.decompose()
    for a in article.select("a"):
        if a.get_text(strip=True) == "[source]":
            a.decompose()

    # 2) expand type classifiers to real text
    for cls in article.select("span.classifier"):
        cls.insert_before(NavigableString(" : "))
        cls.unwrap()

    # 3) signatures -> one line
    for sig in article.select("dt.sig"):
        txt = _tidy_sig("".join(sig.stripped_strings))
        sig.clear()
        sig.append(NavigableString(txt))

    # 4) flatten paragraphs & version blocks (avoid newlines in the middle of sentences)
    for blk in article.select(
        "p, div.versionchanged, div.versionadded, div.deprecated"
    ):
        flat = _flatten_text(blk)
        blk.clear()
        blk.append(NavigableString(flat))

    # 5) field-list sections (Parameters / Returns / Notes)
    for dt in article.select("dl.field-list > dt"):
        label = dt.get_text(strip=True).lower()
        dd = dt.find_next_sibling("dd")
        if not dd:
            continue

        if label == "parameters":
            inner = dd.find("dl")
            if inner:
                lines = []
                for pdt in inner.find_all("dt", recursive=False):
                    term = re.sub(r"\s+", " ", " ".join(pdt.stripped_strings))
                    pdd = pdt.find_next_sibling("dd")
                    desc = ""
                    if pdd:
                        parts = []
                        for el in pdd.find_all(["p", "div"], recursive=False):
                            t = _flatten_text(el)
                            if t:
                                parts.append(t)
                        desc = " ".join(parts)
                    lines.append(f"- {term} — {desc}" if desc else f"- {term}")
                dd.clear()
                dd.append(NavigableString("\n".join(lines)))

        elif label == "returns":
            parts = list(dd.stripped_strings)
            if parts:
                rtype = parts[0]
                rdesc = " ".join(parts[1:]).strip()
                dd.clear()
                dd.append(NavigableString(f"{rtype} — {rdesc}"))

        elif label in ("notes", "note"):
            dd.clear()
            dd.append(NavigableString(_flatten_text(dd)))

    # 6) rubric sections (Methods and Attributes)
    _rewrite_rubric_sections(article, names=("methods", "attributes"))

    # 7) orphan autosummary tables
    _rewrite_orphan_autosummaries(article)

    # 8) render code blocks as fenced
    for pre in article.select("pre"):
        code_text = pre.get_text()
        pre.clear()
        pre.append(NavigableString(f"\n```python\n{code_text}\n```\n"))

    # 9) final text cleanup
    text = article.get_text(separator="\n", strip=True)

    # Normalize unicode & spaces
    text = fix_text(text)
    # Normalize NBSP / narrow spaces to plain space; remove soft-hyphen & zero-widths
    text = re.sub(
        r"[\u00A0\u2007\u202F]", " ", text
    )  # NBSP / figure / narrow NBSP -> " "
    text = re.sub(
        r"[\u00AD\u200B\u200C\u200D\uFEFF]", "", text
    )  # SHY, ZW* and BOM -> ""
    # Some pages still leave a lone U+00C2 (Â) after the above; drop it:
    text = text.replace("\u00c2", "")

    # Strip any leftover HTML *outside* code fences (keeps code intact)
    text = _strip_tags_outside_code(text)

    # Tidy headings / whitespace
    text = re.sub(
        r"(?m)^(Parameters|Methods|Attributes|Returns|Notes)\n\1$", r"\1", text
    )
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# Main function to build the JSONL file
def build_jsonl(
    base_url: str = "https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/",
    out_path: str = "data/pyspark_docs.jsonl",
    rps: float = 2.0,  # requests per second
    limit: int | None = None,  # set e.g. 50 for quick tests; None = all
) -> str:
    """
    Scrape Spark SQL API docs, clean them, and save as JSONL.
    Args:
        base_url (str): Base URL of the Spark SQL API documentation.
        out_path (str): Output path for the JSONL file.
        rps (float): Requests per second to avoid overloading the server.
        limit (int | None): Limit the number of pages to scrape; None for all.
    Returns:
        str: Path to the output JSONL file.
    """
    # Index page: get all doc links
    resp = requests.get(base_url, timeout=10)
    resp.raise_for_status()
    index_soup = BeautifulSoup(resp.text, "html.parser")

    # Collect all links to doc pages

    hrefs = {
        urljoin(base_url, a["href"])
        for a in index_soup.select("a[href]")
        if a["href"] and not a["href"].startswith("#")
    }
    # Sort, filter to keep only HTML pages and those starting with base_url
    urls = [u for u in sorted(hrefs) if u.startswith(base_url) and u.endswith(".html")]

    if limit:
        urls = urls[:limit]

    print("Found links on index:", len(index_soup.select("a[href]")))
    print("Kept candidate doc pages:", len(urls))

    # Strainer for the main article body
    # This will only parse the <article.bd-article> part of the page
    # to avoid loading the entire page into memory
    strainer = SoupStrainer("article", {"class": "bd-article"})

    # Crawl each page, clean, and accumulate rows
    print("Crawling pages...")
    rows = []
    for i, page_url in enumerate(urls, 1):
        try:
            r = requests.get(page_url, timeout=10)
            r.raise_for_status()

            # Parse the page with the strainer to get only the article part
            article_soup = BeautifulSoup(r.text, "html.parser", parse_only=strainer)
            article = article_soup.select_one("article.bd-article")
            if not article:
                print(f"[warn] no <article.bd-article> on {page_url}")
                continue
            # Clean the article content
            text = clean_article(article)

            # Grab a title
            full_soup = BeautifulSoup(r.text, "html.parser")
            h1 = full_soup.select_one("h1")
            title = h1.get_text(strip=True) if h1 else page_url.rsplit("/", 1)[-1]
            # Append the row with URL, title, and cleaned text
            rows.append({"url": page_url, "title": title, "text": text})

            # Print progress every 10 pages
            if i % 10 == 0:
                print(f"[{i}/{len(urls)}] {page_url}")
            if rps:
                time.sleep(1.0 / rps)
        except Exception as e:
            print(f"Skip {page_url}: {e}")

    print(f"Collected {len(rows)} pages")

    # Save the results to a JSONL file
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Wrote", out_file)
    return str(out_file)


if __name__ == "__main__":
    build_jsonl(limit=None)
