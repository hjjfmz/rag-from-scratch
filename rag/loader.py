"""
📚 LEARN: Document Loaders — The First Step in RAG
====================================================
Before a RAG system can answer questions, it needs to READ your documents.
That's what this module does: it takes a file path or URL and returns plain text.

Why plain text? Because the next step (chunking) works on strings, and the
step after that (embedding) converts strings into vectors. Everything starts
with getting clean text out of your documents.

This module supports three formats:
  1. Plain text / Markdown files  (.txt, .md)
  2. PDF files                    (.pdf)
  3. Web pages                    (http:// or https://)

Each loader is a simple function: input → string of text.
"""

import os

# 📚 LEARN: We import PyPDF2 for PDF parsing. This is one of our only library
# dependencies. PDFs are complex binary files — they contain fonts, images,
# layout data, and text all mixed together. Writing a PDF parser from scratch
# would be a project on its own, so we use PyPDF2 for this one task.
from PyPDF2 import PdfReader

# 📚 LEARN: requests lets us fetch web pages via HTTP. We could use Python's
# built-in urllib, but requests has a much cleaner API.
import requests

# 📚 LEARN: BeautifulSoup parses HTML and extracts the visible text content,
# stripping away all the HTML tags, scripts, and styles. Web pages are full of
# junk (navigation bars, footers, ads) — we want just the article text.
from bs4 import BeautifulSoup


def load_text_file(file_path: str) -> str:
    """
    Load a plain text or Markdown file and return its contents as a string.

    This is the simplest loader — just read the file. Markdown is already
    mostly plain text, so it works great for RAG without any special parsing.

    Args:
        file_path: Path to a .txt or .md file

    Returns:
        The file's contents as a string
    """
    # 📚 LEARN: We use encoding="utf-8" to handle special characters properly.
    # Without it, Python might use your system's default encoding, which can
    # cause errors on files with accented characters, emoji, etc.
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 📚 LEARN: .strip() removes leading/trailing whitespace. This is a small
    # but important cleanup step — extra blank lines at the start/end of a
    # document would waste space in our chunks later.
    return text.strip()


def load_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file and return it as a single string.

    PDFs store text in a complex binary format with fonts, coordinates, and
    rendering instructions. PyPDF2 does the heavy lifting of extracting the
    actual text content from each page.

    📚 LEARN — LIMITATION: PyPDF2 can only extract TEXT that is stored as
    text in the PDF. It CANNOT read text from images. This means:
      - ✅ Text typed in Word/Google Docs and exported as PDF → works great
      - ✅ Text + images mixed together → text is extracted, images are skipped
      - ❌ Scanned documents (photos of paper) → returns empty text
      - ❌ Text embedded inside diagrams/charts → not extracted

    To handle scanned PDFs, you would need OCR (Optical Character Recognition)
    via a library like pytesseract + Tesseract. That's a possible future
    enhancement but outside the scope of this educational project.

    Args:
        file_path: Path to a .pdf file

    Returns:
        All text from all pages, concatenated with newlines between pages
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(file_path)

    # 📚 LEARN: We loop through every page in the PDF and extract text from
    # each one. Some pages might have no extractable text (e.g., scanned
    # images without OCR). We join pages with double newlines so we don't
    # accidentally merge the last word of one page with the first of the next.
    pages_text = []
    skipped_pages = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            pages_text.append(page_text.strip())
        else:
            skipped_pages.append(i + 1)

    # 📚 LEARN: Warn the user about pages that had no extractable text.
    # These are likely scanned images or pages with only diagrams.
    if skipped_pages:
        print(f"  ⚠️  Skipped {len(skipped_pages)} page(s) with no extractable text "
              f"(pages: {skipped_pages}). These may contain images or scans.")

    return "\n\n".join(pages_text)


def load_web_page(url: str) -> str:
    """
    Fetch a web page and extract its visible text content.

    Web pages are HTML — a mix of content, navigation, scripts, styles, and ads.
    We use BeautifulSoup to parse the HTML and extract just the readable text.

    Args:
        url: A URL starting with http:// or https://

    Returns:
        The visible text content of the page
    """
    # 📚 LEARN: We set a User-Agent header so the web server knows we're a bot.
    # Some servers block requests without a User-Agent, returning 403 Forbidden.
    # Being transparent about what we are is good practice.
    headers = {
        "User-Agent": "RAG-From-Scratch/1.0 (Educational Project)"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()  # Raise an error for 4xx/5xx status codes

    # 📚 LEARN: BeautifulSoup parses the raw HTML string into a tree structure.
    # "html.parser" is Python's built-in HTML parser — no extra dependencies.
    soup = BeautifulSoup(response.text, "html.parser")

    # 📚 LEARN: We remove <script> and <style> tags before extracting text.
    # Without this, we'd get JavaScript code and CSS rules mixed in with
    # the actual content — that would confuse the LLM later.
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # 📚 LEARN: get_text() walks the HTML tree and concatenates all visible text.
    # separator="\n" puts a newline between each element's text so paragraphs
    # don't merge into one giant line.
    text = soup.get_text(separator="\n")

    # Clean up: collapse multiple blank lines into one
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    return text


def load_document(source: str) -> dict:
    """
    Smart loader: auto-detect the source type and load accordingly.

    This is the main entry point for loading documents. It figures out whether
    the source is a URL, PDF, or text file, and calls the right loader.

    Args:
        source: A file path or URL

    Returns:
        A dict with keys:
          - "text": the extracted text content
          - "source": the original source path/URL
          - "type": "text", "pdf", or "web"
    """
    # 📚 LEARN: This function is a simple "dispatcher" pattern. Instead of
    # making the caller figure out which loader to use, we detect the type
    # automatically. This makes the rest of the pipeline simpler — it just
    # calls load_document() and gets text back.

    if source.startswith("http://") or source.startswith("https://"):
        text = load_web_page(source)
        doc_type = "web"
    elif source.lower().endswith(".pdf"):
        text = load_pdf(source)
        doc_type = "pdf"
    else:
        text = load_text_file(source)
        doc_type = "text"

    return {
        "text": text,
        "source": source,
        "type": doc_type,
    }


# 📚 LEARN: The "if __name__ == '__main__'" block runs only when you execute
# this file directly (python loader.py), not when it's imported by other code.
# It's great for quick manual testing during development.
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m rag.loader <file_path_or_url>")
        print("Examples:")
        print("  python -m rag.loader sample_data/sample.md")
        print("  python -m rag.loader sample_data/sample.pdf")
        print("  python -m rag.loader https://example.com")
        sys.exit(1)

    source = sys.argv[1]
    print(f"Loading: {source}")
    print("-" * 60)

    doc = load_document(source)
    print(f"Type: {doc['type']}")
    print(f"Length: {len(doc['text'])} characters")
    print("-" * 60)
    # Show first 500 chars as a preview
    preview = doc["text"][:500]
    print(preview)
    if len(doc["text"]) > 500:
        print(f"\n... ({len(doc['text']) - 500} more characters)")
