"""Part 1: turn every kind of source into text Documents.
Supported sources: PDF, DOCX, TXT, source code, PowerPoint (.pptx and old .ppt), web pages, Wikipedia pages and audio.

Every loader returns a list of LangChain Documents: the text, plus metadata that says where it came from.
Example: [Document(page_content="Orbit Bikes is a bike rental company ...",
                   metadata={"source": "orbit_handbook.pdf", "type": "pdf", "page": 1})]
Part 2 splits these Documents into chunks, embeds them and stores them, whatever the original format was.
"""
import io
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup
from docx import Document as WordDocument
from langchain_core.documents import Document
from pptx import Presentation
from pypdf import PdfReader

from speech import transcribe

logger = logging.getLogger(__name__)

# A file extension -> the name of the language, as the code splitter of part 2 knows it
CODE_LANGUAGES = {
    ".py": "python", ".js": "js", ".ts": "ts", ".java": "java", ".c": "c", ".cpp": "cpp", ".cs": "csharp",
    ".go": "go", ".rb": "ruby", ".php": "php", ".kt": "kotlin", ".swift": "swift", ".rs": "rust", ".html": "html",
}
AUDIO_TYPES = (".wav", ".mp3", ".m4a")
HEADERS = {"User-Agent": "EvaluatorGeneratorRAG/1.0 (internship project)"}   # Wikipedia asks every program to name itself
TIMEOUT = 20   # seconds to wait for a web page


# ---------- Small helpers ----------
def clean_text(text):
    """Remove spaces at the end of lines and keep at most one empty line between paragraphs."""
    lines = [line.strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def decode_text(data):
    """Bytes of a text file -> a string. Most files are UTF-8; older Arabic Windows files use cp1256."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1256")


def make_document(text, **metadata):
    """One Document, only if the text is not empty (an empty page or slide is skipped)."""
    text = clean_text(text)
    return [Document(page_content=text, metadata=metadata)] if text else []


# ---------- Files ----------
def load_pdf(name, data):
    """One Document per page. Scanned PDFs (images only) have no text to read."""
    documents = []
    for number, page in enumerate(PdfReader(io.BytesIO(data)).pages, start=1):
        documents += make_document(page.extract_text() or "", source=name, type="pdf", page=number)
    return documents


def load_docx(name, data):
    """One Document with the paragraphs, then the tables (each row as "cell | cell | cell")."""
    word = WordDocument(io.BytesIO(data))
    parts = [paragraph.text for paragraph in word.paragraphs]
    for table in word.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return make_document("\n".join(parts), source=name, type="docx")


def load_pptx(name, data):
    """One Document per slide: the text of every shape, the tables and the speaker notes."""
    documents = []
    for number, slide in enumerate(Presentation(io.BytesIO(data)).slides, start=1):
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
            if shape.has_table:
                for row in shape.table.rows:
                    parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        if slide.has_notes_slide:
            parts.append("Notes: " + slide.notes_slide.notes_text_frame.text)
        documents += make_document("\n".join(parts), source=name, type="pptx", slide=number)
    return documents


def find_libreoffice():
    """The LibreOffice program, which can convert old .ppt files to .pptx (None if it is not installed)."""
    windows_path = Path("C:/Program Files/LibreOffice/program/soffice.exe")
    return shutil.which("soffice") or (str(windows_path) if windows_path.exists() else None)


def load_ppt(name, data):
    """Old PowerPoint files: LibreOffice converts them to .pptx in a temporary folder, then load_pptx reads them."""
    program = find_libreoffice()
    if program is None:
        raise ValueError("Reading old .ppt files needs LibreOffice (free, libreoffice.org). "
                         "You can also save the file as .pptx in PowerPoint.")
    with tempfile.TemporaryDirectory() as folder:
        old_file = Path(folder) / "slides.ppt"
        old_file.write_bytes(data)
        subprocess.run([program, "--headless", "--convert-to", "pptx", "--outdir", folder, str(old_file)],
                       capture_output=True, timeout=120)
        new_file = Path(folder) / "slides.pptx"
        if not new_file.exists():
            raise ValueError("LibreOffice could not convert this .ppt file.")
        documents = load_pptx(name, new_file.read_bytes())
    for document in documents:
        document.metadata["type"] = "ppt"
    return documents


def load_text(name, data):
    return make_document(decode_text(data), source=name, type="txt")


def load_code(name, data):
    """The code as it is. The language is kept, so part 2 can split it between functions and classes."""
    language = CODE_LANGUAGES[Path(name).suffix.lower()]
    return make_document(decode_text(data), source=name, type="code", language=language)


def load_audio(name, data, language=None):
    """Speech to text with Faster-Whisper, so the spoken words become searchable text."""
    result = transcribe(data, language)
    return make_document(result["text"], source=name, type="audio", language=result["language"])


def load_file(name, data):
    """Choose the loader from the file extension. data is the content of the file as bytes."""
    extension = Path(name).suffix.lower()
    if extension == ".pdf":
        documents = load_pdf(name, data)
    elif extension == ".docx":
        documents = load_docx(name, data)
    elif extension == ".pptx":
        documents = load_pptx(name, data)
    elif extension == ".ppt":
        documents = load_ppt(name, data)
    elif extension == ".txt":
        documents = load_text(name, data)
    elif extension in CODE_LANGUAGES:
        documents = load_code(name, data)
    elif extension in AUDIO_TYPES:
        documents = load_audio(name, data)
    else:
        raise ValueError(f"The file type {extension or '(none)'} is not supported.")
    if not documents:
        raise ValueError(f"No text was found in {name}.")
    logger.info("Loaded %s: %d document(s)", name, len(documents))
    return documents


# ---------- Links ----------
def load_web_page(url):
    """The readable text of a web page, without its menus, scripts and styles."""
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()   # 404, 500 ... -> an error with the status code
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()   # remove the parts that are not the content of the page
    title = soup.title.get_text(strip=True) if soup.title else url
    return make_document(soup.get_text("\n"), source=url, type="web", title=title)


def load_wikipedia(title, language="en"):
    """The plain text of a Wikipedia article, from the Wikipedia API (no HTML to clean)."""
    api = f"https://{language}.wikipedia.org/w/api.php"
    params = {"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
              "titles": title, "format": "json"}
    response = requests.get(api, params=params, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    page = next(iter(response.json()["query"]["pages"].values()))   # the answer has one page
    if "missing" in page or not page.get("extract"):
        raise ValueError(f"There is no Wikipedia page named {title}.")
    url = f"https://{language}.wikipedia.org/wiki/{page['title'].replace(' ', '_')}"
    return make_document(page["extract"], source=url, type="wikipedia", title=page["title"])


def load_url(url):
    """A link: Wikipedia articles go to the Wikipedia API, every other link is loaded as a web page."""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError("Please give a full link that starts with http:// or https://")
    if parts.netloc.endswith("wikipedia.org") and parts.path.startswith("/wiki/"):
        language = parts.netloc.split(".")[0]           # "en.wikipedia.org" -> "en", "ar.wikipedia.org" -> "ar"
        title = unquote(parts.path[len("/wiki/"):])      # "/wiki/Alan_Turing" -> "Alan_Turing" (Arabic titles too)
        documents = load_wikipedia(title, language)
    else:
        documents = load_web_page(url)
    if not documents:
        raise ValueError(f"No text was found at {url}.")
    logger.info("Loaded %s: %d document(s)", url, len(documents))
    return documents


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    def show(documents):
        for document in documents:
            preview = document.page_content[:150].replace("\n", " ")
            print(f"   {document.metadata}\n   -> {preview}...")

    # 1) Every file in sample_sources
    for path in sorted(Path("sample_sources").iterdir()):
        print(f"===== {path.name}")
        try:
            show(load_file(path.name, path.read_bytes()))
        except Exception as error:
            print("   Error:", error)

    # 2) Links (they need the internet)
    for url in ["https://en.wikipedia.org/wiki/Bicycle-sharing_system",
                "https://ar.wikipedia.org/wiki/دراجة_هوائية",
                "https://www.python.org/about/"]:
        print(f"===== {url}")
        try:
            documents = load_url(url)
            print(f"   {len(documents[0].page_content):,} characters")
            show(documents)
        except Exception as error:
            print("   Error:", error)

    # 3) Things that must fail with a clear message
    for name, data in [("notes.xyz", b"hello"), ("empty.txt", b"   ")]:
        try:
            load_file(name, data)
        except ValueError as error:
            print(f"===== {name}\n   Error (expected):", error)
