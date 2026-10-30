"""Part 1: turn every kind of source into text Documents.
Supported sources: PDF, DOCX, TXT, source code, PowerPoint (.pptx and old .ppt), web pages, Wikipedia pages and audio.

Every loader returns a list of LangChain Documents: the text, plus metadata that says where it came from.
Example: [Document(page_content="Orbit Bikes is a bike rental company ...",
                   metadata={"source": "orbit_handbook.pdf", "type": "pdf", "page": 1})]

    Loader                  the parent of every loader: text -> a clean Document
        FileLoader          the parent of the file loaders (abstract: every child writes its own load)
            PdfLoader, DocxLoader, PptxLoader -> PptLoader, TextLoader -> CodeLoader, AudioLoader
        WebPageLoader, WikipediaLoader
    SourceLoader            chooses the right loader for a file or a link
"""
import io
import logging
import re
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup
from docx import Document as WordDocument
from langchain_core.documents import Document
from pptx import Presentation
from pypdf import PdfReader

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "EvaluatorGeneratorRAG/1.0 (internship project)"}   # Wikipedia asks every program to name itself
TIMEOUT = 20   # seconds to wait for a web page


# ---------- The parents ----------
class Loader(ABC):
    """The parent of every loader. It turns text into clean LangChain Documents."""

    @staticmethod
    def clean_text(text):
        """Remove spaces at the end of lines and keep at most one empty line between paragraphs."""
        lines = [line.strip() for line in text.splitlines()]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

    def make_document(self, text, **metadata):
        """One Document, only if the text is not empty (an empty page or slide is skipped)."""
        text = self.clean_text(text)
        return [Document(page_content=text, metadata=metadata)] if text else []


class FileLoader(Loader):
    """The parent of the file loaders. Every child says which extensions it reads, and writes its own load."""
    EXTENSIONS = ()

    @abstractmethod
    def load(self, name, data):
        """name: the file name, data: the content of the file as bytes -> a list of Documents."""


# ---------- Files ----------
class PdfLoader(FileLoader):
    """One Document per page. Scanned PDFs (images only) have no text to read."""
    EXTENSIONS = (".pdf",)

    def load(self, name, data):
        documents = []
        for number, page in enumerate(PdfReader(io.BytesIO(data)).pages, start=1):
            documents += self.make_document(page.extract_text() or "", source=name, type="pdf", page=number)
        return documents


class DocxLoader(FileLoader):
    """One Document with the paragraphs, then the tables (each row as "cell | cell | cell")."""
    EXTENSIONS = (".docx",)

    def load(self, name, data):
        word = WordDocument(io.BytesIO(data))
        parts = [paragraph.text for paragraph in word.paragraphs]
        for table in word.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        return self.make_document("\n".join(parts), source=name, type="docx")


class PptxLoader(FileLoader):
    """One Document per slide: the text of every shape, the tables and the speaker notes."""
    EXTENSIONS = (".pptx",)

    def load(self, name, data):
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
            documents += self.make_document("\n".join(parts), source=name, type="pptx", slide=number)
        return documents


class PptLoader(PptxLoader):
    """Old PowerPoint files: LibreOffice converts them to .pptx, then the parent (PptxLoader) reads them."""
    EXTENSIONS = (".ppt",)

    @staticmethod
    def find_libreoffice():
        """The LibreOffice program (None if it is not installed)."""
        windows_path = Path("C:/Program Files/LibreOffice/program/soffice.exe")
        return shutil.which("soffice") or (str(windows_path) if windows_path.exists() else None)

    def load(self, name, data):
        program = self.find_libreoffice()
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
            documents = super().load(name, new_file.read_bytes())   # the .pptx way of the parent
        for document in documents:
            document.metadata["type"] = "ppt"
        return documents


class TextLoader(FileLoader):
    EXTENSIONS = (".txt",)

    @staticmethod
    def decode(data):
        """Bytes of a text file -> a string. Most files are UTF-8; older Arabic Windows files use cp1256."""
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return data.decode("cp1256")

    def load(self, name, data):
        return self.make_document(self.decode(data), source=name, type="txt")


class CodeLoader(TextLoader):
    """Source code is read like text. The language is kept, so part 2 can split the code between functions."""
    # A file extension -> the name of the language, as the code splitter of part 2 knows it
    LANGUAGES = {
        ".py": "python", ".js": "js", ".ts": "ts", ".java": "java", ".c": "c", ".cpp": "cpp", ".cs": "csharp",
        ".go": "go", ".rb": "ruby", ".php": "php", ".kt": "kotlin", ".swift": "swift", ".rs": "rust", ".html": "html",
    }
    EXTENSIONS = tuple(LANGUAGES)

    def load(self, name, data):
        language = self.LANGUAGES[Path(name).suffix.lower()]
        return self.make_document(self.decode(data), source=name, type="code", language=language)


class AudioLoader(FileLoader):
    """Speech to text, so the spoken words become searchable text. The SpeechToText object is given from outside."""
    EXTENSIONS = (".wav", ".mp3", ".m4a")

    def __init__(self, speech):
        self.speech = speech

    def load(self, name, data):
        result = self.speech.transcribe(data)
        return self.make_document(result["text"], source=name, type="audio", language=result["language"])


# ---------- Links ----------
class WebPageLoader(Loader):
    """The readable text of a web page, without its menus, scripts and styles."""

    def load(self, url):
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()   # 404, 500 ... -> an error with the status code
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
            tag.decompose()   # remove the parts that are not the content of the page
        title = soup.title.get_text(strip=True) if soup.title else url
        return self.make_document(soup.get_text("\n"), source=url, type="web", title=title)


class WikipediaLoader(Loader):
    """The plain text of a Wikipedia article, from the Wikipedia API (no HTML to clean)."""

    @staticmethod
    def can_load(url):
        parts = urlparse(url)
        return parts.netloc.endswith("wikipedia.org") and parts.path.startswith("/wiki/")

    def load(self, url):
        parts = urlparse(url)
        language = parts.netloc.split(".")[0]          # "en.wikipedia.org" -> "en", "ar.wikipedia.org" -> "ar"
        title = unquote(parts.path[len("/wiki/"):])     # "/wiki/Alan_Turing" -> "Alan_Turing" (Arabic titles too)
        params = {"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                  "titles": title, "format": "json"}
        response = requests.get(f"https://{language}.wikipedia.org/w/api.php", params=params,
                                headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        page = next(iter(response.json()["query"]["pages"].values()))   # the answer has one page
        if "missing" in page or not page.get("extract"):
            raise ValueError(f"There is no Wikipedia page named {title}.")
        page_url = f"https://{language}.wikipedia.org/wiki/{page['title'].replace(' ', '_')}"
        return self.make_document(page["extract"], source=page_url, type="wikipedia", title=page["title"])


# ---------- The loader that the rest of the project uses ----------
class SourceLoader:
    """Chooses the right loader: for a file from its extension, for a link from the address."""

    def __init__(self, speech):
        self.file_loaders = {}   # ".pdf" -> the PdfLoader object, ".py" -> the CodeLoader object ...
        for loader in [PdfLoader(), DocxLoader(), PptxLoader(), PptLoader(), TextLoader(), CodeLoader(),
                       AudioLoader(speech)]:
            for extension in loader.EXTENSIONS:
                self.file_loaders[extension] = loader
        self.web_page_loader = WebPageLoader()
        self.wikipedia_loader = WikipediaLoader()

    def load_file(self, name, data):
        """A file (its name and its bytes) -> a list of Documents."""
        extension = Path(name).suffix.lower()
        loader = self.file_loaders.get(extension)
        if loader is None:
            raise ValueError(f"The file type {extension or '(none)'} is not supported.")
        documents = loader.load(name, data)
        if not documents:
            raise ValueError(f"No text was found in {name}.")
        logger.info("Loaded %s: %d document(s)", name, len(documents))
        return documents

    def load_url(self, url):
        """A link -> a list of Documents. Wikipedia articles go to the Wikipedia API, other links are web pages."""
        if urlparse(url).scheme not in ("http", "https"):
            raise ValueError("Please give a full link that starts with http:// or https://")
        if self.wikipedia_loader.can_load(url):
            documents = self.wikipedia_loader.load(url)
        else:
            documents = self.web_page_loader.load(url)
        if not documents:
            raise ValueError(f"No text was found at {url}.")
        logger.info("Loaded %s: %d document(s)", url, len(documents))
        return documents


if __name__ == "__main__":
    from src.speech import SpeechToText

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    source_loader = SourceLoader(SpeechToText())

    def show(documents):
        for document in documents:
            preview = document.page_content[:150].replace("\n", " ")
            print(f"   {document.metadata}\n   -> {preview}...")

    # 1) Every file in sample_sources
    for path in sorted(Path("sample_sources").iterdir()):
        print(f"===== {path.name}")
        try:
            show(source_loader.load_file(path.name, path.read_bytes()))
        except Exception as error:
            print("   Error:", error)

    # 2) Links (they need the internet)
    for url in ["https://en.wikipedia.org/wiki/Bicycle-sharing_system",
                "https://ar.wikipedia.org/wiki/دراجة_هوائية",
                "https://www.python.org/about/"]:
        print(f"===== {url}")
        try:
            documents = source_loader.load_url(url)
            print(f"   {len(documents[0].page_content):,} characters")
            show(documents)
        except Exception as error:
            print("   Error:", error)

    # 3) Things that must fail with a clear message
    for name, data in [("notes.xyz", b"hello"), ("empty.txt", b"   ")]:
        try:
            source_loader.load_file(name, data)
        except ValueError as error:
            print(f"===== {name}\n   Error (expected):", error)
