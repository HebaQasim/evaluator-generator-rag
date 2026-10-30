"""Part 2: the External Knowledge layer. The Documents of part 1 are split into chunks, embedded and stored in Chroma.
It also lists and removes sources, and gives the retriever that the Generator uses in part 4.

    KeywordIndex   the keyword search (BM25) over the chunks
    KnowledgeBase  add_file / add_url -> load (part 1) -> split into chunks -> embed -> Chroma
                   get_retriever()    -> the best chunks for a question, from a keyword search and a meaning search
"""
import logging
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class KeywordIndex:
    """BM25: finds the chunks that share the most important words with the question.
    A rare word (like "Zarqa") counts more than a common one (like "bike")."""

    # Words that are in almost every question but say nothing about its topic. The keyword search skips them.
    STOP_WORDS = {"a", "an", "the", "is", "are", "was", "what", "which", "who", "when", "where", "why", "how", "much",
                  "many", "do", "does", "did", "i", "you", "my", "of", "to", "in", "on", "for", "with", "and", "or",
                  "can", "will"}

    def __init__(self):
        self.index = None               # the BM25 index (None when there are no chunks)
        self.chunks = []                # the chunks, in the same order as the index
        self.built_for_version = None   # the knowledge version when the index was built

    def words(self, text):
        """'How much does an e-bike cost?' -> ['e', 'bike', 'cost']: small letters, no punctuation, no stop words."""
        return [word for word in re.findall(r"\w+", text.lower()) if word not in self.STOP_WORDS]

    def build(self, chunks, version):
        self.chunks = chunks
        self.index = BM25Okapi([self.words(chunk.page_content) for chunk in chunks]) if chunks else None
        self.built_for_version = version
        logger.info("Built the keyword index (%d chunks)", len(chunks))

    def search(self, question, k):
        """The k chunks with the highest scores. Chunks with no shared word are skipped."""
        if self.index is None:   # the knowledge is empty
            return []
        scores = self.index.get_scores(self.words(question))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in best if scores[i] > 0]


class KnowledgeBase:
    """The External Knowledge: every source that was added, split into chunks and saved in Chroma."""
    TOP_K = 4            # chunks given to the Generator for each question
    RANK_CONSTANT = 60   # the usual constant of Reciprocal Rank Fusion (see _merge_results)

    def __init__(self, source_loader, folder="chroma_database", model_name="intfloat/multilingual-e5-small"):
        self.source_loader = source_loader   # part 1: files and links -> Documents
        self.version_file = Path(folder) / "version.txt"

        # A multilingual model (English, Arabic and about 100 more languages) that runs on the CPU.
        # It was trained with "query: " before questions and "passage: " before texts, so we add them the same way.
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True, "prompt": "passage: "},
            query_encode_kwargs={"normalize_embeddings": True, "prompt": "query: "},
        )
        # Chroma opens the saved database, or creates it the first time
        self.vectordb = Chroma(collection_name="knowledge", embedding_function=self.embedding_model,
                               persist_directory=folder)

        # Text: chunks of about 1,000 characters (well under the 512-token limit of the model).
        # 150 characters are repeated between two chunks, so a sentence cut at the border is still whole in one of them.
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        self.keyword_index = KeywordIndex()

    # ---------- Chunks ----------
    def split_documents(self, documents):
        """Split the Documents into chunks. Code is cut between functions and classes first."""
        chunks = []
        for document in documents:
            if document.metadata["type"] == "code":
                splitter = RecursiveCharacterTextSplitter.from_language(
                    Language(document.metadata["language"]), chunk_size=1500, chunk_overlap=0
                )
            else:
                splitter = self.text_splitter
            chunks += splitter.split_documents([document])   # every chunk keeps the metadata of its Document
        return chunks

    # ---------- The knowledge version ----------
    def version(self):
        """A number that grows every time the knowledge changes. Part 7 puts it in the cache keys,
        so an answer that was cached before a source was added or removed is never used again."""
        return int(self.version_file.read_text()) if self.version_file.exists() else 0

    def _update_version(self):
        self.version_file.parent.mkdir(exist_ok=True)
        self.version_file.write_text(str(self.version() + 1))

    # ---------- Add and remove sources ----------
    def _remove_chunks(self, source):
        """Delete all the chunks of one source. Return how many were deleted."""
        ids = self.vectordb.get(where={"source": source})["ids"]
        if ids:
            self.vectordb.delete(ids=ids)
        return len(ids)

    def add_documents(self, documents):
        """Split, embed and store the Documents. A source that is added again replaces its old chunks."""
        for source in {document.metadata["source"] for document in documents}:
            self._remove_chunks(source)
        chunks = self.split_documents(documents)
        self.vectordb.add_documents(chunks)   # Chroma embeds the chunks with the embedding model
        self._update_version()
        logger.info("Stored %d chunk(s) from %d document(s)", len(chunks), len(documents))
        return len(chunks)

    def add_file(self, name, data):
        """A file (its name and its bytes) -> the knowledge. Return the number of chunks."""
        return self.add_documents(self.source_loader.load_file(name, data))

    def add_url(self, url):
        """A web page or a Wikipedia article -> the knowledge. Return the number of chunks."""
        return self.add_documents(self.source_loader.load_url(url))

    def remove_source(self, source):
        count = self._remove_chunks(source)
        if count == 0:
            raise ValueError(f"There is no source named {source}.")
        self._update_version()
        logger.info("Removed %s (%d chunk(s))", source, count)
        return count

    def list_sources(self):
        """Every source in the knowledge, with its type and its number of chunks."""
        # Example: [{"source": "orbit_faq.txt", "type": "txt", "chunks": 1}, {"source": "orbit_handbook.pdf", ...}]
        sources = {}
        for metadata in self.vectordb.get(include=["metadatas"])["metadatas"]:
            source = metadata["source"]
            if source not in sources:
                sources[source] = {"source": source, "type": metadata["type"], "chunks": 0}
            sources[source]["chunks"] += 1
        return sorted(sources.values(), key=lambda info: info["source"])

    # ---------- Search: keywords + meaning (hybrid search) ----------
    def _all_chunks(self):
        """All the chunks in Chroma, as Documents with their Chroma ids."""
        data = self.vectordb.get(include=["documents", "metadatas"])
        return [Document(id=chunk_id, page_content=text, metadata=metadata)
                for chunk_id, text, metadata in zip(data["ids"], data["documents"], data["metadatas"])]

    def keyword_search(self, question):
        """The keyword index is built again only when the knowledge version has changed."""
        if self.keyword_index.built_for_version != self.version():
            self.keyword_index.build(self._all_chunks(), self.version())
        return self.keyword_index.search(question, self.TOP_K)

    def embed_question(self, question):
        """The embedding of a question. The cache of part 7 compares two questions with it."""
        return self.embedding_model.embed_query(question)

    def meaning_search(self, question):
        """The chunks whose embeddings are closest to the embedding of the question."""
        return self.vectordb.similarity_search(question, k=self.TOP_K)

    def _merge_results(self, results):
        """Reciprocal Rank Fusion. results = {"keywords": [chunks], "meaning": [chunks]}.
        Every list gives a chunk 1 / (60 + its rank) points: rank 1 -> 1/61, rank 2 -> 1/62 ...
        A chunk that both searches found gets points from both lists, so it goes to the top. Return the TOP_K best."""
        points = {}
        chunks = {}
        for found in results.values():
            for rank, chunk in enumerate(found, start=1):
                points[chunk.id] = points.get(chunk.id, 0) + 1 / (self.RANK_CONSTANT + rank)
                chunks[chunk.id] = chunk   # the same chunk has the same id in both lists
        best = sorted(points, key=points.get, reverse=True)[:self.TOP_K]
        return [chunks[chunk_id] for chunk_id in best]

    def get_retriever(self):
        """question -> the TOP_K best chunks. This is a Runnable, so the Generator can put it in its chain.
        The two searches run side by side (RunnableParallel), then _merge_results joins their two lists."""
        both_searches = RunnableParallel(keywords=RunnableLambda(self.keyword_search),
                                         meaning=RunnableLambda(self.meaning_search))
        return (both_searches | RunnableLambda(self._merge_results)).with_config(run_name="Hybrid search")


if __name__ == "__main__":
    from src.loaders import SourceLoader
    from src.speech import SpeechToText

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)   # hide the request lines of the model download

    knowledge = KnowledgeBase(SourceLoader(SpeechToText()))
    print("Knowledge version at the start:", knowledge.version())

    # 1) Add every sample file, and one Wikipedia article (it needs the internet)
    for path in sorted(Path("sample_sources").iterdir()):
        try:
            print(f"{path.name}: {knowledge.add_file(path.name, path.read_bytes())} chunk(s)")
        except ValueError as error:
            print(f"{path.name}: not added ({error})")
    try:
        print("Wikipedia:", knowledge.add_url("https://en.wikipedia.org/wiki/Bicycle-sharing_system"), "chunk(s)")
    except Exception as error:
        print("Wikipedia: not added", error)

    # 2) The .ppt file has the same slides as the .pptx file: remove it, so the answers do not repeat
    try:
        print("Removed the .ppt file:", knowledge.remove_source("orbit_results_2025.ppt"), "chunk(s)")
    except ValueError as error:
        print(error)

    # 3) Adding the same file again replaces it: its number of chunks stays the same
    path = Path("sample_sources/orbit_faq.txt")
    knowledge.add_file(path.name, path.read_bytes())

    print("\nSources:")
    for info in knowledge.list_sources():
        print(f"   {info['source']} | {info['type']} | {info['chunks']} chunk(s)")
    print("Knowledge version at the end:", knowledge.version())

    # 4) Search: the 4 chunks that the Generator will receive, for questions whose answers are in different sources.
    #    [keywords], [meaning] or [keywords + meaning] says which search found each chunk.
    questions = [
        "How much does an e-bike cost per hour?",                 # the PDF
        "What do I get with the Plus plan?",                      # the table in the DOCX
        "When will the Zarqa branch open?",                       # the speaker notes of a slide
        "What time will the Irbid branch close next month?",      # the audio ("Irbid" was heard as "Irvid")
        "How much is the late fee?",                              # the code
        "كم سعر الدراجة الكهربائية بالساعة؟",                       # an Arabic question about an English text
        "What is a bicycle-sharing system?",                      # Wikipedia
    ]
    retriever = knowledge.get_retriever()
    for question in questions:
        print(f"\n{question}")
        keyword_ids = {chunk.id for chunk in knowledge.keyword_search(question)}
        meaning_ids = {chunk.id for chunk in knowledge.meaning_search(question)}
        for rank, chunk in enumerate(retriever.invoke(question), start=1):
            found_by = " + ".join(name for name, ids in [("keywords", keyword_ids), ("meaning", meaning_ids)]
                                  if chunk.id in ids)
            place = chunk.metadata.get("page") or chunk.metadata.get("slide") or ""
            preview = chunk.page_content[:60].replace("\n", " ")
            print(f"   {rank}. {chunk.metadata['source']} {place} [{found_by}] | {preview}")
