"""Part 2: the External Knowledge layer: split the Documents of part 1 into chunks, embed them and store them in Chroma.
It also lists and removes sources, and gives the retriever that the Generator uses in part 4.

    add_file / add_url -> load (part 1) -> split into chunks -> embed -> Chroma
    get_retriever()    -> the best chunks for a question, from a keyword search and a meaning search together
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

from loaders import load_file, load_url

logger = logging.getLogger(__name__)

DB_FOLDER = "chroma_database"
VERSION_FILE = Path(DB_FOLDER) / "version.txt"
TOP_K = 4            # chunks given to the Generator for each question
RANK_CONSTANT = 60   # the usual constant of Reciprocal Rank Fusion (see merge_results)

# Words that are in almost every question but say nothing about its topic. The keyword search skips them.
STOP_WORDS = {"a", "an", "the", "is", "are", "was", "what", "which", "who", "when", "where", "why", "how", "much",
              "many", "do", "does", "did", "i", "you", "my", "of", "to", "in", "on", "for", "with", "and", "or",
              "can", "will"}

# A multilingual model (English, Arabic and about 100 more languages) that runs on the CPU.
# It was trained with "query: " before questions and "passage: " before texts, so we add them the same way.
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    encode_kwargs={"normalize_embeddings": True, "prompt": "passage: "},
    query_encode_kwargs={"normalize_embeddings": True, "prompt": "query: "},
)

# Chroma opens the saved database, or creates it the first time
vectordb = Chroma(collection_name="knowledge", embedding_function=embedding_model, persist_directory=DB_FOLDER)

# Text: chunks of about 1,000 characters (well under the 512-token limit of the model).
# 150 characters are repeated between two chunks, so a sentence cut at the border is still whole in one of them.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def split_documents(documents):
    """Split the Documents into chunks. Code is cut between functions and classes first, like in the HumanEval project."""
    chunks = []
    for document in documents:
        if document.metadata["type"] == "code":
            splitter = RecursiveCharacterTextSplitter.from_language(
                Language(document.metadata["language"]), chunk_size=1500, chunk_overlap=0
            )
        else:
            splitter = text_splitter
        chunks += splitter.split_documents([document])   # every chunk keeps the metadata of its Document
    return chunks


# ---------- The knowledge version ----------
def knowledge_version():
    """A number that grows every time the knowledge changes. Part 7 puts it in the cache keys,
    so an answer that was cached before a source was added or removed is never used again."""
    return int(VERSION_FILE.read_text()) if VERSION_FILE.exists() else 0


def update_version():
    VERSION_FILE.parent.mkdir(exist_ok=True)
    VERSION_FILE.write_text(str(knowledge_version() + 1))


# ---------- Add and remove sources ----------
def remove_chunks(source):
    """Delete all the chunks of one source. Return how many were deleted."""
    ids = vectordb.get(where={"source": source})["ids"]
    if ids:
        vectordb.delete(ids=ids)
    return len(ids)


def add_documents(documents):
    """Split, embed and store the Documents. A source that is added again replaces its old chunks."""
    for source in {document.metadata["source"] for document in documents}:
        remove_chunks(source)
    chunks = split_documents(documents)
    vectordb.add_documents(chunks)   # Chroma embeds the chunks with the embedding model
    update_version()
    logger.info("Stored %d chunk(s) from %d document(s)", len(chunks), len(documents))
    return len(chunks)


def add_file(name, data):
    """A file (its name and its bytes) -> the knowledge. Return the number of chunks."""
    return add_documents(load_file(name, data))


def add_url(url):
    """A web page or a Wikipedia article -> the knowledge. Return the number of chunks."""
    return add_documents(load_url(url))


def remove_source(source):
    count = remove_chunks(source)
    if count == 0:
        raise ValueError(f"There is no source named {source}.")
    update_version()
    logger.info("Removed %s (%d chunk(s))", source, count)
    return count


def list_sources():
    """Every source in the knowledge, with its type and its number of chunks."""
    # Example: [{"source": "orbit_faq.txt", "type": "txt", "chunks": 1}, {"source": "orbit_handbook.pdf", ...}]
    sources = {}
    for metadata in vectordb.get(include=["metadatas"])["metadatas"]:
        source = metadata["source"]
        if source not in sources:
            sources[source] = {"source": source, "type": metadata["type"], "chunks": 0}
        sources[source]["chunks"] += 1
    return sorted(sources.values(), key=lambda info: info["source"])


# ---------- Search: keywords + meaning (hybrid search) ----------
def words(text):
    """'How much does an e-bike cost?' -> ['e', 'bike', 'cost']: small letters, no punctuation, no stop words."""
    return [word for word in re.findall(r"\w+", text.lower()) if word not in STOP_WORDS]


# The keyword index stays in memory. It is built again only when the knowledge version changes.
keyword_index = None
indexed_chunks = []
indexed_version = None


def build_keyword_index():
    """A BM25 index over all the chunks in Chroma (the same chunks that the meaning search uses)."""
    global keyword_index, indexed_chunks, indexed_version
    data = vectordb.get(include=["documents", "metadatas"])
    indexed_chunks = [Document(id=chunk_id, page_content=text, metadata=metadata)
                      for chunk_id, text, metadata in zip(data["ids"], data["documents"], data["metadatas"])]
    keyword_index = BM25Okapi([words(chunk.page_content) for chunk in indexed_chunks]) if indexed_chunks else None
    indexed_version = knowledge_version()
    logger.info("Built the keyword index (%d chunks)", len(indexed_chunks))


def keyword_search(question):
    """BM25: the TOP_K chunks that share the most important words with the question.
    A rare word (like "Zarqa") counts more than a common one (like "bike"). Chunks with no shared word are skipped."""
    if indexed_version != knowledge_version():
        build_keyword_index()
    if keyword_index is None:   # the knowledge is empty
        return []
    scores = keyword_index.get_scores(words(question))
    best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
    return [indexed_chunks[i] for i in best if scores[i] > 0]


def merge_results(results):
    """Reciprocal Rank Fusion. results = {"keywords": [chunks], "meaning": [chunks]}.
    Every list gives a chunk 1 / (60 + its rank) points: rank 1 -> 1/61, rank 2 -> 1/62 ...
    A chunk that both searches found gets points from both lists, so it goes to the top. Return the TOP_K best."""
    points = {}
    chunks = {}
    for found in results.values():
        for rank, chunk in enumerate(found, start=1):
            points[chunk.id] = points.get(chunk.id, 0) + 1 / (RANK_CONSTANT + rank)
            chunks[chunk.id] = chunk   # the same chunk has the same id in both lists
    best = sorted(points, key=points.get, reverse=True)[:TOP_K]
    return [chunks[chunk_id] for chunk_id in best]


def get_retriever():
    """question -> the TOP_K best chunks. This is a Runnable, so the Generator can put it in its chain in part 4.
    The two searches run side by side (RunnableParallel), then merge_results joins their two lists."""
    meaning_search = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    both_searches = RunnableParallel(keywords=RunnableLambda(keyword_search), meaning=meaning_search)
    return both_searches | RunnableLambda(merge_results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)   # hide the request lines of the model download

    print("Knowledge version at the start:", knowledge_version())

    # 1) Add every sample file, and one Wikipedia article (it needs the internet)
    for path in sorted(Path("sample_sources").iterdir()):
        try:
            print(f"{path.name}: {add_file(path.name, path.read_bytes())} chunk(s)")
        except ValueError as error:
            print(f"{path.name}: not added ({error})")
    try:
        print("Wikipedia:", add_url("https://en.wikipedia.org/wiki/Bicycle-sharing_system"), "chunk(s)")
    except Exception as error:
        print("Wikipedia: not added", error)

    # 2) The .ppt file has the same slides as the .pptx file: remove it, so the answers do not repeat
    try:
        print("Removed the .ppt file:", remove_source("orbit_results_2025.ppt"), "chunk(s)")
    except ValueError as error:
        print(error)

    # 3) Adding the same file again replaces it: its number of chunks stays the same
    path = Path("sample_sources/orbit_faq.txt")
    add_file(path.name, path.read_bytes())

    print("\nSources:")
    for info in list_sources():
        print(f"   {info['source']} | {info['type']} | {info['chunks']} chunk(s)")
    print("Knowledge version at the end:", knowledge_version())

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
    retriever = get_retriever()
    for question in questions:
        print(f"\n{question}")
        keyword_ids = {chunk.id for chunk in keyword_search(question)}
        meaning_ids = {chunk.id for chunk in vectordb.similarity_search(question, k=TOP_K)}
        for rank, chunk in enumerate(retriever.invoke(question), start=1):
            found_by = " + ".join(name for name, ids in [("keywords", keyword_ids), ("meaning", meaning_ids)]
                                  if chunk.id in ids)
            place = chunk.metadata.get("page") or chunk.metadata.get("slide") or ""
            preview = chunk.page_content[:60].replace("\n", " ")
            print(f"   {rank}. {chunk.metadata['source']} {place} [{found_by}] | {preview}")
