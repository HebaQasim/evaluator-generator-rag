"""Part 7: the cache. Final answers are kept in Redis, and found again by the MEANING of the question,
so a question that was already answered (even in other words) does not run the whole loop again.

    lookup: question -> its embedding -> compared with every cached question -> the closest one
            -> similar enough? -> the light model checks that the two questions really ask the same thing
            -> yes: the saved answer / no: None
    save:   question + its embedding + the final answer -> Redis

Why the second check: questions like "How much does an e-bike cost per hour?" and "How much does a city bike
cost per hour?" have very close embeddings, but different answers. The embedding finds the candidate quickly and
for free; one short yes/no question to the light model makes sure that a wrong answer is never returned.

The Redis key has the knowledge version (part 2) in it. When a source is added or removed, the version changes,
so the old answers are never used again (Redis deletes them later, after TIME_TO_LIVE).
If Redis is not running, the app still works, only without a cache.
"""
import json
import logging

import redis
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class SemanticCache:
    TIME_TO_LIVE = 7 * 24 * 60 * 60   # seconds: a cached answer is kept for 7 days

    CHECK_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You compare two questions. Reply with only one word: yes or no."),
        ("human", "Do these two questions ask for exactly the same information, so that the same answer is "
                  "correct for both of them? They must also be in the same language, because the answer is "
                  "written in the language of the question.\nQuestion 1: {first}\nQuestion 2: {second}"),
    ])

    def __init__(self, knowledge, llm=None, redis_url="redis://localhost:6379", min_similarity=0.90,
                 prefix="answers"):
        self.knowledge = knowledge            # part 2: the embedding of a question, and the knowledge version
        self.min_similarity = min_similarity  # 1.0 = exactly the same meaning
        self.prefix = prefix
        self.checker = ((self.CHECK_PROMPT | llm | StrOutputParser()).with_config(run_name="Cache check")
                        if llm else None)
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True,
                                          socket_connect_timeout=2, socket_timeout=2)

    def _key(self):
        """One Redis hash for every knowledge version, for example answers:v32 -> {question: entry, ...}"""
        return f"{self.prefix}:v{self.knowledge.version()}"

    @staticmethod
    def similarity(first, second):
        """Cosine similarity. The embeddings are normalized, so it is the dot product."""
        return sum(a * b for a, b in zip(first, second))

    def is_available(self):
        try:
            return self.redis.ping()
        except redis.RedisError:
            return False

    def find_closest(self, question):
        """The cached entry whose question is the closest to this question, and their similarity."""
        entries = self.redis.hvals(self._key())
        if not entries:
            return None, 0.0
        vector = self.knowledge.embed_question(question)
        best_entry, best_score = None, 0.0
        for text in entries:
            entry = json.loads(text)
            score = self.similarity(vector, entry["embedding"])
            if score > best_score:
                best_entry, best_score = entry, score
        return best_entry, best_score

    def same_question(self, question, cached_question):
        """The second check, with the light model (not needed when the two texts are the same)."""
        if question.strip().lower() == cached_question.strip().lower() or self.checker is None:
            return True
        try:
            reply = self.checker.invoke({"first": cached_question, "second": question})
        except Exception as error:   # the model did not answer: do not risk a wrong answer
            logger.warning("The cache check failed, so the cache is skipped (%s)", error)
            return False
        return reply.strip().lower().startswith("yes")

    def lookup(self, question):
        """The saved result of a question with the same meaning, or None."""
        try:
            entry, score = self.find_closest(question)
        except redis.RedisError as error:
            logger.warning("Redis is not available, so the cache is skipped (%s)", error)
            return None
        if entry is None or score < self.min_similarity:
            logger.info("Cache miss for %r (closest similarity %.3f)", question, score)
            return None
        if not self.same_question(question, entry["question"]):
            logger.info("Cache miss for %r: close to %r (%.3f), but not the same question",
                        question, entry["question"], score)
            return None
        logger.info("Cache hit: %r is like the cached %r (similarity %.3f)", question, entry["question"], score)
        return entry["result"]

    def save(self, question, result):
        entry = {"question": question, "embedding": self.knowledge.embed_question(question), "result": result}
        key = self._key()
        try:
            self.redis.hset(key, question, json.dumps(entry))
            self.redis.expire(key, self.TIME_TO_LIVE)
            logger.info("Cached the answer of %r", question)
        except redis.RedisError as error:
            logger.warning("Redis is not available, so the answer was not cached (%s)", error)

    def clear(self):
        """Delete the cached answers of the current knowledge version."""
        try:
            self.redis.delete(self._key())
        except redis.RedisError as error:
            logger.warning("Redis is not available (%s)", error)


if __name__ == "__main__":
    from src.knowledge import KnowledgeBase
    from src.llm import GeminiModels
    from src.loaders import SourceLoader
    from src.speech import SpeechToText

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for name in ("httpx", "httpx2", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    knowledge = KnowledgeBase(SourceLoader(SpeechToText()))
    cache = SemanticCache(knowledge, GeminiModels().fast, prefix="test-answers")   # not the real cache
    print("Redis available:", cache.is_available())

    cached_question = "How much does an e-bike cost per hour?"
    cache.clear()
    cache.save(cached_question, {"answer": "An e-bike costs 6 JOD per hour [1]."})
    print(f"\nCached: {cached_question}   (the check runs from a similarity of {cache.min_similarity})")
    print("   similarity | same question? | result | expected")
    for question, should_hit in [
        ("How much does an e-bike cost per hour?", True),            # the same question
        ("how much does an ebike cost per hour", True),              # the same, written differently
        ("What is the hourly price of an electric bike?", True),     # the same meaning, other words
        ("كم سعر الدراجة الكهربائية بالساعة؟", False),                # the same meaning, but the answer must be Arabic
        ("How much does a city bike cost per hour?", False),         # close, but another bike
        ("How much does an e-bike cost for a full day?", False),     # close, but another price
        ("How much is the deposit for an e-bike?", False),           # another question
        ("Who is the CEO of Orbit Bikes?", False),                   # not related
    ]:
        entry, score = cache.find_closest(question)
        same = cache.same_question(question, entry["question"]) if score >= cache.min_similarity else None
        hit = score >= cache.min_similarity and same
        checked = "-" if same is None else ("yes" if same else "no")
        print(f"   {score:.3f}      | {checked:3}            | {'HIT ' if hit else 'miss'}   | "
              f"{'ok' if hit == should_hit else 'WRONG'}   {question}")
    cache.clear()
