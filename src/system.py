"""Part 9: the whole system in one object. It builds every part once and connects them, and the user interface
talks only to this class.

    SpeechToText -> SourceLoader -> KnowledgeBase --+--> Generator (+ its own GeneratorMemory) --+
                                                    |                                           +--> Workflow
    GeminiModels ---------------------------------- +--> Evaluator (+ its own EvaluatorMemory) --+       |
                                                    +--> SemanticCache (Redis) ---------------------------+
"""
import os

from src.cache import SemanticCache
from src.evaluator import Evaluator
from src.generator import Generator
from src.knowledge import KnowledgeBase
from src.llm import GeminiModels
from src.loaders import SourceLoader
from src.memory import EvaluatorMemory, GeneratorMemory
from src.monitoring import Monitoring
from src.speech import SpeechToText
from src.workflow import EvaluatorGeneratorWorkflow


class EvaluatorGeneratorSystem:
    def __init__(self):
        self.monitoring = Monitoring()   # first, so every part below writes its logs to logs/app.log
        models = GeminiModels()
        self.source_loader = SourceLoader(SpeechToText())
        self.knowledge = KnowledgeBase(self.source_loader)
        # Each agent gets its own memory, and never the memory of the other one
        self.generator = Generator(self.knowledge, GeneratorMemory(), models.main, models.fast)
        self.evaluator = Evaluator(EvaluatorMemory(), models.main)
        self.cache = SemanticCache(self.knowledge, models.fast,
                                   redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"))
        self.workflow = EvaluatorGeneratorWorkflow(self.generator, self.evaluator, self.cache)

    # ---------- The knowledge ----------
    def supported_extensions(self):
        return sorted(self.source_loader.file_loaders)

    def add_file(self, name, data):
        return self.knowledge.add_file(name, data)

    def add_url(self, url):
        return self.knowledge.add_url(url)

    def remove_source(self, source):
        return self.knowledge.remove_source(source)

    def list_sources(self):
        return self.knowledge.list_sources()

    # ---------- The questions ----------
    def ask(self, question, session_id):
        return self.workflow.ask(question, session_id)

    def clear_conversation(self, session_id):
        self.generator.memory.clear(session_id)
        self.evaluator.memory.clear(session_id)

    def status(self):
        return {"knowledge_version": self.knowledge.version(), "cache_on": self.cache.is_available(),
                "langsmith_on": self.monitoring.langsmith_on}
