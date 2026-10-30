"""Part 8: logging and monitoring.

    Logging:    every file of the project writes its messages with logging.getLogger(__name__).
                Monitoring() sends them to the screen and to logs/app.log, with the time of every message.
                A new log file is started at 1 MB, and the last 3 files are kept.
    Monitoring: LangSmith (its free Developer plan). When these lines are in .env, LangChain sends a trace of every
                question: every step of the chains (Generator, Evaluator, Round, Hybrid search ...), with its input,
                output, time, model calls and errors.
                    LANGSMITH_TRACING=true
                    LANGSMITH_API_KEY=...
                    LANGSMITH_PROJECT=evaluator-generator-rag
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class Monitoring:
    FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    # Libraries that write too many messages: only their warnings and errors are shown
    QUIET_LIBRARIES = ("httpx", "httpx2", "urllib3", "sentence_transformers", "huggingface_hub", "chromadb")

    def __init__(self, log_folder="logs", level=logging.INFO):
        load_dotenv()   # the LangSmith settings are in .env
        Path(log_folder).mkdir(exist_ok=True)
        file_handler = RotatingFileHandler(Path(log_folder) / "app.log", maxBytes=1_000_000, backupCount=3,
                                           encoding="utf-8")
        logging.basicConfig(level=level, format=self.FORMAT, handlers=[logging.StreamHandler(), file_handler],
                            force=True)
        for name in self.QUIET_LIBRARIES:
            logging.getLogger(name).setLevel(logging.WARNING)

        self.langsmith_on = (os.getenv("LANGSMITH_TRACING", "").lower() == "true"
                             and bool(os.getenv("LANGSMITH_API_KEY")))
        if self.langsmith_on:
            logger.info("LangSmith tracing is on (project: %s)", os.getenv("LANGSMITH_PROJECT", "default"))
        else:
            logger.info("LangSmith tracing is off (add LANGSMITH_TRACING=true and LANGSMITH_API_KEY to .env)")


if __name__ == "__main__":
    monitoring = Monitoring()
    logger.info("A test message: it is on the screen and in logs/app.log")
    print("LangSmith on:", monitoring.langsmith_on)
    print("Last lines of logs/app.log:")
    print("".join(Path("logs/app.log").read_text(encoding="utf-8").splitlines(keepends=True)[-2:]))
