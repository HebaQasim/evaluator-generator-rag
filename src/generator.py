"""Part 4: the Generator. It finds the chunks for the question, writes an answer from them only,
and keeps every attempt in its own memory (part 3).

    first answer: question -> follow-up? rewrite it -> hybrid search (part 2) -> first prompt -> LLM -> answer
    improvement:  earlier attempts (from its memory) + the Evaluator feedback -> improve prompt -> LLM -> better answer
"""
import logging
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

logger = logging.getLogger(__name__)


class Generator:
    """Writes the answers. It gets everything it needs from outside: the knowledge, its own memory and the models.
    It never gets the Evaluator memory, so it cannot read it."""

    NOT_FOUND = "The information needed to answer this question is not available in the provided sources."
    NO_HISTORY = "(none)"

    # The grounding rules: the same in the first prompt and in the improve prompt
    RULES = (
        "Rules:\n"
        "1. Use only facts that are written in the sources. Do not use your own knowledge and do not guess.\n"
        "2. After every fact, write the number of its source in brackets, like [1].\n"
        f"3. If the sources do not answer the question, reply only with this sentence: {NOT_FOUND} "
        "(in the language of the question).\n"
        "4. If the sources answer only a part of the question, answer that part and say that the rest is not in "
        "the sources.\n"
        "5. Answer in the language of the question, clearly and without extra words."
    )

    FIRST_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are the Generator of a question-answering system. "
                   "Answer the question using only the sources below.\n\n" + RULES + "\n\nSources:\n{context}"),
        ("human", "Earlier conversation (only to understand the question, it is not a source):\n{history}\n\n"
                  "Question: {question}"),
    ])

    IMPROVE_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are the Generator of a question-answering system. You answered the question below, and the "
                   "Evaluator asked you to improve your answer. Write a new answer using only the sources below. "
                   "Fix every point of the latest feedback, and do not bring back a problem that earlier feedback "
                   "has already fixed.\n\n" + RULES + "\n\nSources:\n{context}"),
        ("human", "Question: {question}\n\nYour earlier attempts and the feedback on each one:\n{attempts}\n\n"
                  "Write only the new answer."),
    ])

    # A follow-up question like "And for a full day?" cannot be searched alone, so it is rewritten first
    REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the last question of the user so that it can be understood without the conversation. "
                   "Change it only if it needs the conversation to be understood (for example \"it\", \"this plan\" "
                   "or \"and for a full day?\"). If it can already be understood alone, return it exactly as it is, "
                   "without changing or adding any word. Keep its language and its meaning. "
                   "Return only the question."),
        ("human", "Conversation:\n{history}\n\nLast question: {question}"),
    ])

    def __init__(self, knowledge, memory, llm, fast_llm):
        self.retriever = knowledge.get_retriever()   # part 2: the hybrid search
        self.memory = memory                         # part 3: the Generator memory only
        self.llm = llm                               # for the answers
        self.fast_llm = fast_llm                     # for rewriting follow-up questions
        self.chain = self._build_chain()

    # ---------- What the rest of the project uses ----------
    def generate(self, session_id, run_id, question, attempt=0, feedback=None, search_question=None):
        """One answer. attempt 0 is the first answer; 1 to 4 are improvements with the Evaluator feedback.
        search_question: the question already rewritten by standalone_question (made here when it is not given).
        Returns {"answer": ..., "context": the numbered sources that the answer was written from}."""
        return self.chain.invoke({"session_id": session_id, "run_id": run_id, "question": question,
                                  "attempt": attempt, "feedback": feedback, "search_question": search_question})

    def standalone_question(self, session_id, question):
        """The question as it can be understood alone: a follow-up question is rewritten with the history.
        The workflow asks for it before the loop, because the cache (part 7) compares these questions."""
        history = self._format_history(self.memory.get_history(session_id))
        return self.search_question_chain.invoke({"history": history, "question": question})

    def save_final_answer(self, session_id, run_id, question, answer):
        """After the loop: the answer that the user received becomes part of the conversation history."""
        self.memory.save_turn(session_id, run_id, question, answer)

    # ---------- Helpers ----------
    def _format_history(self, turns):
        if not turns:
            return self.NO_HISTORY
        return "\n".join(f"User: {turn['question']}\nAssistant: {turn['answer']}" for turn in turns)

    @staticmethod
    def _source_label(chunk):
        """orbit_handbook.pdf, page 1 / orbit_results_2025.pptx, slide 3 / a link"""
        label = chunk.metadata["source"]
        if "page" in chunk.metadata:
            label += f", page {chunk.metadata['page']}"
        if "slide" in chunk.metadata:
            label += f", slide {chunk.metadata['slide']}"
        return label

    def _format_context(self, chunks):
        """The chunks as numbered sources, so the answer can point to them: [1] orbit_handbook.pdf, page 1 ..."""
        return "\n\n".join(f"[{number}] {self._source_label(chunk)}\n{chunk.page_content}"
                           for number, chunk in enumerate(chunks, start=1))

    def _find_chunks(self, inputs):
        logger.info("Searching for: %s", inputs["search_question"])
        return self.retriever.invoke(inputs["search_question"])

    def _load_attempts(self, inputs):
        attempts = self.memory.get_attempts(inputs["run_id"])
        if not attempts:
            raise ValueError("An improvement needs the first answer of this run in the memory.")
        return attempts

    @staticmethod
    def _format_attempts(attempts, feedback):
        """Every earlier attempt with the feedback it received. In the memory, each attempt keeps the feedback
        that it tried to fix, so the feedback on attempt 0 is saved with attempt 1, and the newest feedback
        (on the last attempt) is the one that the workflow has just given."""
        received = [attempt["feedback"] for attempt in attempts[1:]] + [feedback]
        return "\n\n".join(f"Attempt {attempt['attempt']}:\n{attempt['answer']}\nFeedback on it: {its_feedback}"
                           for attempt, its_feedback in zip(attempts, received))

    def _save_attempt(self, inputs):
        """Keep the attempt in the Generator memory, and return only what the workflow needs."""
        self.memory.save_attempt(inputs["session_id"], inputs["run_id"], inputs["question"], inputs["attempt"],
                                 inputs["context"], inputs["answer"], inputs["feedback"])
        logger.info("Generator attempt %d saved (run %s)", inputs["attempt"], inputs["run_id"])
        return {"answer": inputs["answer"], "context": inputs["context"]}

    # ---------- The chains ----------
    def _build_chain(self):
        # The search question: the question itself, or a rewritten one when there is an earlier conversation
        self.search_question_chain = RunnableBranch(
            (lambda inputs: inputs["history"] == self.NO_HISTORY, itemgetter("question")),
            self.REWRITE_PROMPT | self.fast_llm | StrOutputParser(),
        )

        first_answer = (
            RunnablePassthrough.assign(
                history=lambda inputs: self._format_history(self.memory.get_history(inputs["session_id"])))
            | RunnablePassthrough.assign(search_question=RunnableBranch(
                (lambda inputs: inputs.get("search_question"), itemgetter("search_question")),   # given by the workflow
                self.search_question_chain,
            ))
            | RunnablePassthrough.assign(chunks=RunnableLambda(self._find_chunks))
            | RunnablePassthrough.assign(context=lambda inputs: self._format_context(inputs["chunks"]))
            | RunnablePassthrough.assign(answer=RunnableBranch(
                (lambda inputs: not inputs["chunks"], lambda inputs: self.NOT_FOUND),   # empty knowledge: no LLM call
                self.FIRST_PROMPT | self.llm | StrOutputParser(),
            ))
        )

        # An improvement does not search again: it uses the same sources, read from the memory,
        # so the Evaluator judges every attempt against the same context
        improvement = (
            RunnablePassthrough.assign(earlier=RunnableLambda(self._load_attempts))
            | RunnablePassthrough.assign(
                context=lambda inputs: inputs["earlier"][0]["context"],
                attempts=lambda inputs: self._format_attempts(inputs["earlier"], inputs["feedback"]),
            )
            | RunnablePassthrough.assign(answer=self.IMPROVE_PROMPT | self.llm | StrOutputParser())
        )

        # Prompt routing: attempt 0 goes to the first prompt, every other attempt goes to the improve prompt
        return (RunnableBranch(
            (lambda inputs: inputs["attempt"] == 0, first_answer),
            improvement,
        ) | RunnableLambda(self._save_attempt)).with_config(run_name="Generator")   # the name in LangSmith


if __name__ == "__main__":
    import uuid

    from src.knowledge import KnowledgeBase
    from src.llm import GeminiModels
    from src.loaders import SourceLoader
    from src.memory import GeneratorMemory
    from src.speech import SpeechToText

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for name in ("httpx", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Make the objects, each one with what it needs
    models = GeminiModels()
    knowledge = KnowledgeBase(SourceLoader(SpeechToText()))
    generator = Generator(knowledge, GeneratorMemory(), models.main, models.fast)

    session = "test-generator"
    generator.memory.clear(session)   # start the test with an empty conversation every time

    def ask(question, attempt=0, feedback=None, run=None):
        run = run or str(uuid.uuid4())
        result = generator.generate(session, run, question, attempt, feedback)
        sources = [line for line in result["context"].splitlines() if line.startswith("[")]
        print(f"\nQ: {question}  (attempt {attempt})\nA: {result['answer']}\nSources: {sources}")
        return run, result

    # 1) A first answer
    run, result = ask("How much does an e-bike cost per hour?")

    # 2) An improvement of the same run, with feedback written by hand (the Evaluator comes in part 5)
    feedback = "Also give the price of a full day for an e-bike, and the hourly price of a city bike."
    run, result = ask("How much does an e-bike cost per hour?", attempt=1, feedback=feedback, run=run)
    print("Attempts in the memory:", [attempt["attempt"] for attempt in generator.memory.get_attempts(run)])
    generator.save_final_answer(session, run, "How much does an e-bike cost per hour?", result["answer"])

    # 3) A follow-up question: it is rewritten with the history before the search (see "Searching for")
    ask("And how much is the deposit for it?")

    # 4) A question that the sources cannot answer
    ask("Who is the CEO of Orbit Bikes?")

    # 5) An Arabic question
    ask("كم سعر الدراجة الكهربائية بالساعة؟")
