"""Part 5: the Evaluator. It judges an answer of the Generator against the same sources, with the 7 criteria
of its own memory (part 3), decides "accept" or "improve", and writes feedback for the Generator.

    question + answer + sources + its earlier evaluations (from its memory) -> prompt -> LLM -> JSON scores + feedback
    -> check the JSON -> compare every score with the minimum of its criterion -> decision -> save in its memory
"""
import logging

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

logger = logging.getLogger(__name__)


class Evaluator:
    """Judges the answers. It gets its own memory and a model from outside.
    It never gets the Generator memory: the only things it sees are the question, the answer and the sources."""

    NO_EARLIER = "(none)"

    PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are the Evaluator of a question-answering system. The Generator wrote an answer to the "
                   "user's question from the sources below. Judge the answer only against these sources. "
                   "Do not use your own knowledge.\n\n"
                   "Give every criterion a whole score from 1 (very poor) to 5 (perfect):\n{criteria}\n\n"
                   "Rules:\n"
                   "1. If the sources do not contain the information, the correct answer says that it is not "
                   "available in the provided sources. Such an answer gets 5 in every criterion.\n"
                   "2. If the answer says that the information is not available, but the sources do contain it, "
                   "give low scores for accuracy and completeness, and say in the feedback which source has it.\n"
                   "3. Never ask for information that the sources do not have.\n"
                   "4. The feedback must say exactly what to change, with the source numbers, like [2]. "
                   "If nothing needs to change, the feedback is an empty string.\n"
                   "5. If you gave feedback before, check that the new answer fixed it, and do not ask again for "
                   "something that is already fixed.\n\n"
                   "Return only a JSON object with two keys: \"scores\", an object with one score for each "
                   "criterion name above, and \"feedback\", a string.\n\n"
                   "Sources:\n{context}"),
        ("human", "Question: {question}\n\nAnswer to judge:\n{answer}\n\n"
                  "Your earlier evaluations of this question:\n{earlier}"),
    ])

    def __init__(self, memory, llm):
        self.memory = memory   # part 3: the Evaluator memory only
        self.llm = llm
        self.criteria = self.memory.get_criteria()   # the criteria and rules, read from its own memory
        self.criteria_text = "\n".join(f"- {criterion['name']}: {criterion['description']}"
                                       for criterion in self.criteria)
        self.chain = self._build_chain()

    # ---------- What the rest of the project uses ----------
    def evaluate(self, session_id, run_id, question, attempt, answer, context):
        """Judge one answer. context is the numbered sources that the Generator used.
        Returns {"scores": {...}, "decision": "accept" or "improve", "feedback": ..., "failed": [...]}."""
        return self.chain.invoke({"session_id": session_id, "run_id": run_id, "question": question,
                                  "attempt": attempt, "answer": answer, "context": context})

    # ---------- Helpers ----------
    def _format_earlier(self, evaluations):
        """The earlier evaluations of this question, so the Evaluator can check that its feedback was fixed."""
        if not evaluations:
            return self.NO_EARLIER
        return "\n".join(f"Attempt {evaluation['attempt']}: {evaluation['decision']}, "
                         f"feedback: {evaluation['feedback'] or '(none)'}" for evaluation in evaluations)

    def _check_result(self, result):
        """The LLM must return a score from 1 to 5 for every criterion. If it did not, an error is raised,
        and the judge is asked once more (see with_retry in _build_chain)."""
        scores = result.get("scores") if isinstance(result, dict) else None
        if not isinstance(scores, dict):
            raise ValueError("The Evaluator did not return any scores.")
        checked = {}
        for criterion in self.criteria:
            score = scores.get(criterion["name"])
            if not isinstance(score, (int, float)) or not 1 <= score <= 5:
                raise ValueError(f"The Evaluator did not return a score from 1 to 5 for {criterion['name']}.")
            checked[criterion["name"]] = int(score)
        return {"scores": checked, "feedback": str(result.get("feedback") or "").strip()}

    def _decide_and_save(self, inputs):
        """The decision is made here in the code, not by the LLM: every score must reach the minimum of its
        criterion. Then the evaluation is saved in the Evaluator memory."""
        scores = inputs["result"]["scores"]
        feedback = inputs["result"]["feedback"]
        failed = [f"{criterion['name']} ({scores[criterion['name']]}/5)" for criterion in self.criteria
                  if scores[criterion["name"]] < criterion["min_score"]]
        if failed:
            decision = "improve"
            if not feedback:   # the scores are too low, but the LLM wrote no feedback
                feedback = "Improve these criteria: " + ", ".join(failed) + "."
        else:
            decision = "accept"
            feedback = ""      # an accepted answer needs no changes
        self.memory.save_evaluation(inputs["session_id"], inputs["run_id"], inputs["question"], inputs["attempt"],
                                    inputs["answer"], scores, decision, feedback)
        logger.info("Evaluator attempt %d: %s %s (run %s)", inputs["attempt"], decision, failed, inputs["run_id"])
        return {"scores": scores, "decision": decision, "feedback": feedback, "failed": failed}

    # ---------- The chain ----------
    def _build_chain(self):
        # The judge: prompt -> LLM -> the JSON text as a dictionary -> checked.
        # If the JSON is broken or a score is missing, the judge is asked one more time.
        judge = (
            self.PROMPT | self.llm | JsonOutputParser() | RunnableLambda(self._check_result)
        ).with_retry(stop_after_attempt=2)

        return (
            RunnablePassthrough.assign(
                criteria=lambda inputs: self.criteria_text,
                earlier=lambda inputs: self._format_earlier(self.memory.get_evaluations(inputs["run_id"])),
            )
            | RunnablePassthrough.assign(result=judge)
            | RunnableLambda(self._decide_and_save)
        ).with_config(run_name="Evaluator")   # the name in LangSmith


if __name__ == "__main__":
    import uuid

    from src.generator import Generator
    from src.knowledge import KnowledgeBase
    from src.llm import GeminiModels
    from src.loaders import SourceLoader
    from src.memory import EvaluatorMemory, GeneratorMemory
    from src.speech import SpeechToText

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for name in ("httpx", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Make the objects: each agent gets its own memory, and they share the knowledge and the models
    models = GeminiModels()
    knowledge = KnowledgeBase(SourceLoader(SpeechToText()))
    generator = Generator(knowledge, GeneratorMemory(), models.main, models.fast)
    evaluator = Evaluator(EvaluatorMemory(), models.main)

    session = "test-evaluator"
    generator.memory.clear(session)
    evaluator.memory.clear(session)

    def judge(title, run, question, attempt, answer, context):
        result = evaluator.evaluate(session, run, question, attempt, answer, context)
        print(f"\n===== {title}\nQ: {question}\nA: {answer}")
        print(f"Scores: {result['scores']}\nDecision: {result['decision']}\nFeedback: {result['feedback']}")

    # 1) A real answer of the Generator: it should be accepted
    question = "How much does an e-bike cost per hour?"
    run = str(uuid.uuid4())
    written = generator.generate(session, run, question)
    judge("1) The answer of the Generator", run, question, 0, written["answer"], written["context"])

    # 2) A wrong answer (written by hand) with the same sources: it should be sent back with feedback
    run = str(uuid.uuid4())
    judge("2) A wrong price", run, question, 0, "An e-bike costs 10 JOD per hour [1].", written["context"])

    # 3) The fixed answer of the same run: the Evaluator reads its earlier feedback from its memory
    judge("3) The fixed answer", run, question, 1, "An e-bike costs 6 JOD per hour [1].", written["context"])
    print("Evaluations of this run in the memory:",
          [(row["attempt"], row["decision"]) for row in evaluator.memory.get_evaluations(run)])

    # 4) A question that the sources cannot answer: "not available" is the correct answer
    question = "Who is the CEO of Orbit Bikes?"
    run = str(uuid.uuid4())
    written = generator.generate(session, run, question)
    judge("4) An unknown question", run, question, 0, written["answer"], written["context"])
