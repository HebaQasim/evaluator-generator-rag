"""Part 6: the Evaluator-Generator loop, built with LCEL (and, from part 7, the cache in front of it).

    question -> follow-up? rewrite it -> in the cache (same meaning)? -> yes: the saved answer
                                                                      -> no:  the loop, then save the answer
    one round: Generator writes (or improves) the answer -> Evaluator judges it
    then:      accepted                   -> finish: the answer goes to the user
               4 improvements were made   -> finish: the last answer, with a note that it could not be fully validated
               otherwise                  -> the feedback goes back to the Generator, and the round runs again

The loop counter ("attempt") is part of the workflow state: 0 is the first answer, 1 to 4 are the improvements.
"""
import logging
import uuid

from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

logger = logging.getLogger(__name__)


class EvaluatorGeneratorWorkflow:
    """Connects the Generator and the Evaluator. They never meet: the workflow passes the answer and the sources
    to the Evaluator, and only the feedback text back to the Generator."""
    MAX_IMPROVEMENTS = 4
    MAX_QUESTION_LENGTH = 1000
    NOT_VALIDATED_NOTE = ("Note: this answer could not be fully validated. "
                          "It is the latest answer after the maximum of 4 improvement rounds.")

    def __init__(self, generator, evaluator, cache=None):
        self.generator = generator
        self.evaluator = evaluator
        self.cache = cache   # part 7 (None: no cache)
        self.chain = self._build_chain()

    # ---------- What the rest of the project uses ----------
    def ask(self, question, session_id="default"):
        """A question of the user -> the final answer, with how it was reached."""
        question = self._check_question(question)
        run_id = str(uuid.uuid4())
        logger.info("New question (run %s): %s", run_id, question)
        state = {"session_id": session_id, "run_id": run_id, "question": question,
                 "attempt": 0, "feedback": None, "rounds": []}
        # The metadata and tags are shown with the trace in LangSmith (part 8), so one question can be found easily
        config = {"metadata": {"session_id": session_id, "run_id": run_id}, "tags": ["evaluator-generator"]}
        try:
            return self.chain.invoke(state, config=config)
        except Exception as error:
            logger.exception("The workflow failed (run %s)", run_id)
            raise RuntimeError("The answer could not be created right now. "
                               "The language model may be busy, please try again in a minute.") from error

    # ---------- Helpers ----------
    def _check_question(self, question):
        question = (question or "").strip()
        if not question:
            raise ValueError("Please write a question.")
        if len(question) > self.MAX_QUESTION_LENGTH:
            raise ValueError(f"The question is too long (more than {self.MAX_QUESTION_LENGTH} characters).")
        return question

    def _look_in_cache(self, state):
        return self.cache.lookup(state["search_question"]) if self.cache else None

    def _answer_from_cache(self, state):
        """The same question (in meaning) was answered before: no Generator, no Evaluator, no model calls."""
        cached = state["cached"]
        self.generator.save_final_answer(state["session_id"], state["run_id"], state["question"], cached["answer"])
        logger.info("Answered from the cache (run %s)", state["run_id"])
        return {**cached, "run_id": state["run_id"], "question": state["question"],
                "search_question": state["search_question"], "from_cache": True}

    def _save_in_cache(self, result):
        """Only a validated answer is cached, so an answer that did not pass the Evaluator is never repeated."""
        if self.cache and result["validated"]:
            self.cache.save(result["search_question"],
                            {key: result[key] for key in ("answer", "validated", "improvements", "context", "rounds")})
        return {**result, "from_cache": False}

    def _record_round(self, state):
        """Keep a short record of every round, so the user (and the logs) can see how the answer changed."""
        evaluation = state["evaluation"]
        logger.info("Round %d (run %s): %s %s", state["attempt"], state["run_id"],
                    evaluation["decision"], evaluation["failed"])
        round_info = {"attempt": state["attempt"], "answer": state["answer"], "decision": evaluation["decision"],
                      "scores": evaluation["scores"], "feedback": evaluation["feedback"]}
        return {**state, "rounds": state["rounds"] + [round_info]}

    def _prepare_next_round(self, state):
        """The loop counter goes up, and the feedback of the Evaluator is given to the Generator."""
        return {**state, "attempt": state["attempt"] + 1, "feedback": state["evaluation"]["feedback"]}

    def _finish(self, state):
        """End of the loop: accepted, or the maximum was reached. The answer joins the conversation history."""
        validated = state["evaluation"]["decision"] == "accept"
        self.generator.save_final_answer(state["session_id"], state["run_id"], state["question"], state["answer"])
        answer = state["answer"] if validated else state["answer"] + "\n\n" + self.NOT_VALIDATED_NOTE
        logger.info("Finished (run %s): validated=%s after %d improvement(s)",
                    state["run_id"], validated, state["attempt"])
        return {"run_id": state["run_id"], "question": state["question"],
                "search_question": state["search_question"], "answer": answer, "validated": validated,
                "improvements": state["attempt"], "context": state["context"], "rounds": state["rounds"]}

    # ---------- The chains ----------
    def _build_chain(self):
        # One round: the Generator writes, the Evaluator judges
        one_round = (
            RunnablePassthrough.assign(written=self.generator.chain)
            | RunnablePassthrough.assign(answer=lambda state: state["written"]["answer"],
                                         context=lambda state: state["written"]["context"])
            | RunnablePassthrough.assign(evaluation=self.evaluator.chain)
            | RunnableLambda(self._record_round)
        ).with_config(run_name="Round")

        # After every round, one of three ways. The third way runs the loop again:
        # a RunnableLambda that returns a Runnable makes LangChain run that Runnable with the new state.
        self.loop = (one_round | RunnableBranch(
            (lambda state: state["evaluation"]["decision"] == "accept", RunnableLambda(self._finish)),
            (lambda state: state["attempt"] >= self.MAX_IMPROVEMENTS, RunnableLambda(self._finish)),
            RunnableLambda(self._prepare_next_round) | RunnableLambda(lambda state: self.loop),
        )).with_config(run_name="Feedback loop")

        # The whole workflow: the question that can be understood alone, then the cache, then (if needed) the loop
        return (
            RunnablePassthrough.assign(search_question=lambda state: self.generator.standalone_question(
                state["session_id"], state["question"]))
            | RunnablePassthrough.assign(cached=RunnableLambda(self._look_in_cache))
            | RunnableBranch(
                (lambda state: state["cached"] is not None, RunnableLambda(self._answer_from_cache)),
                self.loop | RunnableLambda(self._save_in_cache),
            )
        ).with_config(run_name="Evaluator-Generator workflow")


if __name__ == "__main__":
    from src.cache import SemanticCache
    from src.evaluator import Evaluator
    from src.generator import Generator
    from src.knowledge import KnowledgeBase
    from src.llm import GeminiModels
    from src.loaders import SourceLoader
    from src.memory import EvaluatorMemory, GeneratorMemory
    from src.monitoring import Monitoring
    from src.speech import SpeechToText

    Monitoring()   # part 8: logs on the screen and in logs/app.log, and LangSmith tracing if it is set in .env

    def show(result):
        print(f"\nQ: {result['question']}   (from the cache: {result['from_cache']})")
        for round_info in result["rounds"]:
            print(f"   round {round_info['attempt']}: {round_info['decision']} | {round_info['answer'][:90]}")
            if round_info["feedback"]:
                print(f"      feedback: {round_info['feedback'][:150]}")
        print(f"A: {result['answer']}\n   validated: {result['validated']}, improvements: {result['improvements']}")

    # 1) The real system: every object gets what it needs, and each agent gets its own memory
    models = GeminiModels()
    knowledge = KnowledgeBase(SourceLoader(SpeechToText()))
    generator = Generator(knowledge, GeneratorMemory(), models.main, models.fast)
    evaluator = Evaluator(EvaluatorMemory(), models.main)
    cache = SemanticCache(knowledge, models.fast)
    workflow = EvaluatorGeneratorWorkflow(generator, evaluator, cache)

    session = "test-workflow"
    generator.memory.clear(session)
    evaluator.memory.clear(session)
    cache.clear()   # start without cached answers, so the first question runs the loop
    print("Redis available:", cache.is_available())

    for question in ["How much does an e-bike cost per hour?",
                     "How much does an e-bike cost per hour?",                      # the same: from the cache
                     "What is the hourly price of an electric bike?",               # the same meaning: from the cache
                     "And how much is the deposit for it?",                         # a follow-up question
                     "What do I get with the Plus plan, and how much does it cost?",
                     "Who is the CEO of Orbit Bikes?"]:                             # not in the sources
        show(workflow.ask(question, session))

    # 2) Bad questions are stopped before any model is called
    for question in ["", "   "]:
        try:
            workflow.ask(question, session)
        except ValueError as error:
            print(f"\nError (expected) for {question!r}: {error}")

    # 3) The maximum of 4 improvements. A real Evaluator is rarely this strict, so two small fake objects
    #    are given to the workflow instead (no model is called, no cache): an Evaluator that never accepts.
    class FakeGenerator:
        chain = RunnableLambda(lambda state: {"answer": f"draft number {state['attempt']}", "context": "[1] a source"})

        def standalone_question(self, session_id, question):
            return question

        def save_final_answer(self, session_id, run_id, question, answer):
            pass

    class NeverHappyEvaluator:
        chain = RunnableLambda(lambda state: {"scores": {"accuracy": 2}, "decision": "improve",
                                              "feedback": f"draft number {state['attempt']} is not good enough",
                                              "failed": ["accuracy (2/5)"]})

    show(EvaluatorGeneratorWorkflow(FakeGenerator(), NeverHappyEvaluator()).ask("A question that never passes"))
