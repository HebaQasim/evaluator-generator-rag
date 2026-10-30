"""Part 3: the two memories. The Generator and the Evaluator each have their own memory, in their own SQLite file.

    GeneratorMemory -> generator_memory.db: the conversation, the retrieved context, the answers and the improvement attempts
    EvaluatorMemory -> evaluator_memory.db: the criteria and rules, the evaluations, the feedback and the decisions

The memories are saved on the disk, so they are still there after the app is closed.
Nothing is shared between them: different files, different tables and different objects.
Every file also remembers which memory it belongs to, so the other memory cannot open it by mistake.

Two ids are used:
    session_id: one conversation. Follow-up questions use the history of the same conversation.
    run_id:     one question and its loop (the first answer and its improvements). The workflow of part 6 makes it.
"""
import json
import logging
import sqlite3

logger = logging.getLogger(__name__)


class Memory:
    """What the two memories have in common: one SQLite file that belongs to one owner."""
    OWNER = ""    # "generator" or "evaluator"
    TABLES = []   # the SQL that creates the tables of this memory

    def __init__(self, path=None):
        self.path = path or f"{self.OWNER}_memory.db"
        # First check the owner, so nothing is written into a file of the other memory
        self.run("CREATE TABLE IF NOT EXISTS owner (name TEXT)")
        rows = self.run("SELECT name FROM owner")
        if not rows:   # a new file: write down who owns it
            self.run("INSERT INTO owner VALUES (?)", (self.OWNER,))
        elif rows[0]["name"] != self.OWNER:
            raise PermissionError(f"{self.path} belongs to the {rows[0]['name']} memory, "
                                  f"so the {self.OWNER} memory cannot open it.")
        for sql in self.TABLES:
            self.run(sql)
        logger.info("Opened the %s memory (%s)", self.OWNER, self.path)

    def run(self, sql, values=()):
        """Run one SQL command and return the rows as dictionaries. Every call opens the file and closes it again."""
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row   # rows can be read by column name
        try:
            rows = [dict(row) for row in db.execute(sql, values)]
            db.commit()
            return rows
        finally:
            db.close()


class GeneratorMemory(Memory):
    OWNER = "generator"
    TABLES = [
        # The final answer of every question: what the user received
        """CREATE TABLE IF NOT EXISTS conversation (
               id INTEGER PRIMARY KEY, session_id TEXT, run_id TEXT, question TEXT, answer TEXT,
               created_at TEXT DEFAULT CURRENT_TIMESTAMP)""",
        # Every answer the Generator wrote: attempt 0 is the first answer, 1 to 4 are the improvements
        """CREATE TABLE IF NOT EXISTS attempts (
               id INTEGER PRIMARY KEY, session_id TEXT, run_id TEXT, question TEXT, attempt INTEGER,
               context TEXT, feedback TEXT, answer TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""",
    ]

    def save_attempt(self, session_id, run_id, question, attempt, context, answer, feedback=None):
        """One answer. context is the retrieved text it was written from.
        feedback is the Evaluator feedback that this attempt tried to fix (None for the first answer)."""
        self.run("INSERT INTO attempts (session_id, run_id, question, attempt, context, feedback, answer) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?)", (session_id, run_id, question, attempt, context, feedback, answer))

    def get_attempts(self, run_id):
        """The earlier attempts for this question, in order. The Generator reads them before it improves its answer."""
        return self.run("SELECT attempt, context, feedback, answer FROM attempts WHERE run_id = ? ORDER BY attempt",
                        (run_id,))

    def save_turn(self, session_id, run_id, question, answer):
        """The final answer of a question, after the loop has ended."""
        self.run("INSERT INTO conversation (session_id, run_id, question, answer) VALUES (?, ?, ?, ?)",
                 (session_id, run_id, question, answer))

    def get_history(self, session_id, limit=3):
        """The last questions and final answers of this conversation, oldest first (for follow-up questions)."""
        rows = self.run("SELECT question, answer FROM conversation WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                        (session_id, limit))
        return rows[::-1]

    def clear(self, session_id):
        for table in ("conversation", "attempts"):
            self.run(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))


# The criteria of the task, and the rule of each one: the lowest score (from 1 to 5) that is still acceptable
EVALUATION_CRITERIA = [
    ("accuracy", "The facts in the answer are correct according to the sources.", 4),
    ("relevance", "The answer is about what the user asked.", 4),
    ("completeness", "The answer covers every part of the question that the sources can answer.", 4),
    ("consistency", "The answer agrees with the external knowledge and does not contradict itself.", 4),
    ("grounding", "Every statement in the answer comes from the retrieved context.", 4),
    ("no_unsupported_claims", "The answer adds nothing that the sources do not say.", 4),
    ("overall_quality", "The answer is clear, well organised and helpful.", 4),
]


class EvaluatorMemory(Memory):
    OWNER = "evaluator"
    TABLES = [
        """CREATE TABLE IF NOT EXISTS criteria (name TEXT PRIMARY KEY, description TEXT, min_score INTEGER)""",
        """CREATE TABLE IF NOT EXISTS evaluations (
               id INTEGER PRIMARY KEY, session_id TEXT, run_id TEXT, question TEXT, attempt INTEGER, answer TEXT,
               scores TEXT, decision TEXT, feedback TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""",
    ]

    def __init__(self, path=None):
        super().__init__(path)
        # The list above is written into the memory every time, so a change in the code is always used
        self.run("DELETE FROM criteria")
        for name, description, min_score in EVALUATION_CRITERIA:
            self.run("INSERT INTO criteria VALUES (?, ?, ?)", (name, description, min_score))

    def get_criteria(self):
        """The criteria and rules that the Evaluator judges by."""
        return self.run("SELECT name, description, min_score FROM criteria")

    def save_evaluation(self, session_id, run_id, question, attempt, answer, scores, decision, feedback):
        """scores: {"accuracy": 5, "relevance": 4, ...}, decision: "accept" or "improve"."""
        self.run("INSERT INTO evaluations (session_id, run_id, question, attempt, answer, scores, decision, feedback) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                 (session_id, run_id, question, attempt, answer, json.dumps(scores), decision, feedback))

    def get_evaluations(self, run_id):
        """The earlier evaluations of this question. The Evaluator reads them to check that its feedback was fixed."""
        rows = self.run("SELECT attempt, scores, decision, feedback FROM evaluations WHERE run_id = ? ORDER BY attempt",
                        (run_id,))
        for row in rows:
            row["scores"] = json.loads(row["scores"])   # the text in the database -> a dictionary again
        return rows

    def clear(self, session_id):
        self.run("DELETE FROM evaluations WHERE session_id = ?", (session_id,))


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # The test uses its own files in a temporary folder, so the real memories are not touched
    with tempfile.TemporaryDirectory() as folder:
        generator_file = str(Path(folder) / "generator_memory.db")
        evaluator_file = str(Path(folder) / "evaluator_memory.db")
        generator_memory = GeneratorMemory(generator_file)
        evaluator_memory = EvaluatorMemory(evaluator_file)

        # 1) One question with one improvement, as the loop of part 6 will do it.
        #    The answers and scores are written by hand (no LLM yet), and only 3 scores are given to keep it short.
        session, run = "session-1", "run-1"
        question = "How much does an e-bike cost per hour?"
        context = "[orbit_handbook.pdf, page 1] ... an electric bike (e-bike) costs 6 JOD per hour ..."

        generator_memory.save_attempt(session, run, question, 0, context, "An e-bike costs 6 JOD.")
        evaluator_memory.save_evaluation(session, run, question, 0, "An e-bike costs 6 JOD.",
                                         {"accuracy": 5, "relevance": 5, "completeness": 3},
                                         "improve", "Say that the price is per hour.")
        feedback = evaluator_memory.get_evaluations(run)[-1]["feedback"]   # the workflow passes only this text on
        generator_memory.save_attempt(session, run, question, 1, context, "An e-bike costs 6 JOD per hour.", feedback)
        evaluator_memory.save_evaluation(session, run, question, 1, "An e-bike costs 6 JOD per hour.",
                                         {"accuracy": 5, "relevance": 5, "completeness": 5}, "accept", "")
        generator_memory.save_turn(session, run, question, "An e-bike costs 6 JOD per hour.")

        print("\nGenerator - attempts of run-1:")
        for row in generator_memory.get_attempts(run):
            print(f"   {row['attempt']}: {row['answer']} | feedback: {row['feedback']}")
        print("Generator - history of session-1:", generator_memory.get_history(session))
        print("Evaluator - evaluations of run-1:")
        for row in evaluator_memory.get_evaluations(run):
            print(f"   {row['attempt']}: {row['decision']} {row['scores']} | feedback: {row['feedback']}")
        print("Evaluator - criteria:", [row["name"] for row in evaluator_memory.get_criteria()])

        # 2) The memories stay after the app is closed: new objects on the same files still find everything
        print("\nAfter opening the files again:", GeneratorMemory(generator_file).get_history(session))

        # 3) Another conversation does not see this conversation
        print("History of session-2:", generator_memory.get_history("session-2"))

        # 4) Isolation: every file has only its own tables, and the other memory cannot open it
        print()
        for memory_class, path in [(GeneratorMemory, evaluator_file), (EvaluatorMemory, generator_file)]:
            try:
                memory_class(path)
            except PermissionError as error:
                print("Error (expected):", error)
        list_tables = "SELECT name FROM sqlite_master WHERE type = 'table'"
        print("Tables in generator_memory.db:", [row["name"] for row in generator_memory.run(list_tables)])
        print("Tables in evaluator_memory.db:", [row["name"] for row in evaluator_memory.run(list_tables)])

        # 5) Clear one conversation
        generator_memory.clear(session)
        evaluator_memory.clear(session)
        print("\nAfter clear:", generator_memory.get_history(session), evaluator_memory.get_evaluations(run))
