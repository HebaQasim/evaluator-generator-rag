"""Part 4: connect to Google Gemini through LangChain's ChatOpenAI.
The same setup as the previous projects: several models linked with fallbacks."""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


class GeminiModels:
    """Makes the two chat models of the project. Each one is a list of Gemini models linked with fallbacks:
    if a model fails or is busy, the next one in the list answers."""
    URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    MAIN_MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]   # the strongest first
    FAST_MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.5-flash"]   # a light one first

    def __init__(self):
        load_dotenv()   # reads GEMINI_API_KEY from the .env file
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY was not found. Put it in the .env file of the project.")
        # For writing and judging the answers
        self.main = self.create_llm(self.MAIN_MODELS)
        # For small jobs (rewriting a follow-up question), so the quota of the strong model is kept for the answers
        self.fast = self.create_llm(self.FAST_MODELS)

    def create_model(self, name):
        return ChatOpenAI(
            model=name,
            api_key=self.api_key,
            base_url=self.URL,
            reasoning_effort="low",
            timeout=30,
            max_retries=0,
        )

    def create_llm(self, model_names):
        """The first model, with the others as its fallbacks."""
        models = [self.create_model(name) for name in model_names]
        return models[0].with_fallbacks(models[1:])


if __name__ == "__main__":
    models = GeminiModels()
    for name, llm in [("main", models.main), ("fast", models.fast)]:
        response = llm.invoke("In one sentence, what is retrieval-augmented generation?")
        print(f"{name} -> model: {response.response_metadata.get('model_name')}")
        print("  ", response.content)
