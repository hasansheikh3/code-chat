import os
import dotenv
from agno.agent import Agent

# Load environment variables from .env file
dotenv.load_dotenv()

class LLM:
    def __init__(self, llm=None):
        self.llm = llm or os.getenv("LLM", "GEMINI")
        self.llm_model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        self.api_key = os.getenv("API_KEY")
        if not self.api_key:
            raise ValueError("API key for LLM is not set in environment variables.")

    def init_model(self):
        """
        Generate a response from the LLM based on the provided prompt.
        This method should be overridden by subclasses to implement specific LLM logic.
        """
        
        match self.llm:
            case "GEMINI":
                from agno.models.google import Gemini
                model = Gemini(
                    id=self.llm_model,
                    api_key=self.api_key,
                    vertexai=False,
                )
                self.model = model
            case "GROQ":
                from agno.models.groq import Groq
                model = Groq(
                    id=self.llm_model,
                    api_key=self.api_key,
                )
                self.model = model
            case _:
                raise ValueError(f"Unsupported LLM: {self.llm}")
            
    def generate_response(self, prompt):
        """
        Generate a response from the LLM based on the provided prompt.
        
        Args:
            prompt (str): The input prompt for the LLM.
        
        Returns:
            str: The generated response from the LLM.
        """
        if not hasattr(self, 'model'):
            self.init_model()
        
        agent = Agent(
            model=self.model,
            markdown=True,
        )
        return agent.run(prompt).content.strip() or "No response generated."