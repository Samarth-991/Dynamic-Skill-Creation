import os 
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
import logging


class LLMService:
    def __init__(self, service_name, api_key=None):
        self.service_name = service_name
        self._key = api_key

    def ollama_model(self, model_name = "qwen3:4b"):#'llama3-groq-tool-use:latest'
        llm = ChatOllama(model=model_name,temperature=0)
        return llm
    
    def get_groq_model(self,key,model_name = "gemma2-9b-it"):
        print("Setting GROQ_API_KEY in environment variables...")
        os.environ["GROQ_API_KEY"] = key
        llm_groq = ChatGroq(model=model_name)
        return llm_groq

    def get_googleGemini(self,api_key:str, temperature: float = 0.1) -> str:
    
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=temperature,
        )
        return llm

    def get_llm_model(self,):
        try:
            if self.service_name == "Openai":
                return "Not implemented yet"
            
            elif self.service_name == "Ollama":
                return self.ollama_model()
            
            elif self.service_name == "Google":
                return self.get_googleGemini(self._key)
            
            elif self.service_name == "GROQ":
                return self.get_groq_model(self._key,model_name = "qwen/qwen3-32b")
            else:
                logging.error(f'Invalid model type: {self.service_name}')
                raise ValueError(f'Invalid model type: {self.service_name}')
        except Exception as e:
            logging.error(f'Failed to get LLM model for type: {self.service_name}')
            raise e

