import os 
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
import logging


class LLMService:
    def __init__(self, service_name, api_key=None):
        self.service_name = service_name
        self.key = api_key
    
    def ollama_model(self, model_name = "PetrosStav/gemma3-tools:4b"):#'llama3-groq-tool-use:latest' , PetrosStav/gemma3-tools:4b
        llm = ChatOllama(model=model_name,temperature=0)
        return llm
    
    def get_groq_model(self,key,model_name = "qwen/qwen3-32b"):
        os.environ["GROQ_API_KEY"] = key
        llm_groq = ChatGroq(model=model_name)
        return llm_groq

    def get_googleGemini(self,key):
        os.environ["GOOGLE_API_KEY"] = key
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2)
        return llm

    def get_llm_model(self,):
        try:
            if self.service_name == 'AzureOpenAi':
                logging.error('AzureOpenAi is not implemented yet')
                raise NotImplementedError('AzureOpenAi is not implemented yet')

            if self.service_name == "Openai":
                logging.error('Openai is not implemented yet')
                raise NotImplementedError('Openai is not implemented yet')
            
            elif self.service_name == "Ollama":
                return self.ollama_model()
            
            elif self.service_name == "Google":
                return self.get_googleGemini(self.key)
            
            elif self.service_name == "GROQ":
                return self.get_groq_model(self.key,model_name = "qwen/qwen3-32b")
            else:
                logging.error(f'Invalid model type: {self.service_name}')
                raise ValueError(f'Invalid model type: {self.service_name}')
        except Exception as e:
            logging.error(f'Failed to get LLM model for type: {self.service_name}')
            raise e

