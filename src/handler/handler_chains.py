import os 
from pydantic import BaseModel,Field
from langchain_core.prompts import ChatPromptTemplate , PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import json

def extract_response(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return json.loads(cleaned)

class iskillQuery(BaseModel):
    result: bool = Field(..., description="The user query if query is  asking for skills or not")

def is_skill_query_prompt(parser=None):
    template = """Given a user query, determine if the user is asking about the skills available.
                query: {query_for_skill}
                Return True or False only. in JSON format only with a key 'result'. For example: "result": true or "result": false"""
    
    return PromptTemplate(input_variables=["query_for_skill"], template=template,)# partial_variables={"format_instructions": parser.get_format_instructions()})

def is_skill_query_chain(llm,user_query:str):
    structred_response = StrOutputParser(pydantic_object=iskillQuery)
    chain =  is_skill_query_prompt() | llm |StrOutputParser()
    response = chain.invoke(
        {
            'query_for_skill': user_query
        }
    )
    return extract_response(response)