import os
from deepagents import DeepAgents
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI

def run_deep_agents(input_value: str) -> dict:
    """
    Run the Deep Agents skill with the given input.

    Args:
    input_value (str): The user query or task description.

    Returns:
    dict: A dictionary containing the agent response or task result.
    """

    try:
        # Initialize the Deep Agents
        agents = DeepAgents()

        # Define a prompt template for the agent
        template = PromptTemplate(
            input_variables=["input"],
            template="You are a helpful assistant. You will be given a task. Your response should be a detailed plan to complete the task.\n\nTask: {input}",
        )

        # Initialize the LLM chain
        llm = OpenAI(temperature=0)
        chain = LLMChain(llm=llm, prompt=template)

        # Run the agent with the input
        output = chain.run(input=input_value)

        # Return the agent response
        return {"success": True, "output": output}

    except Exception as e:
        # Handle any exceptions
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Deep Agents skill")
    parser.add_argument("--input", type=str, help="The user query or task description")
    args = parser.parse_args()

    if args.input:
        result = run_deep_agents(args.input)
        print(result)
    else:
        print("Please provide an input value")