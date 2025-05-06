# === LangChain + Groq Complete Tutorial Code - Tunisia Version 🇹🇳 ===
# Author: [Your Name]
# Requirements:
# pip install langchain langchain-groq langchain_community pydantic duckduckgo-search numexpr

# === Set Up Environment ===
import os
os.environ["GROQ_API_KEY"] = "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"  # Replace this with your actual Groq API key

# === Import the Groq chat model ===
from langchain_groq import ChatGroq

# === Basic Query: Facts about Tunisia ===
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.3,  # Low randomness
    max_tokens=500,
)

# Ask for facts about Tunisia
response = llm.invoke("Give me 3 interesting facts about Tunisia.")
print(f"\nFacts about Tunisia:\n{response.content}")

# === Pirate-style Chat with System + Human Message ===
from langchain.schema import HumanMessage, SystemMessage

# Create a chat model with higher creativity
chat_model = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.7,
    max_tokens=500,
)

# System message defines the assistant’s personality
system_message = SystemMessage(
    content="You are a friendly pirate who loves to share knowledge. Always respond in pirate slang and use nautical emojis ☠️🏴‍☠️"
)

# Human asks about Tunisian history
question = "Tell me about the history of Tunisia."

# Combine messages into a list
messages = [
    system_message,
    HumanMessage(content=question)
]

# Get pirate-style response
response = chat_model.invoke(messages)
print("\nPirate-style History of Tunisia:\n")
print(response.content)

# === Using a Prompt Template to List Dishes ===
from langchain.prompts import PromptTemplate

# Create a prompt that takes a number and outputs dish names
prompt_template = PromptTemplate.from_template(
    "List {n} famous Tunisian dishes (name only)."
)

# Create a chain that feeds the prompt to the model
chain = prompt_template | llm

# Request 5 dishes
response = chain.invoke({
    "n": 5,
})

print("\nFamous Tunisian Dishes:")
print(response.content)

# === Structured Output with Pydantic for Movies ===
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Define a movie schema
class Movie(BaseModel):
    title: str = Field(description="The title of the movie.")
    genre: list[str] = Field(description="The genre of the movie.")
    year: int = Field(description="The year the movie was released.")

# Create a parser to enforce this format
parser = PydanticOutputParser(pydantic_object=Movie)

# Prompt with formatting instructions
prompt_template_text = """
Give a Tunisian movie recommendation in this structure:\n
{format_instructions}\n
{query}
"""

# Format instructions from the parser
format_instructions = parser.get_format_instructions()

# Final prompt template using partials
prompt_template = PromptTemplate(
    template=prompt_template_text,
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)

# Create the full chain (prompt → LLM → parse)
chain = prompt_template | llm | parser

# Ask for a movie
response = chain.invoke({"query": "A famous Tunisian movie from the 2000s."})
print("\nStructured Tunisian Movie Output:")
print(response)

# === Build a Simple Agent with Search + Math ===
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import LLMMathChain
from langchain.agents import AgentExecutor, Tool
from langchain.agents.structured_chat.base import StructuredChatAgent

# Math tool using LLM
math_prompt = PromptTemplate.from_template(
    "Calculate the following expression and return the result in the format 'Answer: <number>': {question}"
)

# Create a math tool
llm_math_chain = LLMMathChain.from_llm(llm=llm, prompt=math_prompt, verbose=True)

# Web search tool (using DuckDuckGo)
search = DuckDuckGoSearchRun()

# Create a calculator tool
calculator = Tool(
    name="calculator",
    description="Use this tool for arithmetic calculations.",
    func=lambda x: llm_math_chain.run({"question": x}),
)

# Define the tools the agent can use
tools = [
    Tool(
        name="search",
        description="Search for facts about Tunisia or Tunisian culture.",
        func=search.run
    ),
    calculator
]

# Create the chat agent with tools
agent = StructuredChatAgent.from_llm_and_tools(
    llm=llm,
    tools=tools
)

# Create executor to run the agent
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True
)

# Ask a question that requires search
result = agent_executor.invoke({"input": "What is the population of Tunisia in 2024?"})
print("\nPopulation Info from AI Agent:")
print(result["output"])
