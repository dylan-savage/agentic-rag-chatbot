from crewai import Agent
from tools.vector_search import vector_search_tool

def get_retriever_agent(llm):
    return Agent(
        role='Retriever',
        goal='Search for and retrieve the most relevant document chunks based on the refined user query',
        backstory="""
        You are responsible for searching the knowledge base and returning any relevant information.

        If you find relevant context, return it clearly.

        If you don’t find anything useful, just say: 
        ‘I couldn’t find any relevant documents in the knowledge base.’

        Be honest and never make up sources.
        """,
        tools=[vector_search_tool],
        verbose=True,
        llm=llm
    )
