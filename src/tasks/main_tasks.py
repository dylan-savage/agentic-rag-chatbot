from crewai import Task


def get_main_tasks(manager, query):
    return [
        Task(
            description=f"""
            You are the Manager agent in an agentic RAG system. Your job is to decide how to process the user's query dynamically.
            Based on the query: '{query}', decide which agents to involve and what subtasks they should perform.

            For example:
            - If the query is ambiguous, delegate to the Clarifier first and wait for the user's response before proceeding.
            - If the query is specific, assign a Retriever to fetch information from the knowledge base.
            - If the knowledge base lacks relevant info, ask the Responder to generate an LLM-based answer instead.
            - If helpful, assign the Refiner to clean up a vague or broad query before retrieval.
            - At the end, assign the Responder to answer the user clearly and completely.

            You must dynamically assign and sequence these tasks during runtime rather than following a static pipeline.
            """,
            expected_output="A dynamic execution plan: one or more tasks assigned to relevant agents, executed based on context and user input.",
            agent=manager
        )
    ]