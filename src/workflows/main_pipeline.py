import json
from crewai import Crew
from agents.manager import get_manager_agent
from agents.retriever import get_retriever_agent
from agents.responder import get_responder_agent
from agents.refiner import get_refiner_agent
from agents.clarifier import get_clarifier_agent
from utils.llm_config import get_together_llm
from tasks.main_tasks import get_main_tasks

def run(query):
    llm = get_together_llm()

    # Initialize agents
    manager = get_manager_agent(llm)
    retriever = get_retriever_agent(llm)
    responder = get_responder_agent(llm)
    refiner = get_refiner_agent(llm)
    clarifier = get_clarifier_agent(llm)

    query = query

#   Loop until the user's query is clear
    while True:
        # Build and run the dynamic crew
        tasks = get_main_tasks(manager, query)
        crew = Crew(
            agents=[manager, retriever, responder, refiner, clarifier],
            tasks=tasks,
            verbose=True
        )

        result = crew.kickoff()
        result_text = str(result)

        try:
            # parse structured JSON in case it's Clarifier output
            response_json = json.loads(result_text)

            if response_json.get("needs_clarification", False):
                print("\nClarifier Agent needs more info:")
                print(response_json["question"])

                # Prompt the user to clarify
                query = input("\nPlease clarify your question: ")
                continue  # Loop again with updated query
            else:
                print("\nFinal Answer:\n", result_text)
                break  # Exit loop if clarification is not needed

        except json.JSONDecodeError:
            # Not Clarifier output, show the result
            print("\nFinal Answer:\n", result_text)
            break

if __name__ == "__main__":
    query = input("Enter your question: ")
    run(query)
