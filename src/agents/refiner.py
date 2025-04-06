from crewai import Agent

def get_refiner_agent(llm):
    return Agent(
        role='Refiner',
        goal='Rephrase vague or underspecified queries into precise and actionable prompts',
        backstory='You enhance fuzzy user input by rewriting it in a way that downstream agents can understand and act on more effectively.',
        verbose=True,
        llm=llm
    )

