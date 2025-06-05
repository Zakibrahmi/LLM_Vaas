from crewai import Agent, Task, Crew
from tools.time_tools import validate_datetime_in_city
from utils.models import get_LLM_model, load_config

agents_config = load_config("config/agents.yaml")

# Inputs depuis terminal
city = input("🏙️ Entrez la ville : ")
date_str = input("📅 Entrez la date (format ISO 8601 ex: 2025-06-02T14:00:00) : ")

# Création de l’agent
agent = Agent(
    config=agents_config["time_checker"],
    llm=get_LLM_model("OpenAI"),
    tools=[validate_datetime_in_city],
    verbose=True
)

# Définir la tâche
task = Task(
    description=f"L'utilisateur veut vérifier si la date {date_str} est passée ou non à {city}.",
    expected_output="Indique si la date est passée ou dans le futur, selon l'heure actuelle locale.",
    agent=agent
)

crew = Crew(agents=[agent], tasks=[task], verbose=True)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n🧭 Résultat final :")
    print(result)
