from crewai import Crew, CrewOutput
from task_VaaS import VaaSTasks
from vaas_agents import VaaSAgents
import json


class ComposerCrew:
    def __init__(self, final_query: dict, vaas: list, regions: list):
        self.final_query = final_query
        self.vaas = vaas
        self.regions = regions

    def objective(self) -> str:
        agents = VaaSAgents()
        tasks = VaaSTasks()
        composer = agents.create_agent(agent_name="composer")
        task = tasks.create_task("objective_function_task", agent=composer, output=None)
        crew = Crew(agents=[composer], tasks=[task], verbose=True)
        result: CrewOutput = crew.kickoff(inputs={
            "final_query": self.final_query,
            "vaas": self.vaas,
            "regions": self.regions
        })
        return str(result.tasks_output[0])

    def generate_code(self, objective_description: str) -> str:
        agents = VaaSAgents()
        tasks = VaaSTasks()
        composer = agents.create_agent(agent_name="composer")
        task = tasks.create_task("generate_code_task", agent=composer, output=None)
        crew = Crew(agents=[composer], tasks=[task], verbose=True)
        result: CrewOutput = crew.kickoff(inputs={
            "objective_description": objective_description,
            "vaas": self.vaas
        })
        return str(result.tasks_output[0])

    def execute_optimization(self, generated_code: str) -> str:
        agents = VaaSAgents()
        tasks = VaaSTasks()
        composer = agents.create_agent(agent_name="composer")
        task = tasks.create_task("run_optimizer_task", agent=composer, output=None)  # NOM corrigé
        crew = Crew(agents=[composer], tasks=[task], verbose=True)
        result: CrewOutput = crew.kickoff(inputs={
            "generated_code": generated_code
        })
        return str(result.tasks_output[0])

    def recommend_solution(self, optimization_result: str) -> str:
        agents = VaaSAgents()
        tasks = VaaSTasks()
        composer = agents.create_agent(agent_name="composer")
        task = tasks.create_task("final_recommendation_task", agent=composer, output=None)  # NOM corrigé
        crew = Crew(agents=[composer], tasks=[task], verbose=True)
        result: CrewOutput = crew.kickoff(inputs={
            "final_query": self.final_query,
            "optimization_result": optimization_result,
            "vaas": self.vaas
        })
        return str(result.tasks_output[0])


if __name__ == "__main__":
    final_query = {
        "origin": "Paris",
        "destination": "Nice",
        "departure_time": "2025-07-01T09:00:00",
        "constraints": [
            {"type": "budget", "value": "120"},
            {"type": "electric_plug", "value": "true"},
            {"type": "meal", "value": "végétarien"}
        ]
    }

    vaas = [
        {"uid": "X001", "cost": 50, "speed": 90, "coverd_regions": [21, 23], "electric_plug": False, "meal": "standard"},
        {"uid": "X002", "cost": 60, "speed": 110, "coverd_regions": [22, 24], "electric_plug": True, "meal": "standard"},
        {"uid": "X003", "cost": 80, "speed": 100, "coverd_regions": [21, 25], "electric_plug": False, "meal": "végétarien"},
        {"uid": "V001", "cost": 60, "speed": 130, "coverd_regions": [21, 24], "electric_plug": True, "meal": "végétarien"},
        {"uid": "V002", "cost": 70, "speed": 100, "coverd_regions": [22, 26], "electric_plug": True, "meal": "végétarien"},
        {"uid": "V003", "cost": 65, "speed": 90, "coverd_regions": [21, 22], "electric_plug": True, "meal": "végétarien"},
        {"uid": "V004", "cost": 55, "speed": 80, "coverd_regions": [22, 23], "electric_plug": True, "meal": "végétarien"},
        {"uid": "Z999", "cost": 40, "speed": 120, "coverd_regions": [30, 31], "electric_plug": True, "meal": "végétarien"}
    ]

    regions = [21, 22]

    crew = ComposerCrew(final_query, vaas, regions)

    print("\n=== 🧠 Étape 1 : Génération de la fonction objectif ===\n")
    description = crew.objective()
    print(description)

    print("\n=== 🧩 Étape 2 : Génération automatique du code Python ===\n")
    code = crew.generate_code(description)
    print(code)

    print("\n=== ⚙️ Étape 3 : Exécution de l’optimisation ===\n")
    result = crew.execute_optimization(code)
    print(result)

    print("\n=== 🎯 Étape 4 : Recommandation finale ===\n")
    recommendation = crew.recommend_solution(result)
    print(recommendation)
