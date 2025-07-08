from crewai import Crew, CrewOutput
from task_VaaS import VaaSTasks
from vaas_agents import VaaSAgents


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
            task = tasks.create_task("run_optimizer_task", agent=composer, output=None)
            crew = Crew(agents=[composer], tasks=[task], verbose=True)
            result: CrewOutput = crew.kickoff(inputs={
                "generated_code": generated_code,
                "vaas": self.vaas
            })
            return str(result.tasks_output[0])

    def final_recommendation(self, optimization_result: str) -> str:
        agents = VaaSAgents()
        tasks = VaaSTasks()
        composer = agents.create_agent(agent_name="composer")
        task = tasks.create_task("final_recommendation_task", agent=composer, output=None)
        crew = Crew(agents=[composer], tasks=[task], verbose=True)
        result: CrewOutput = crew.kickoff(inputs={
            "final_query": self.final_query,
            "optimization_result": optimization_result,
            "vaas": self.vaas
        })
        return str(result.tasks_output[0])

if __name__ == "__main__":
    final_query = {
        "origin": "Lyon",
        "destination": "Strasbourg",
        "departure_time": "2025-10-15T10:30:00",
        "constraints": [
            {"type": "budget", "value": 120},
            {"type": "climatisation", "value": True},
            {"type": "places_min", "value": 3},
            {"type": "support_velo", "value": True},
            {"type": "type_energy", "value": "électrique"}
        ]
    }

    vaas = [
        # Région 69 (Lyon)
        {'uid': 'LY1', 'cost': 40, 'speed': 60, 'emission': 60, 'coverd_regions': [69], 'climatisation': True, 'places_min': 3, 'support_velo': True, 'type_energy': 'électrique'},
        {'uid': 'LY2', 'cost': 50, 'speed': 80, 'emission': 45, 'coverd_regions': [69], 'climatisation': True, 'places_min': 5, 'support_velo': True, 'type_energy': 'électrique'},
        {'uid': 'LY3', 'cost': 45, 'speed': 75, 'emission': 50, 'coverd_regions': [69], 'climatisation': True, 'places_min': 4, 'support_velo': True, 'type_energy': 'électrique'},
        
        # Région 25 (Besançon)
        {'uid': 'BS1', 'cost': 38, 'speed': 70, 'emission': 40, 'coverd_regions': [25], 'climatisation': True, 'places_min': 3, 'support_velo': True, 'type_energy': 'électrique'},
        {'uid': 'BS2', 'cost': 42, 'speed': 85, 'emission': 38, 'coverd_regions': [25], 'climatisation': True, 'places_min': 5, 'support_velo': True, 'type_energy': 'électrique'},
        {'uid': 'BS3', 'cost': 39, 'speed': 65, 'emission': 43, 'coverd_regions': [25], 'climatisation': True, 'places_min': 3, 'support_velo': True, 'type_energy': 'électrique'},

        # Région 67 (Strasbourg)
        {'uid': 'ST1', 'cost': 48, 'speed': 85, 'emission': 35, 'coverd_regions': [67], 'climatisation': True, 'places_min': 3, 'support_velo': True, 'type_energy': 'électrique'},
        {'uid': 'ST2', 'cost': 52, 'speed': 90, 'emission': 37, 'coverd_regions': [67], 'climatisation': True, 'places_min': 4, 'support_velo': True, 'type_energy': 'électrique'},
        {'uid': 'ST3', 'cost': 50, 'speed': 78, 'emission': 39, 'coverd_regions': [67], 'climatisation': True, 'places_min': 3, 'support_velo': True, 'type_energy': 'électrique'}
    ]

    regions = [69, 25, 67]


    crew = ComposerCrew(final_query, vaas, regions)

    print("\n=== 🧠 Étape 1 : Génération de la fonction objectif ===\n")
    description = crew.objective()
    print(description)

    print("\n=== 🧩 Étape 2 : Génération automatique du code Python ===\n")
    code = crew.generate_code(description)
    print(code)

    print("\n=== ⚙️ Étape 3 : Exécution du code et récupération de la meilleure solution ===\n")
    result = crew.execute_optimization(code)
    print(result)

    print("\n=== 💡 Étape 4 : Génération de la recommandation utilisateur finale ===\n")
    recommendation = crew.final_recommendation(result)
    print(recommendation)
