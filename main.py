from crewai import Crew, CrewOutput
from task_VaaS import VaaSTasks
from vaas_agents import VaaSAgents
from shemas.shemas import QueryAnalysisOutput, UserFeedbackOutput, FinalQueryOutput
import json


class VaaSCrew:
    def __init__(self, query: str):
        self.query = query

    def extract(self) -> dict:
        agents = VaaSAgents()
        tasks = VaaSTasks()
        coord = agents.create_agent(agent_name="coordinator")

        query_task = tasks.create_task(
            task_name="query_refinement_task",
            agent=coord,
            output=QueryAnalysisOutput
        )

        crew = Crew(
            agents=[coord],
            tasks=[query_task],
            verbose=False
        )
        result: CrewOutput = crew.kickoff(inputs={"query": self.query})
        return result.tasks_output[0].json_dict

    def feedback(self, partial: dict) -> dict:
        agents = VaaSAgents()
        tasks = VaaSTasks()
        coord = agents.create_agent(agent_name="coordinator")

        fb_task = tasks.create_task(
            task_name="user_feedback_task",
            agent=coord,
            output=UserFeedbackOutput
        )

        crew = Crew(
            agents=[coord],
            tasks=[fb_task],
            verbose=False
        )
        result: CrewOutput = crew.kickoff(inputs={"analysis_output": partial})
        return result.json_dict

    def finalize(self, extracted: dict, feedback: dict) -> dict:
        agents = VaaSAgents()
        tasks = VaaSTasks()
        coord = agents.create_agent(agent_name="coordinator")

        final_task = tasks.create_task(
            task_name="finalize_query_task",
            agent=coord,
            output=FinalQueryOutput
        )

        crew = Crew(
            agents=[coord],
            tasks=[final_task],
            verbose=False
        )
        result: CrewOutput = crew.kickoff(inputs={
            "analysis_output": extracted,
            "feedback_output": feedback
        })
        return result.json_dict

if __name__ == "__main__":
    raw = input("✈️  Dites-moi votre requête de voyage : ")
    shell = VaaSCrew(raw)

    # Étape 1 : Extraction automatique de la requête
    extracted = shell.extract()

    # Étape 2 : Détection des infos manquantes ou irréalistes
    feedback_output = shell.feedback(extracted)

    # Étape 3 : Interaction terminale pour compléter les infos
    responses = {}
    for field, question in zip(feedback_output["missing_or_unrealistic_fields"], feedback_output["messages"]):
        answer = input(f"❓ {question} ")
        responses[field] = answer
    feedback_output["responses"] = responses

    print("\n📋 Résultat complet de user_feedback_task :")
    print(json.dumps(feedback_output, indent=2, ensure_ascii=False))

    # Étape 4 : Fusion finale des infos dans un JSON complet
    final_result = shell.finalize(extracted, feedback_output)

    print("\n✅ Requête structurée complète :")
    print(json.dumps(final_result, indent=2, ensure_ascii=False))
