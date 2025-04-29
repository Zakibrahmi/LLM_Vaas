from crewai import Agent, Crew, Task
from typing import Dict, Any
import yaml
import json
from textwrap import dedent
from task_VaaS import *
from vaas_agents import *
from tools.scope_agent_toosl import *
from shemas.shemas import *

class VaaSCrew:
    def __init__(self, query: str=None):
        self.query = query

    def run(self) -> Dict[str, Any]:
        agents = VaaSAgents()
        tasks = VaaSTasks()
        
        # Agent coordiator 
        coordinator_agent = agents.create_agent(agent_name="coordinator")
        # Task for coordinator agent
        query_task = tasks.create_task(task_name= "query_refinement_task", agent=coordinator_agent, output=QueryAnalysisOutput)

        crew = Crew(
            agents=[coordinator_agent],
            tasks=[query_task]
        )
        result = crew.kickoff(inputs={"query": "I want to travel to Jeddah under 5 Riyals"})

        print("Analysis Results:")
        print(result)
        
        """
        # SmartScope Agent. Responsable on extract relavant regions and VaaS to user query
        extract_samples_tool = ExtractSamplesTool()
        ml_model_tool = MLModelTool()
        SmartScope_Agent= agents.create_agent(agent_name="smartScope", tools=[extract_samples_tool,ml_model_tool])
        analysis_task= tasks.create_task(task_name="analysis_task", 
                                      agent=SmartScope_Agent, 
                                      agent_name="smartScope"                                                                            
                                      )
        ml_prediction_task = tasks.create_task(task_name="ml_prediction_task", 
                                      agent=SmartScope_Agent, 
                                      agent_name="smartScope"                                                                            
                                      )
        ml_prediction_task.async_execution=False
        ml_prediction_task.context = [analysis_task]
       
        
        crew = Crew(
            agents=[SmartScope_Agent],
            tasks=[analysis_task, ml_prediction_task]
        )
        result = crew.kickoff(inputs={
            "target_datetime": "2018-08-03 20:00:00",
            "num_samples": 5,
            "column": "date_time"  # Can be changed to other timestamp columns
        })

        print("Analysis Results:")
        print(result)
        
       # return coordinator_agent.last_step_output
       """

if __name__ == "__main__":
    v = VaaSCrew()
    v.run()

    """
    test_queries = [
        "I want to go to Berlin next month",
        "Find me flights under $500",
        "From Madrid to Rome with pet accommodation"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}\nProcessing query: '{query}'\n{'='*50}")
        crew = VaaSCrew(query)
        result = crew.run()
        
        print("\nExtracted Data:")
        print(json.dumps(result.get('extracted_data', {}), indent=2))
        
        if messages := result.get('validation_messages', []):
            print("\nMissing Information:")
            for msg in messages:
                print(f"- {msg}")
        else:
            print("\nAll required information is complete!")
    """