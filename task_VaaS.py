from typing import Dict, Any, Optional
import yaml
from textwrap import dedent
from crewai import Task, Agent
from pydantic import BaseModel
from typing import Dict, Optional, List, Set, Tuple, Union
import json
from utils.models import *

class VaaSTasks:
    def __init__(self, config_path: str = 'config/tasks.yaml'):
        """
        Initialize with path to multi-task YAML config.
        Config format:
        tasks:
          extraction:
            description: "..."
            expected_output: "..."
            output_json: true
          validation:
            description: "..."
            ...
        """
        self.tasks_config = load_config(config_path)
    
    def create_task(self, task_name: str, agent: Agent, output = None, agent_name=None) -> Task:
        """Factory method to create any configured task"""
        if task_name not in self.tasks_config:
            raise ValueError(f"Task '{task_name}' not found in config")
        
        config = self.tasks_config[task_name]

        return Task(
            config=config,
            agent=agent,
            output_json= output
            #guardrail=self.validate_json_output if agent_name == 'coordinator' else None            
        )

   
    def _format_description(self, description: str, context: Optional[Dict]) -> str:
        """Inject context variables into task description"""
        if not context:
            return dedent(description)
        
        try:
            return dedent(description.format(**context))
        except KeyError as e:
            raise ValueError(f"Missing context variable {str(e)} in task description")
    
    def validate_json_output(result: str) -> Tuple[bool, Union[dict, str]]:
        """Validate that the output is valid JSON."""
        missing = []
        try:
            data = json.loads(result)
            if data["origine"] is None:
                missing.appende('origine')
            if data['destination'] is None:
                missing.appende('destination')
            if data["departure_time"] is None:
                missing.appende('departure_time')
            if not data["constraints"]:
                missing.appende('constraints')
            
            return (True, f'These information are missed: {missing}')
        except json.JSONDecodeError:
            return (False, "Output must be valid JSON")