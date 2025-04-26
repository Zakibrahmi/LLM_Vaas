from typing import Dict, Any, Optional
import yaml
import json
from crewai import Agent, LLM
from utils.models import *


class VaaSAgents:
    def __init__(self, config_path: str = 'config/agents.yaml'):
        """
        Initialize with path to multi-agent YAML config.
        Config format:
        agents:
          coordinator:
            role: "..."
            goal: "..."
            backstory: "..."
          validator:
            role: "..."
            ...
        """
        self.agents_config = load_config(config_path)
        #self.agents_config = self.config.get('agents', {})

    def create_agent(self, agent_name: str, tools=[], llm_name="OpenAI") -> Agent:
        """Factory method to create any configured agent"""
        if agent_name not in self.agents_config:
            raise ValueError(f"Agent '{agent_name}' not found in config")
        
        return Agent(
            config=self.agents_config[agent_name],
            llm= get_LLM_model(llm_name),
            tools=tools,
            verbose= True,
            allow_delegation= False,
            #step_callback=self._process_output if agent_name == 'coordinator' else None
        )

   

    