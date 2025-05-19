# MASLLM-VaaSC: LLM-based Multi-Agent Framework for trip VaaS Composition in smart urban

This repository contains the implementation of the solution proposed in the paper:  
**"MASLLM-VaaSC: LLM-based Multi-Agent Framework for trip VaaS Composition in smart urban"** *(currently under review)*.

The solution is designed to generate a **Vehicle-as-a-Service (VaaS) Composition** in **Smart Urban Networks**,  using LLM and agentic AI.

## Getting Started

### Prerequisites
Ensure you have Python installed on your system. You can install the required dependencies using:
```bash
pip install -r requirements.txt
```
## Running the Code

To execute the solution, run the following command while commited instruction according to the scenario to be executed:
```bash
python main.py
```

## 📂 Repository Structure

The repository is organized as follows:


### Folder Descriptions:
- **`main.py`**: The main script to execute the solution and experiments.
- **`task_VaaS.py.py`**: Class VaaSTasks uses to create task object.
- **`vaas_agents.py`**: Class VaaSAgents used to creat Agent object.
- **`config/`**: Cobntains Agents (agents.yaml) and tasks (tasks.yaml) prompt templates. Also the file config.yaml contains some configuration maily LLm models
- **`tools/`**: Contains tools used by agents or tasks. each file .py contains tools (function) related to an agent.
- **`shema/`**: Contains shema that defines structured data mode. 
- **`requirements.txt`**: Lists all Python dependencies needed for the project.
- **`README.md`**: This file contains documentation for the repository.


