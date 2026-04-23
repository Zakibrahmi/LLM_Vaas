######### CLASSIFICATION EVALUATION VERSION 2 ######### --valid and runnable: 
#Zero-shot CrewAI Version with 5 tasks with yaml prompt and code saved in csv file ##########################
#The VaaSComposer agent then receives these verified requests and
#performs the following tasks sequentially: (1) identifying the type (or class) of the optimization problem and the
#associated metric; (2) formulating the corresponding objective function; (3) generating the executable code; and (4)
#computing the optimized objective metric.
#############################################################################################

from crewai import Agent, LLM
import time
import yaml
import pandas as pd
import csv
import json
import re
import os

# ==============================
# FILES
# ==============================
f1 = "test.csv"
f2 = "test2_classified_llmgptCoT.csv"

# ==============================
# LOAD PROMPTS
# ==============================

#Zero-Shot
'''
with open("./zeroshotPrompt1_1.yaml", "r") as f:
    prompt_task1 = yaml.safe_load(f)

with open("./zeroshotPrompt2_1.yaml", "r") as f:
    prompt_task2 = yaml.safe_load(f)

with open("./zeroshotPrompt3_2.yaml", "r") as f:
    prompt_task3 = yaml.safe_load(f)

with open("./zeroshotPrompt4_1.yaml", "r") as f:
    prompt_task4 = yaml.safe_load(f)
print("With Zero-shot: Current working directory:", os.getcwd())

#Few-Shot
with open("./fewShotPrompt1.yaml", "r") as f:
    prompt_task1 = yaml.safe_load(f)

with open("./fewShotPrompt2.yaml", "r") as f:
    prompt_task2 = yaml.safe_load(f)

with open("./fewShotPrompt3.yaml", "r") as f:
    prompt_task3 = yaml.safe_load(f)

with open("./zeroshotPrompt4_1.yaml", "r") as f:
    prompt_task4 = yaml.safe_load(f)

print("With Fiew-shot: Current working directory:", os.getcwd())
'''

#CoT-based prompts
with open("./CoTPrompt1.yaml", "r") as f:
    prompt_task1 = yaml.safe_load(f)

with open("./CoTPrompt2.yaml", "r") as f:
    prompt_task2 = yaml.safe_load(f)

with open("./CoTPrompt3.yaml", "r") as f:
    prompt_task3 = yaml.safe_load(f)

with open("./zeroshotPrompt4_1.yaml", "r") as f:
    prompt_task4 = yaml.safe_load(f)

print("With CoT-Prompts: Current working directory:", os.getcwd())

# ==============================
# JSON EXTRACTOR (KEY FIX)
# ==============================
def extract_json(text):
    try:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            json_str = match.group(0)
            json_str = json_str.replace('""', '"')
            return json.loads(json_str)
    except:
        pass
    return {"Best Score": None, "Best Services": None}

# ==============================
# LLM
# ==============================
#baseline LLM
#llm = LLM(model="ollama/llama3", base_url="http://localhost:11434")

# strong raisonning and problem solvability #but long take much time
#llm = LLM(model="ollama/deepseek-v3.2:cloud", base_url="http://localhost:11434") 

# following prompt precisely respecting conditions :glm general language model Zhipu AI -- validated
#llm = LLM(model="ollama/glm-4.6:cloud", base_url="http://localhost:11434") 

#fast = low latency -- validate fast
#llm = LLM(model="ollama/gemini-3-flash-preview:cloud", base_url="http://localhost:11434")

#fats+raisonning : modèle GPT open source de 20 milliards de paramètres exécuté localement via Ollama.
llm = LLM(model="ollama/gpt-oss:20b-cloud", base_url="http://localhost:11434")

# ==============================
# DATA STRUCTURES
# ==============================
class Service:
    def __init__(self, id, start, end, cost, time, reputation, service_type):
        self.id = id
        self.start = start
        self.end = end
        self.cost = cost
        self.time = time
        self.reputation = reputation
        self.type = service_type

    def to_dict(self):
        return {
            "id": self.id,
            "start_region": self.start,
            "end_region": self.end,
            "cost": self.cost,
            "time": self.time,
            "reputation": self.reputation,
            "type": self.type
        }

# ==============================
# GRAPH
# ==============================
services = [
    Service("s0", "South Region", "City Center", 10, 5, 4.5, "smart"),
    Service("s1", "South Region", "City Center", 10, 5, 5, "smart"),
    Service("s2", "South Region", "City Center", 5, 15, 1, "smart"),
    Service("s3", "South Region", "City Center", 5, 17, 0.5, "smart"),
    Service("s4", "South Region", "City Center", 20, 25, 4.0, "classic"),
    Service("s7", "City Center", "Metro Station", 0, 15, 4.5, "electric"),
    Service("s8", "City Center", "Metro Station", 8, 5, 4.2, "smart"),
    Service("s10", "Metro Station", "Airport", 4, 30, 4, "classic"),
    Service("s11", "Metro Station", "Airport", 30, 3, 1, "classic"),
]

path_regions = ["South Region", "City Center", "Metro Station", "Airport"]

services_by_step = {
    ("South Region", "City Center"): [s.to_dict() for s in services[:5]],
    ("City Center", "Metro Station"): [services[5].to_dict(), services[6].to_dict()],
    ("Metro Station", "Airport"): [services[7].to_dict(), services[8].to_dict()],
}

# ==============================
# AGENT
# ==============================
class VaaSComposer:
    def __init__(self, llm):
        self.llm = llm
        self.class_label = None
        self.objective_func = None
        self.composed_services = None

        self.agent = Agent(
            role="VaaS Composer",
            goal="Optimize trip services",
            backstory="Transportation optimization expert",
            llm=llm
        )

    def task_classify(self, user_request):
        full_prompt = prompt_task1["system"] + "\n\n" + \
                      prompt_task1["user"].format(user_request=user_request)
        self.class_label = self.agent.llm.call(full_prompt).strip()
        return self.class_label

    def task_objective(self):
        full_prompt = prompt_task2["system"] + "\n\n" + \
                      prompt_task2["user"].format(class_label=self.class_label)
        self.objective_func = self.agent.llm.call(full_prompt).strip()
        return self.objective_func

    #def task_compose_services(self):
    def task_compose_services(self, user_request):
        full_prompt = prompt_task3["system"] + "\n\n" + \
                      prompt_task3["user"].format(
                          user_request=user_request,   # ✅ ADD THIS
                          path_regions=path_regions,
                          services_by_step=services_by_step,
                          objective_func=self.objective_func,
                          class_label=self.class_label
                      )
        self.composed_services = self.agent.llm.call(full_prompt).strip()
        return self.composed_services

    def run(self, user_request):
        self.task_classify(user_request)
        self.task_objective()
        #self.task_compose_services()
        self.task_compose_services(user_request)   # ✅ IMPORTANT FIX

        return {
            "class_label": self.class_label,
            "objective_function": self.objective_func,
            "composed_services": self.composed_services
        }

# ==============================
# LOAD REQUESTS
# ==============================
df_requests = pd.read_csv(f1)
test_requests = df_requests['Request'].tolist()

print(f"Loaded {len(test_requests)} requests")

# ==============================
# RUN PIPELINE
# ==============================
composer = VaaSComposer(llm)
results_list = []

for req in test_requests:
    print("\n=== Request ===\n", req)

    start = time.time()
    result = composer.run(req)
    duration = time.time() - start

    # SHOW FULL LLM OUTPUT (for analysis)
    print(f"\n--- RAW LLM OUTPUT ---{llm.model}")
    print(result['class_label'])    
    print(result['composed_services'])

    # ✅ EXTRACT CLEAN JSON
    parsed = extract_json(result['composed_services'])

    # SAVE CLEAN DATA ONLY
    results_list.append({
        "Request": req,
        "Class": result['class_label'],
        "Best_Score": parsed.get("Best Score"),
        "Best_Services": parsed.get("Best Services"),
        "Time_s": round(duration, 3),
        "OF":result['objective_function']
    })

# ==============================
# SAVE CSV
# ==============================
df_results = pd.DataFrame(results_list)

df_results.to_csv(
    f2,
    index=False,
    encoding="utf-8-sig",
    quoting=csv.QUOTE_ALL
)

print("\n✅ Results saved CLEAN (JSON only)")