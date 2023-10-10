from langchain.agents import initialize_agent, AgentType
from util.tools import tools
from util.llm import llmc
from flask import Flask, request, jsonify
from typing import List
import re



app = Flask(__name__)

PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

def create_suffix(histories: str = ""):
    return "Begin! If you can't answer the question or perform a command, stop the process immediately.\n\nQuestion: {input}\n" + histories + "\n" + "Thought:{agent_scratchpad}"


# question = "Who is Taufik Pragusga?"
histories = []

def run_agent(question: str, prev_histories: List[str] = [], human_obs: str = ""):
    prev_histories_str = ""
    if human_obs != "" and len(prev_histories) > 0:
        prev_histories[-1] = prev_histories[-1] + f"\nObservation: {human_obs}"
    for prev in prev_histories:
        prev_histories_str += prev +"\n\n"
        
    SUFFIX = create_suffix(prev_histories_str)
    histories = []
    
    agent = initialize_agent(
    tools, 
    llmc, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    agent_kwargs={
        'prefix':PREFIX,
        'format_instructions':FORMAT_INSTRUCTIONS,
        'suffix':SUFFIX
    },
    handle_parsing_errors=True
    )
    
    for step in agent.iter(question):
        print(f"STEP: {step}")
        if output := step.get("intermediate_step"):
            action, value = output[0]
            message = "Thought: " + action.log
            

            if not isinstance(value, str):
                value = str(value)
            
            if action.tool != "human":
                message += "\nObservation: " + value
                
            thought_match = re.search(r'Thought:\s*(.*?)\n', message)
            action_match = re.search(r'Action:\s*(.*?)\n', action.log)
            action_input_match = re.search(r'Action Input:\s*(.*?)$', action.log)
            
            thought = thought_match.group(1) if thought_match else None
            action_s = action_match.group(1) if action_match else None
            action_input = action_input_match.group(1) if action_input_match else None
            
            # Menampilkan hasil ekstraksi
            print("Thought:", thought)
            print("Action:", action_s)
            print("Action Input:", action_input)
            print("Observation:", value)
            
            obj = {
                "thought": thought,
                "action": action_s,
                "action_input": action_input,
                "observation": value
            }

            histories.append({
                "str": message,
                "obj": obj
            })
            
            if action.tool == "human":
                obj.pop("observation")
                break
        elif output := step.get("output"):
            message = "Thought: I now know the final answer\n" + f"Final Answer: {output}"
            obj = {"thought": "I now know the final answer", "final_answer": output}
            histories.append({
                "obj": obj,
                "str": message
            })
            break
    
    return histories


@app.post("/ask")
def post_ask():
    payload = request.get_json()
    initial_question = payload["initial_question"]
    histories = payload["histories"]
    obs = payload["obs"]
    ret_histories = run_agent(initial_question, histories, obs)
    return jsonify({"ret_histories": ret_histories})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001)