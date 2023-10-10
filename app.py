from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from util.llm import llmc, llm
from util.tools import tools, get_account_tool, get_asset_tool, req_asset_tool
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.agent import AgentFinish
from langchain.agents import AgentExecutor, initialize_agent, AgentType, Agent, ZeroShotAgent
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True
)
question = "What is user_id with email taufik@tokopedia.com"

tools_cfg = {
    "get_asset_tool": get_asset_tool,
    "get_account_tool": get_account_tool,
    "req_asset_tool": req_asset_tool
}

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True
)

for step in agent.iter(question):
    if output := step.get("intermediate_step"):
        action, value = output[0]
        print(f"ACTION: {action}, VALUES: {value}")
        if action.tool in tools_cfg:
            tools_cfg[action.tool](value)
        # if action.tool == "GetPrime":
        #     print(f"Checking whether {value} is prime...")
        #     assert is_prime(int(value))
        # # Ask user if they want to continue
        # _continue = input("Should the agent continue (Y/n)?:\n")
        # if _continue != "Y":
        #     break

# agent_chain.run(input="Request asset with id a and email taufik@tokopedia.com")
print("="*20)
print(memory.chat_memory.messages)

# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# agent_executor.invoke({"input": "request asset with id a and user's email taufik@tokopedia.com"})

# class AskPayload(BaseModel):
#     message: str
#     histories: List[dict[str, str]]

# app = FastAPI()

# @app.post("/ask")
# def post_ask():
#     pass
