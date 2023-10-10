from langchain.agents import Tool
import time, json
from langchain.tools import DuckDuckGoSearchRun, HumanInputRun
from util.tools import HumanInputRun
from langchain.tools.base import BaseTool
from typing import Callable, Optional
from langchain.callbacks.manager import CallbackManagerForToolRun

class CustomHumanTool(BaseTool):
    name = "human"
    description = (
        "You can ask a human for guidance when you think you "
        "got stuck or you are not sure what to do next. "
        "The input should be a question for the human."
    )
    prompt_func: Callable[[str], None]
    input_func: Callable[[None], str]

    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        self.prompt_func(query)
        return self.input_func()
    
search = DuckDuckGoSearchRun()
human = HumanInputRun()
accounts = [
    {"user_id": "a", "name": "Taufik", "email": "taufik@tokopedia.com"},
    {"user_id": "b", "name": "Pragusga", "email": "pragusga@tokopedia.com"},
]

assets = [
    {"id": "a", "name": "Laptop"},
    {"id": "b", "name": "Monitor"},
]


def get_account_api(email: str):
    time.sleep(1.5)
    for acc in accounts:
        acc_email = acc.get("email")
        if acc_email == email:
            return acc
    raise Exception(f"Account with email {email} not found")


def get_asset_api(id: str):
    time.sleep(1.5)
    for asset in assets:
        asset_id = asset.get("id")
        if asset_id == id:
            return asset
    raise Exception(f"Asset with id {id} not found")

def get_account_by_user_id(user_id: str):
    for acc in accounts:
        acc_id = acc.get("user_id")
        if acc_id == user_id:
            return acc
    return None

def request_asset_api(payload: dict[str, str]):
    if "user_id" not in payload:
        raise Exception("user_id not in the payload")
    
    if "asset_id" not in payload:
        raise Exception("asset_id not in the payload")
    
    user_id = payload["user_id"]
    asset_id = payload["asset_id"]
    
    acc = get_account_by_user_id(user_id)
    if not acc:
        raise Exception(f"user with user id {user_id} not found. Make sure your input is an user id, not an email")
    
    asset = get_asset_api(payload["asset_id"])
    if not asset:
        raise Exception(f"asset with {asset_id} not found")
    
    time.sleep(2)
    return {"success": True}

def get_account_tool(query: str):
    # print(f"ACCOUNT TOOL: {query}")
    time.sleep(1)
    try:
        res = get_account_api(query)
        return str(res)
    except Exception as e:
        return e
    
def get_asset_tool(query: str):
    # print(f"ASSET TOOL: {query}")
    try:
        res = get_asset_api(query)
        return str(res)
    except Exception as e:
        return e

def req_asset_tool(query: str):
    # print(f"REQ ASSET TOOL: {query}")
    asset_id, user_id = query.split(";")
    try:
        res = request_asset_api({"asset_id": asset_id, "user_id": user_id})
        return "Request asset suceess"
    except Exception as e:
        return e

tools = [
    Tool.from_function(
        name = "get_account_tool",
        description="Useful to get account data. Input must be an email only.",
        func=get_account_tool
    ),
    Tool.from_function(
        name = "get_asset_tool",
        description="Useful to get asset data. Input must be an asset id only.",
        func=get_asset_tool
    ),
    Tool.from_function(
        name = "req_asset_tool",
        description="Useful to request asset. The input is asset_id and user_id separated by comma without a space, for example a;b not a; b.",
        func=req_asset_tool
    ),
    Tool.from_function(
        name="duck_duck_go",
        func=search.run,
        description="Useful for when you need to answer questions about current events. Input should be a search query."
    ),
    # Tool.from_function(
    #     name="human",
    #     func=human.run,
    #     description="You can ask a human for guidance when you think you got stuck or you are not sure what to do next. The input should be a question for the human."
    # ),
    CustomHumanTool(input_func=lambda: "human", prompt_func=lambda x: print(x))
]

