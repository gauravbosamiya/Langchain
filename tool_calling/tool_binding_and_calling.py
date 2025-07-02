from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv

load_dotenv()

@tool
def multiply(a:int, b:int)->int:
    """Multiply two numbers a and b"""
    return a*b


llm = ChatGroq(model="llama-3.1-8b-instant")

# print(multiply.invoke({"a":3,"b":5}))

llm_with_tools = llm.bind_tools([multiply])

query = HumanMessage('can you multiply 3 with 1000')
message = [query]

result = llm_with_tools.invoke(message)
message.append(result)

tool_result = multiply.invoke(result.tool_calls[0])
message.append(tool_result)

print(llm_with_tools.invoke(message).content)

