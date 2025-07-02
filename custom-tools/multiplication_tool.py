from langchain_core.tools import tool
from torch import mul

# step-1 create a function
def multiply(a,b):
    """Multiply two numbers"""
    return a*b

# step-2 add type hints
def multiply(a: int, b:int) -> int:
    """Multiply two numbers"""
    return a*b

# step-3 add tool decorator
@tool
def multiply(a:int, b:int)->int:
    """Multiply two numbers"""
    return a*b

result = multiply.invoke({'a':3,'b':2})
print(result)

print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.args_schema.model_json_schema())