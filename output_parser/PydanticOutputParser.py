from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

class Person(BaseModel):
    name:str = Field(description="name of the person")
    age : int = Field(gt=18, description="age of the person")
    city: str = Field(description="city of that person")
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.invoke({'place':'indian'})

# print(prompt)

# result = llm.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)


chain = template | llm | parser

result = chain.invoke({'place':'sri lankan'})
print(result)
