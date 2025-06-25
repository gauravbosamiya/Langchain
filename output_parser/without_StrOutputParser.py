from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

template1 = PromptTemplate(
    template="Write a detailed report on a {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="write a 5 line summary on the following text. /n {text}",
    input_variables=["text"]
)

prompt1 = template1.invoke({'topic':'finance'})

result = llm.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})

final_result = llm.invoke(prompt2)

print(final_result.content)
