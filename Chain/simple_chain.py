from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate one liner 5 intresting fact about {topic}',
    input_variables=['topic']
)

llm = ChatGroq(model='llama-3.1-8b-instant')

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({'topic':'india'})

print(result)

chain.get_graph().print_ascii()