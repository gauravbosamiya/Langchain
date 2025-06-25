from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(model='llama-3.1-8b-instant')

prompt1 = PromptTemplate(
    template='Generate a details finance report on {company_name}',
    input_variables=['company_name']
)

prompt2 = PromptTemplate(
    template='Give me most 3 step which was compnay has taken {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser

result = chain.invoke({'company_name':'balaji'})

print(result)

chain.get_graph().print_ascii()

