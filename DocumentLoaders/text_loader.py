from itertools import chain
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('DocumentLoaders/cricket.txt', encoding='utf-8')
docs = loader.load()

llm = ChatGroq(model='llama-3.1-8b-instant')

parser = StrOutputParser()

prompt = PromptTemplate(
    template='write a 3 liner summary of the following poem -\n {poem}',
    input_variables=['poem']
)

chain = prompt | llm | parser
result = chain.invoke({'poem':docs[0].page_content})
print(result)

