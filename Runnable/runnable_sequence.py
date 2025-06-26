from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='explain the following joke {text}',
    input_variables=["text"]
)

llm = ChatGroq(model='llama-3.1-8b-instant')

parser = StrOutputParser()

chain = RunnableSequence(prompt1, llm, parser, prompt2, llm, parser)
result = chain.invoke({"topic":"AI"})
print(result)