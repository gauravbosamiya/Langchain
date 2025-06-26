from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence
from dotenv import load_dotenv

load_dotenv()


prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Generte a Linkedin post about {topic}",
    input_variables=["topic"]
)

llm = ChatGroq(model='llama-3.1-8b-instant')

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet':RunnableSequence(prompt1, llm, parser),
    'linkedin':RunnableSequence(prompt2, llm, parser)
})

result = parallel_chain.invoke({'topic':'AI'})
print(result["tweet"])
print(result["linkedin"])
