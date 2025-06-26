from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv


load_dotenv()

prompt1 = PromptTemplate(
    template="generate a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="give explaination of the following joke - {text}",
    input_variables=["text"]
)

llm = ChatGroq(model='llama-3.1-8b-instant')

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, llm, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation' : RunnableSequence(prompt2, llm, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({"topic":"AI"})
print(result)