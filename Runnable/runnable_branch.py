from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Summarize the following \n {text} in form of single para",
    input_variables=["text"]
)

llm = ChatGroq(model='llama-3.1-8b-instant')
parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, llm, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, RunnableSequence(prompt2, llm, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)
result = final_chain.invoke({"topic":"AI"})
print(result)