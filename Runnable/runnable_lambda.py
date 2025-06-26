from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=["topic"]
)

llm = ChatGroq(model='llama-3.1-8b-instant')

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, llm, parser)


def word_count(text):
    return len(text.split())
    
    
parallel_chain = RunnableParallel({
    "joke":RunnablePassthrough(),
    "word_count": RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result = final_chain.invoke({"topic":"AI"})
final_result = """{} \nword count - {}""".format(result['joke'], result['word_count'])
print(final_result)