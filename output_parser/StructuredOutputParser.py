from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic')
]

parser=StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about the {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.invoke({'topic':'mutual fund'})

# ------------------------------------------
# without chain 
# ------------------------------------------
# result = llm.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)

# -------------------------------------------------
# with chain
# --------------------------------------------------
chain = template | llm | parser
result = chain.invoke({'topic':'small cap mutual fund'})
print(result)
