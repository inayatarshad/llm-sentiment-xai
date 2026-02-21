from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.output_parsers import OutputFixingParser  

load_dotenv()

pipe = pipeline(
    model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation",
)

llm = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name = "fact_1", description = "description of fact 1"),
    ResponseSchema(name = "fact_2", description = "description of fact 2"),
    ResponseSchema(name = "fact_3", description = "description of fact 3")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template1 = PromptTemplate(
    template = "What are some interesting facts about {singer}? Answer in the following format: {format_instructions}",
    input_variables = ["singer"],
    partial_variables = {"format_instructions" : parser.get_format_instructions()}
)

chain = template1 | model | parser
result = chain.invoke({"singer": "Selena Gomez"})
print(result)
