from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

import json_output_parser

load_dotenv()

pipe = pipeline(
    model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation"
)

llm = HuggingFacePipeline(pipeline = pipe)
model = ChatHuggingFace (llm=llm)
parser = json_output_parser.JsonOutputParser()

template = PromptTemplate(
    template = "What do you like about {topic}",
    input_variables = [],
    partial_variables = {"format_instructions" : parser.get_format_instructions()}
)

chain = template | model | parser 
result = chain.invoke({"topic": "programming"})
print(result)

#again json output wont be returned because , tinyllama is a small model
