from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)

print("Model loaded!")

llm = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()

template1 = PromptTemplate(
    template = "what do you know about this singer? {singer}",
    input_variables = [],
    partial_variables = {"singer" : parser.get_format_instructions()}
)

template2 = PromptTemplate(
    template = "Tell me the top selling albums of this singer {album_list}",
    input_variables = [],
    partial_variables = {"album_list" : parser.get_format_instructions()}
)

chain = template1 | model | parser

result = chain.invoke({"singer": "Taylor Swift"})
print(result)
