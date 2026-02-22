from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 

load_dotenv()

pipe = pipeline(
    model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation"
)

llm = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()
template1 = PromptTemplate(
    template = "Tell me a joke about {topic}",
    input_variables = ["topic"]
)

chain = template1 | model | parser 
result = chain.invoke({"topic": "cats"})
print(result)