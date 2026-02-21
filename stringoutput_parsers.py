from transformers import pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

print("Loading model...")

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=512
)

print("Model loaded!")

llm = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm)

print("Invoking model...")


template1 = PromptTemplate(
    template = "Answer the question: {question}",
    input_variables = ["question"]
)
template2 = PromptTemplate(
    template = "Write a 10 line summary on the following text:/n {text}",
    input_variables = ["text"]
)
prompt1 = template1.invoke({"question" : "Which companies are present in silicon valley?"})
result = model.invoke(prompt1)

prompt2 = template2.invoke({"text" : f"10 line summary of: {result.content}"})
result2 = model.invoke(prompt2)

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser

result =  chain.invoke({"question": "Which companies are present in silicon valley?"})
#print(result.content)
#print(result2.content)
