from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

pipe = pipeline(
    model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation"
)

llm = HuggingFacePipeline(pipeline = pipe)
model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = "What are the rules of {game}?",
    input_variables = ["game"]
)
template2 = PromptTemplate(
    template = "10 line summary of {text}",
    input_variables = ["text"]
)
parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({"game": "chess"})
print(result)

chain.get_graph().print_ascii()

