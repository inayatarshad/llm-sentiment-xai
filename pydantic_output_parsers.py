#tinyllama is too small, whie=le dealing with pydantic output parser and structured output parsers that give json outputs try using mistral or other big models
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field ,EmailStr

load_dotenv()

class Student(BaseModel):
    name: str = Field(description = "The name of the student")
    age: int = Field(description = "the age of the student", gt = 18)
    email: EmailStr = Field(description = "The email of the student")

pipe = pipeline(
    model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation"
)

llm = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm)

parser = PydanticOutputParser(pydantic_object = Student)

template = PromptTemplate(
    template = "Generate a student profile with name, age and email",
    input_variables = [],
    partial_variables = {"format_instructions" : parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({})
print(result)