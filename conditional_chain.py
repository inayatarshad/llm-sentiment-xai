from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(description = "The sentiment of the feedback")

Template1 = PromptTemplate(
    template="What is the sentiment of this feedback {feedback}\n{format_instructions}", 
    input_variables=["feedback"],
    partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=Feedback).get_format_instructions()}
)

parser = PydanticOutputParser(pydantic_object=Feedback)

chain = Template1 | llm | parser
result= chain.invoke({"feedback": "I love this product!"})
result1=chain.invoke({"feedback": "I hate the software they developed"})
print(result)
print(result1)

Template2 = PromptTemplate(
    template = "write a appropriate response to this positive feedback {feedback}",
    input_variables = ["feedback"]
)

Template3 = PromptTemplate(
    template = "write a negative response to this feedback {feedback}",
    input_variables = ["feedback"]
)
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive" , Template2 | llm| parser),
    (lambda x:x.sentiment == "negative" , Template3 | llm | parser),
    RunnableLambda(lambda x: f"Could not determine sentiment for: {x.feedback}")
)

final_chain = chain | branch_chain
result2 = final_chain.invoke({"feedback" : "the product is so bad , i want refund"}).sentiment

print(result2)