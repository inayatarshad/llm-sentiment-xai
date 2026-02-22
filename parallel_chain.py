from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

pipe = pipeline(
    model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation"
)

llm = HuggingFacePipeline(pipeline = pipe)
model = ChatHuggingFace(llm=llm)

model2=ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0.7,

)

template1 = PromptTemplate(
    template = "Make notes on the following topic: {topic}",
    input_variables = ["topic"]
)

template2 = PromptTemplate(
    template = "Make a 5 MCQ 5 fill in the blanks and 5 true false test from the notes: {topic}",
    input_variables = ["topic"]
)

template3 = PromptTemplate(
    template = "Compile the notes -> {notes} the quiz ->{quiz} into a single document with proper formatting",
    input_variables = ["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": template1 | model | parser,
    "quiz": template2 | model2 | parser
})

merge_chain = template3 | model | parser
final_chain = parallel_chain | merge_chain
result = final_chain.invoke({"topic": "DSA"})
print(result)

final_chain.get_graph().print_ascii()
