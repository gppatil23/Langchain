import os

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model1 = ChatGoogleGenerativeAI(model='models/gemini-3.5-flash-lite',max_output_tokens=512,task_type='text-generation')

parser = StrOutputParser()


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Give sentiment of the feedback"
    )


parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="classify the sentiment of the following text as positive, negative, or neutral:\n{feedback}\n{format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()},
)

classification_chain = prompt1 | model1 | parser2

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback:\n{feedback}",
    input_variables=["feedback"],
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback:\n{feedback}",
    input_variables=["feedback"],
)

# IF ELSE CHAIN
branched_chain = RunnableBranch(
    (lambda x: getattr(x, "sentiment", None) == "positive", prompt2 | model1 | parser),
    (lambda x: getattr(x, "sentiment", None) == "negative", prompt3 | model1 | parser),
    RunnableLambda(lambda x: "Could not find sentiment"),
)

chain = classification_chain | branched_chain

result = chain.invoke({"feedback": "I did not like the new features in pixel phone!"})
print(result)