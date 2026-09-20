from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated, Optional

load_dotenv()

model = ChatGoogleGenerativeAI(model='models/gemini-3.5-flash-lite',max_output_tokens=256)

#schema
# class Review(TypedDict):
#     summary : str
#     sentiment : str

# result = structured_model.invoke("""The hardware is great, but the software feels bloated.
# There are too many pre-installed apps that I can't remove. Also, the UI looks outdated
# compared to other brands. Hoping for a software update to fix this.""")

#Annotation implementation
class Review(TypedDict):
    key_themes:Annotated[list[str],'Write down all the key themes discussed in review in list']
    summary : Annotated[str,'A brief summary of the review']
    sentiment : Annotated[str,'Return sentiment of the review either negative, possitive or neutral']
    pros:Annotated[Optional[list[str]],'Write all the pros inside the list']
    cons:Annotated[Optional[list[str]],'Write all the cons inside the list']


structured_model = model.with_structured_output(Review)


result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3
processor makes everything lightning fast-whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily
lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me
away is the 200MP camera-the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x
actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with
bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard
pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Bulky and heavy-not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors
""")

print(result)
# print(result['summary'])
# print(result['sentiment'])

'''
How it works?
since we didnt tell LLM to give this but if we provide such type of structure and call
structured output function there is a prompt for LLM behind the scene which automatically
executed
prompt looks similar like this :-
------>You are an Al assistant that extracts structured
    insights from text. Given a product review,
    extract: - Summary: A brief overview of the main
    points. - Sentiment: Overall tone of the review
    (positive, neutral, negative). Return the response
    in JSON format.
'''