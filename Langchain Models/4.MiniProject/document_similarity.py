from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2',output_dimensionality=300)

documents= [
"Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
"MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
"Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
"Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
"Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."]

query = 'tell me about rohit sharma'

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

#IMP - the value you will pass in cosine similarity is 2D vectors
scores = cosine_similarity([query_embedding],doc_embedding)[0]



index,score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1] 

print(query)
print(documents[index])
print("similarity score is:",score)


'''
print(list(enumerate(scores)))  
------->adds index for scores
[(0, np.float64(0.6965204421086388)), (1, np.float64(0.5642226737609771)), 
(2, np.float64(0.5853886908184553)), (3, np.float64(0.5937546458935119)), 
(4, np.float64(0.7247421686067114))]
'''
#---------------------------------------------------------------------------------
'''
print(sorted(list(enumerate(scores)),key=lambda x:x[1]))

--->sort asc according to cosine scores
[(1, np.float64(0.5642226737609771)), (2, np.float64(0.5853886908184553)), 
(3, np.float64(0.5937546458935119)), (0, np.float64(0.6965204421086388)), 
(4, np.float64(0.7247421686067114))]
'''
#---------------------------------------------------------------------------------

'''
print(sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]) 
fetch highest score from last(-1)
op - (4, np.float64(0.7247421686067114))
'''