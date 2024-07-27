from fastapi import FastAPI
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd
import anthropic
import json

load_dotenv()

class TranslationRequest(BaseModel):
    text: str
    member: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OpenAI.api_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
INDEX_NAME = "lyrics"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
claude_client = anthropic.Anthropic(api_key=anthropic_key)
SIMILARITY_THRESHOLD = 0.7  # 유사도 임계값 설정

openai_client = OpenAI()
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return openai_client.embeddings.create(input=[text], model=model).data[0].embedding


with open('bts.json', 'r', encoding='utf-8') as f:
    bts = json.load(f)



def generate_system_prompt(member):
    if member not in bts: raise ValueError(f"Unknown member: {member}")
    
    persona = bts[member]
    traits = "\n".join(f"- {trait}" for trait in persona['traits'])
    
    return f"{persona['description']}, 다음과 같은 특성을 가지고 대화해주세요:{traits}"


@app.post("/reset_database")
async def translate():
    pc = Pinecone(api_key=pinecone_key)
    if INDEX_NAME not in pc.list_indexes().names():
        print(f'Creating index {INDEX_NAME}')
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # The dimensionality of the vectors for text-embedding-3-small
            metric='cosine',  # The similarity metric to use when searching the index
            spec=ServerlessSpec(
                cloud='aws',
                region="us-east-1"
            )
        )

    index = pc.Index(name=INDEX_NAME)
    df = pd.read_csv('lyrics.csv')
    for column in df.select_dtypes(include=['object']):
        df[column] = df[column].str.replace('/', ',')

    for i, row in df.iterrows():
        embedding = get_embedding(text=row['ko_word'])
        index.upsert(vectors=[(str(i), embedding, {"ko_word": row['ko_word'], 
                                                   "title": row['title'], 
                                                   "en_word": row['en_word'], 
                                                   "en_meaning": row['en_meaning'],
                                                   "en_pronun": row['en_pronun'],
                                                   "lyric": row['lyric']})])
    return f"DB Complete! {index.list_paginated}"

@app.post("/translate")
async def translate(request: TranslationRequest):
    query_embedding = get_embedding(request.text)
    index = pc.Index("lyrics")
    results = index.query(vector=query_embedding, top_k=1, include_metadata=True)

    if results['matches'] and results['matches'][0]['score'] >= SIMILARITY_THRESHOLD:
        ko_word = results['matches'][0]['metadata']['ko_word']
        title = results['matches'][0]['metadata']['title']
        en_word = results['matches'][0]['metadata']['en_word']
        en_meaning = results['matches'][0]['metadata']['en_meaning']
        en_pronun = results['matches'][0]['metadata']['en_pronun']
        lyric = results['matches'][0]['metadata']['lyric']
        human_prompt = f'''성격을 반영하지만, 아래 답변형식에 맞게 답변해주세요. Please answer in "English"
            - 답변 형식
            Title : {title}
            Lyrics : {lyric} 
            Word : {en_word}
            Definition : {en_meaning} 
            How to Read : {en_pronun}
            '''
    else: 
        human_prompt = f"{request.text}, Please answer in English" 
    


    system_prompt = generate_system_prompt(request.member)
    response = claude_client.messages.create(
           model="claude-3-5-sonnet-20240620",
            max_tokens=2048,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"{human_prompt}"}
            ]
        ) 
    return { "completion": response.content}