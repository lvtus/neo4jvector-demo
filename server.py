from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import getpass
import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from models import Profile, neo4j_config


load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

app = FastAPI(
    title="Profile Matching API",
    description="API for finding matching profiles based on bio similarity and preferences",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def root():
    return {"message": "Welcome to Profile Matching API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/match/{user_id}", response_model=List[Profile])
async def find_matches(user_id: int):
    try:
        with neo4j_config.driver.session() as session:
            # First get the user's profile
            user_profile = session.run(
                "MATCH (p:Profile {user_id: $user_id}) RETURN p",
                user_id=user_id
            ).single()
            if not user_profile:
                raise HTTPException(status_code=404, detail="User profile not found")
            # Find matching profiles based on:
            # 1. Different sex
            # 2. Same location
            # 3. Similar bio (using vector similarity)
            matches = session.run("""
                MATCH (p1:Profile {user_id: $user_id})
                MATCH (p2:Profile)
                WHERE p2.user_id <> $user_id
                AND p2.sex <> p1.sex
                AND p2.ville = p1.ville
                AND p2.region = p1.region
                AND p2.pays = p1.pays
                AND p2.age >= p1.age - 5
                AND p2.age <= p1.age + 5
                WITH p1, p2,
                    vector.similarity.cosine(p1.bio_embedding, p2.bio_embedding) as similarity
                WHERE similarity > 0.5
                RETURN p2.user_id as user_id, p2.age as age, p2.bio as bio, p2.sex as sex, p2.ville as ville, p2.region as region, p2.pays as pays
                ORDER BY similarity DESC
                LIMIT 10
            """, user_id=user_id)
            return [Profile(**record) for record in matches]
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    neo4j_config.close()
