from pydantic import BaseModel
from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Profile(BaseModel):
    user_id: int
    age: int
    bio: Optional[str] = None
    sex: str
    ville: str
    region: str
    pays: Optional[str] = None
    bio_embedding: Optional[list[float]] = None

class Neo4jConfig:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        
    def close(self):
        self.driver.close()

    def create_constraints(self):
        with self.driver.session() as session:
            # Create unique constraint on user_id
            session.run("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (p:Profile) REQUIRE p.user_id IS UNIQUE")

    def create_vector_index(self):
        with self.driver.session() as session:
            # Create vector index for bio embeddings
            session.run("""
                CREATE VECTOR INDEX bio_embeddings IF NOT EXISTS
                FOR (p:Profile)
                ON p.bio_embedding
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }}
            """)

neo4j_config = Neo4jConfig()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large") 