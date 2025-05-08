from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import getpass
import os
from langchain_openai import OpenAIEmbeddings


load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

app = FastAPI(
    title="FastAPI Demo", description="A simple FastAPI application", version="1.0.0"
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
    return {"message": "Welcome to FastAPI Demo"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
