# Fast API- WEB Framework - to deploy product - Web APIs -- web API + AI --> AI APIs/RAG APIs/ Clause APIs
# Fast API- combination/Raper of Two Libraries - Pydantic ( handle data part) + Starlette (Handle Web Parts)
# FastAPI Web Server - UVICorn - Async Web server
# HTTP method -- request <--> response
# GET # POST # UPDATE    # DELETE

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Annotated
from fastapi import FastAPI, Request
from pydantic import BaseModel, StringConstraints

from app.services.chat import Answer, RAGService

@asynccontextmanager
async def lifespan(app:FastAPI) -> AsyncGenerator[None, None]:  # Lifespan =  how to start & End Server
    app.state.rag = RAGService()
    yield  # Generator Function # Hold Process
    app.state.rag.close()


app = FastAPI(title="Clause APIs", lifespan=lifespan)

class  AskQuestion(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]

# endpoint
@app.get(path="/")   # URL Path

def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post(path="/ask", response_model=Answer)
def ask_query(body: AskQuestion, req: Request) -> None:
    return req.app.state.rag.ask(body.question)

# streaming
# API-- Deploy on WEB
# 

# uv run fastapi dev app/main.py
# delete package from clause -> D:\clause>  
# install this package in web folder
# npx create-next-app@latest .
# launch ollama claude 
# give prompt "integrate the fastapi ask_query enpoint using next js with simple AI User Interface"