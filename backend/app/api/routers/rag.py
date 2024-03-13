"""Endpoints."""

from fastapi import APIRouter
import gradio as gr
from app.utils.models import MessageData, RAGResponse, QAList
from app.utils.rag_functions import (
    get_zero_shot_query,
    get_compiled_rag,
    compile_rag,
    get_list_ollama_models,
)

rag_router = APIRouter()


@rag_router.get("/healthcheck")
async def healthcheck():

    return {"message": "Thanks for playing."}


@rag_router.get("/list-models")
async def list_models():
    return {"models": get_list_ollama_models()}


@rag_router.post("/zero-shot-query", response_model=RAGResponse)
async def zero_shot_query(payload: MessageData):
    return get_zero_shot_query(
        query=payload.query,
        ollama_model_name=payload.ollama_model_name,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    )


@rag_router.post("/compiled-query", response_model=RAGResponse)
async def compiled_query(payload: MessageData):
    return get_compiled_rag(
        query=payload.query,
        ollama_model_name=payload.ollama_model_name,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    )


@rag_router.post("/compile-program")
async def compile_program(qa_list: QAList):

    return compile_rag(
        items=qa_list.items,
        ollama_model_name=qa_list.ollama_model_name,
        temperature=qa_list.temperature,
        top_p=qa_list.top_p,
        max_tokens=qa_list.max_tokens,
    )
