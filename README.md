# AI Evaluation Platform

A production-grade Python RAG and LLM Evaluation platform using FastAPI, FAISS, and Gemini API.

## Features
- **Document Ingestion:** Parses TXT/PDF, chunks text, and generates embeddings.
- **RAG Pipeline:** Vector search via FAISS and generation via `google-genai` SDK using strict Pydantic schemas.
- **Evaluation Engine:** Uses an LLM-as-a-judge to evaluate generation against correctness, relevance, and groundedness.
- **API First:** Fully asynchronous FastAPI application.

## Setup
1. `python -m venv venv`
2. `.\venv\Scripts\activate`
3. `pip install -e .[dev]`
4. Copy `.env.example` to `.env` and add your Gemini API Key.

## Running the API
```bash
uvicorn app.main:app --reload
```
