from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import time
from chatbot_router import chatbot_router

app = FastAPI(
    title="🏛️ 한국 문화유산 AI 챗봇 API",
    description="""
    ## 문화유산 전문 RAG 챗봇 시스템
    """,
    version="2.0.0"
)

# CORS 설정 유지
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Swagger UI 접근을 위해 확장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록 (기존과 동일)
app.include_router(chatbot_router, prefix="/api/chatbot")
