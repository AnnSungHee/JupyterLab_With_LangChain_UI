from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import time

# Import 오류 방지를 위한 try-except
try:
    from chat_with_hybrid_opensource import process_chat
except ImportError:
    def process_chat(text: str) -> str:
        return f"'{text}'에 대한 문화유산 정보를 처리 중입니다... (테스트 모드)"

chatbot_router = APIRouter(tags=["🏛️ 문화유산 챗봇"])

class ChatRequest(BaseModel):
    input_text: str = Field(..., 
                           description="문화유산에 대한 질문",
                           example="경복궁의 특징은?")

class ChatResponse(BaseModel):
    response: str
    processing_time: float = Field(description="처리 시간(초)")

@chatbot_router.post("/chatbot", 
                    response_model=ChatResponse,
                    summary="문화유산 AI 챗봇",
                    description="한국 문화유산에 대해 질문하면 전문적인 답변을 제공합니다")
async def chatbot_endpoint(request: ChatRequest):
    start_time = time.time()
    
    try:
        response_text = process_chat(request.input_text)
        processing_time = time.time() - start_time
        
        return ChatResponse(
            response=response_text,
            processing_time=round(processing_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"챗봇 처리 중 오류가 발생했습니다: {str(e)}"
        )
