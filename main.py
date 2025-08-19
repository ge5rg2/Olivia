# main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Optional
import ollama
import uuid
import asyncio
import os
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI(title="Ollama 채팅 API", description="FastAPI를 사용한 Ollama 채팅 애플리케이션")

# 세션 미들웨어 추가
app.add_middleware(
    SessionMiddleware,
    secret_key=os.urandom(24)
)

# 템플릿 및 정적 파일 설정
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 채팅 기록 저장소 (실제 프로덕션에서는 데이터베이스 사용 권장)
chat_history: Dict[str, List[Dict[str, str]]] = {}

# Pydantic 모델 정의
class ChatMessage(BaseModel):
    message: str
    model: str

class ChatResponse(BaseModel):
    response: str
    history: Optional[List[Dict[str, str]]] = None
    error: Optional[str] = None

# 사용자 세션 ID 가져오기 함수
async def get_user_id(request: Request) -> str:
    if "user_id" not in request.session:
        request.session["user_id"] = str(uuid.uuid4())
        chat_history[request.session["user_id"]] = []
    
    user_id = request.session["user_id"]
    if user_id not in chat_history:
        chat_history[user_id] = []
    
    return user_id

# 홈페이지 라우트
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 일반 채팅 API 엔드포인트
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(chat_req: ChatMessage, request: Request):
    user_id = await get_user_id(request)
    user_message = chat_req.message
    model = chat_req.model
    
    # 사용자 메시지를 기록에 추가
    chat_history[user_id].append({"role": "user", "content": user_message})
    
    try:
        # FastAPI는 기본적으로 비동기이므로, 동기 호출을 위해 별도의 스레드에서 실행
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: ollama.chat(
                model=model,
                messages=chat_history[user_id],
                stream=False
            )
        )
        
        # 모델 응답을 기록에 추가
        assistant_message = response["message"]["content"]
        chat_history[user_id].append({
            "role": "assistant", 
            "content": assistant_message
        })
        
        return ChatResponse(
            response=assistant_message,
            history=chat_history[user_id]
        )
    
    except Exception as e:
        return ChatResponse(error=str(e))


# 스트리밍 채팅 API 엔드포인트 
@app.get("/api/chat/stream")
async def stream_chat_get(request: Request):
    user_id = await get_user_id(request)
    
    async def generate():
        try:
            # 마지막 사용자 메시지 가져오기
            if not chat_history[user_id]:
                yield "data: {\"error\": \"채팅 기록이 없습니다.\"}\n\n"
                return
                
            last_message = chat_history[user_id][-1]
            if last_message["role"] != "user":
                yield "data: {\"error\": \"마지막 메시지가 사용자 메시지가 아닙니다.\"}\n\n"
                return
                
            # 마지막 요청의 모델 정보 가져오기
            model = chat_history[user_id][-1].get("model", "olivia:beta")  # 모델 정보가 없으면 기본값 사용
            
            # 비동기 스트리밍을 위한 처리
            def stream_generator():
                return ollama.chat(
                    model=model,
                    messages=chat_history[user_id],
                    stream=True
                )
            
            loop = asyncio.get_event_loop()
            stream = await loop.run_in_executor(None, stream_generator)
            
            full_response = ""
            for chunk in stream:
                content = chunk["message"]["content"]
                
                # 글자 단위로 분할하여 전송
                for char in content:
                    full_response += char
                    yield f"data: {char}\n\n"
                    await asyncio.sleep(0.01)
            
            # 완성된 응답을 기록에 추가
            chat_history[user_id].append({
                "role": "assistant", 
                "content": full_response
            })
            
            # 응답 완료 후 종료 신호 전송
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

# 스트리밍 채팅 API 엔드포인트 포스트
@app.post("/api/chat/stream")
async def stream_chat_post(chat_req: ChatMessage, request: Request):
    user_id = await get_user_id(request)
    user_message = chat_req.message
    model = chat_req.model
    
    # 사용자 메시지를 기록에 추가
    chat_history[user_id].append({"role": "user", "content": user_message})
    
    return {"status": "success"}


# 채팅 기록 가져오기 API
@app.get("/api/history")
async def get_history(request: Request):
    user_id = await get_user_id(request)
    return chat_history[user_id]

# 채팅 기록 지우기 API
@app.post("/api/clear-history")
async def clear_history(request: Request):
    user_id = await get_user_id(request)
    chat_history[user_id] = []
    return {"status": "success"}

# 서버 실행 코드
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)