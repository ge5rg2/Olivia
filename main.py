# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import ollama
import uuid
import asyncio
import os
from starlette.middleware.sessions import SessionMiddleware
import torch
import torchaudio
import numpy as np
from pathlib import Path
import sys
import subprocess
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Olivia AI", description="Olivia AI Chat")

# 세션 미들웨어 추가
app.add_middleware(
    SessionMiddleware,
    secret_key=os.urandom(24)
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 템플릿 및 정적 파일 설정
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 오디오 파일 저장 디렉토리 생성
AUDIO_DIR = Path("static/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# CSM 모델 경로 설정
CSM_DIR = Path("csm")
if not CSM_DIR.exists():
    print("CSM 모델이 설치되지 않았습니다. 다음 명령어를 실행해주세요:")
    print("git clone git@github.com:SesameAILabs/csm.git")
    print("cd csm")
    print("python3.10 -m venv .venv")
    print("source .venv/bin/activate")
    print("pip install -r requirements.txt")
    print("export NO_TORCH_COMPILE=1")
    sys.exit(1)

# CSM 모델 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = load_csm_1b(device=device)

# 전역 CSM 모델 인스턴스
csm_generator = None

@app.on_event("startup")
async def startup_event():
    global csm_generator
    print("CSM 모델을 로드하는 중입니다...")
    csm_generator = generator
    print("CSM 모델 로드 완료")

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
    audio_url: Optional[str] = None

# 사용자 세션 ID 가져오기 함수
async def get_user_id(request: Request) -> str:
    if "user_id" not in request.session:
        request.session["user_id"] = str(uuid.uuid4())
        chat_history[request.session["user_id"]] = []
    
    user_id = request.session["user_id"]
    if user_id not in chat_history:
        chat_history[user_id] = []
    
    return user_id

# 음성 생성 함수
async def generate_audio(text: str, user_id: str) -> Optional[str]:
    try:
        if csm_generator is None:
            print("CSM 모델이 아직 로드되지 않았습니다.")
            return None
            
        # 오디오 파일 경로 설정
        audio_path = AUDIO_DIR / f"{user_id}_{uuid.uuid4()}.wav"
        
        # 음성 생성
        audio = csm_generator.generate(
            text=text,
            speaker=0,
            context=[],
            max_audio_length_ms=10_000
        )
        
        # 오디오 저장
        torchaudio.save(str(audio_path), audio.unsqueeze(0).cpu(), csm_generator.sample_rate)
        
        return str(audio_path.relative_to("static"))
    except Exception as e:
        print(f"음성 생성 오류: {str(e)}")
        return None

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
        
        # 음성 생성
        audio_url = await generate_audio(assistant_message, user_id)
        
        return ChatResponse(
            response=assistant_message,
            history=chat_history[user_id],
            audio_url=audio_url
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
                yield f"data: {{\"error\": \"채팅 기록이 없습니다.\"}}\n\n"
                return
                
            last_message = chat_history[user_id][-1]
            if last_message["role"] != "user":
                yield f"data: {{\"error\": \"마지막 메시지가 사용자 메시지가 아닙니다.\"}}\n\n"
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
            yield f"data: [DONE]\n\n"
            
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

# WebSocket 연결을 관리하는 클래스 TODO: 통화 기능시 사용
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

manager = ConnectionManager()

# WebSocket 채팅 구현
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    if client_id not in chat_history:
        chat_history[client_id] = []
    
    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")
            model = data.get("model", "llama3.2")
            
            # 사용자 메시지를 기록에 추가
            chat_history[client_id].append({"role": "user", "content": user_message})
            
            # 사용자에게 받은 메시지 확인 응답
            await manager.send_message(
                '{"type": "status", "message": "메시지를 처리 중입니다..."}',
                client_id
            )
            
            try:
                # 스트리밍 응답 생성
                full_response = ""
                
                def stream_generator():
                    return ollama.chat(
                        model=model,
                        messages=chat_history[client_id],
                        stream=True
                    )
                
                loop = asyncio.get_event_loop()
                stream = await loop.run_in_executor(None, stream_generator)
                
                for chunk in stream:
                    content = chunk["message"]["content"]
                    full_response += content
                    await manager.send_message(
                        f'{{"type": "chunk", "content": "{content}"}}',
                        client_id
                    )
                
                # 완성된 응답을 기록에 추가
                chat_history[client_id].append({
                    "role": "assistant", 
                    "content": full_response
                })
                
                # 응답 완료 알림
                await manager.send_message(
                    '{"type": "complete"}',
                    client_id
                )
                
            except Exception as e:
                await manager.send_message(
                    f'{{"type": "error", "message": "{str(e)}"}}',
                    client_id
                )
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)

# 서버 실행 코드
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)