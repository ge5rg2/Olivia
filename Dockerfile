FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# CSM 저장소 클론
RUN git clone https://github.com/SesameAILabs/csm.git

# Python 패키지 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# CSM 의존성 설치
WORKDIR /app/csm
RUN pip install -r requirements.txt

# 메인 애플리케이션 파일 복사
WORKDIR /app
COPY . .

# 환경 변수 설정
ENV NO_TORCH_COMPILE=1
ENV TORCH_COMPILE_DISABLE=1

# 포트 설정
EXPOSE 8000

# 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]