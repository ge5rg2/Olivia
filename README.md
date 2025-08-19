# 현재 가상환경 비활성화

deactivate

# 기존 가상환경 삭제

rm -rf .venv

# 새 가상환경 생성

python -m venv .venv

# 가상환경 활성화

source .venv/bin/activate

# 필요한 패키지만 설치

pip install -r requirements.txt
