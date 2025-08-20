## 콘다 가상환경 설정
conda create -n JupyterLab_With_LangChain_UI python=3.10

## 콘다 가상환경 실행
conda activate JupyterLab_With_LangChain_UI

## Jupyter Lab 실행
```
jupyter lab
```

## 의존성 설치
pip install -r requirements.txt

## 현재 설치된 패키지 목록 확인 및 출력
pip freeze > requirements.txt

## Uvicorn 실행
```bash
uvicorn main:app --reload
```

## 콘다 비활성화
> conda deactivate

## 콘다 환경 리스트
conda env list

## 콘다 환경 삭제
> conda remove --name "환경 이름" --all