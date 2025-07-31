# Tool2_data_analysis.py : 데이터 분석 전용 Tool
import pandas as pd
from langchain_ollama.llms import OllamaLLM
from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import warnings
import os
import re


# 경고 메시지 제거
warnings.filterwarnings('ignore')


# 모델 객체 생성
model = OllamaLLM(model="gemma3:12b-it-qat")


# 데이터 분석 도구 함수 : Agent에서 호출하는 함수
# ============================================================================
def data_analysis_tool(user_input: str) -> str:
    
    # 파일 경로만 추출
    file_match = re.search(r'([/\\][^\s]+\.xlsx?)', user_input, re.IGNORECASE)
    if not file_match:
        return "파일 경로를 찾을 수 없습니다."
    
    file_path = file_match.group(1)
    
    # 깨끗한 질문 만들기
    clean_query = re.sub(r'[/\\][^\s]+\.xlsx?', '', user_input).strip()
    if not clean_query:
        clean_query = "데이터 기본 정보와 요약 통계를 분석해줘"
    
    try:
        df = pd.read_excel(file_path)
        print(f"파일 로드 완료: {file_path}")
        
        agent = create_pandas_dataframe_agent(
            llm=model,
            df=df,
            verbose=True,
            allow_dangerous_code=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            prefix="""
            You are a Korean side dish production analyst. When asked for analysis:
            1. Set Korean font: plt.rcParams['font.family'] = 'AppleGothic'
            2. Create charts with Korean labels and titles
            3. Use plt.show() to display charts
            4. Focus on production metrics: 생산량, 효율성, 라인별 성능
            5. Consider dish categories: 무침류, 볶음류, 국물류 등     
            Always provide actionable insights for Korean food manufacturing.
            """
        )
        
        # 분석 실행
        result = agent.invoke(clean_query)
        return result
        
    except Exception as e:
        return f"오류 발생: {str(e)}"