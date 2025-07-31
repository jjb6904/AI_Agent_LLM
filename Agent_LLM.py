# 필요 라이브러리 가져오기
from langchain_ollama.llms import OllamaLLM
from langchain.agents import initialize_agent, AgentType, Tool
import pandas as pd
import numpy as np
import os
from typing import Optional
from langchain.memory import ConversationBufferWindowMemory


# Tool
import Tool1_opti_vrp           # 최적화
import Tool2_data_analysis      # 데이터 분석
import Tool4_visualization      # plotly기반 시각화
import Tool3_sql                # sql기반 검색
import Tool5_vector_db          # vector_db연결(RAG)



# LLM : Gemma3:12b 양자화 버전
# ============================================================================
model = OllamaLLM(model="gemma3:12b-it-qat",
                  temperature=0.1,     # 아주 약한 창의성 0.1정도
                  repeat_penalty=1.1,  # 반복되는 토큰에 패널티 부여(10%만큼)
                  top_k=20,            # 가장 높은 확률을 지닌 토큰 20개만 답변생성에 사용
                  timeout=30,          # 추론시간 30초로 제한
                  verbose= False)      # 추론과정 생략




# Toolkits
# ============================================================================
tools = [
    Tool(
        name="dish_optimization",
        description="""
        반찬 생산라인 최적화를 수행합니다. 
        Excel 파일의 반찬 주문 데이터를 분석하여 생산라인의 최적 스케줄을 생성합니다.
        입력 형식: '/Users/jibaekjang/VS-Code/AI_Agent/product_data_2022_04_01.xlsx'파일 최적화 해줘
        반찬 생산 스케줄링, 최적화, 생산라인 배치 등의 질문에 사용하세요.
        """,
        func=Tool1_opti_vrp.dish_optimization_tool
    ),
    Tool(
        name="data_analysis", 
        description="반찬 생산 데이터 분석 및 시각화를 수행합니다. 트렌드, 분포, 평균, 차트, 그래프 등이 필요할 때 사용하세요.",
        func=Tool2_data_analysis.data_analysis_tool
    ),
    Tool(
       name="sql_database_query",
       description = "MySQL 데이터베이스에서 정확한 데이터 조회. 사용자가 특정 날짜를 언급하면 반드시 해당 날짜 조건을 WHERE 절에 포함해야 함",
       func=Tool3_sql.sql_query_tool
    ),
    Tool(
        name="optimization_visualizer",
        description="""
        최적화 결과를 전문적인 차트로 시각화합니다.
        - 간트차트: 생산라인별 타임라인 시각화
        - 효율성 분석: 라인별 성능 대시보드
        - 병목 구간: 시간대별 히트맵 분석  
        - 종합 리포트: 텍스트 기반 분석 보고서
        - 전체 시각화: 모든 차트 한번에 생성
        최적화 실행 후에 사용하세요.
        """,
        func=Tool4_visualization.optimization_visualizer_tool
    ),
    Tool(
    name="vector_search",
    description="""
    벡터 데이터베이스 통합 검색:
    - 상품 유사도 검색: '김치와 비슷한 반찬들', '된장 관련 상품들'
    - 최적화 결과 검색: '최적화 결과 어때?', '효율성 분석', '라인별 성능'
    자연어로 상품이나 최적화 결과를 검색할 때 사용하세요.
    """,
    func=Tool5_vector_db.vector_search_tool
)]



# Memory : 대화 맥락 파악
# ============================================================================
memory = ConversationBufferWindowMemory(
    k=3,  # 마지막 3개 대화만 기억
    memory_key="chat_history", 
    return_messages=True
)



# Agent
# ============================================================================
main_agent = initialize_agent(
   tools=tools,
   llm=model,
   agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
   verbose=True,
   memory=memory,
   max_iterations=3,
   handle_parsing_errors=True,
   agent_kwargs={
    'prefix': """
    You are an intelligent Korean side dish production optimization assistant. 
    
    IMPORTANT: If you are not certain about any information, always respond with "I don't know" or "I'm not sure" rather than making assumptions or guessing.

    You have access to four powerful tools:
    1. data_analysis: For statistical analysis and visualization of production data
        - Use for: production trends, efficiency statistics, charts, graphs, data exploration
        - Analyze production volumes, time patterns, and performance metrics

    2. dish_optimization: For Korean side dish production line optimization
        - Use for: production scheduling, line optimization, efficiency improvement
        - Input format: 'file_path' (example: '/Users/jibaekjang/VS-Code/AI_Agent/product_data_2022_04_01.xlsx')
        - This tool analyzes dish similarity, calculates changeover times, and creates optimal schedules
        - Optimizes production lines for maximum efficiency

    3. optimization_visualizer: For visualizing optimization results with professional charts
        - Use for: gantt charts, efficiency analysis, bottleneck identification
        - Creates interactive charts showing production timelines and performance metrics
        - Use ONLY after running dish_optimization first

    4. sql_database_query: For querying MySQL database with precise data retrieval
        - Use for: specific date queries, production records, historical data analysis
        - Always include date conditions in WHERE clause when user mentions specific dates

    5. vector_search: For searching both product similarity and optimization results using vector database
        - Product search: finding similar products, natural language product search
        - Optimization analysis: querying saved optimization results, performance analysis

    Tool selection guide:
        - Data analysis: 생산 데이터 분석/시각화가 필요할 때  
        - Dish optimization: 반찬 생산라인 최적화가 필요할 때
        - Optimization visualizer: 최적화 결과를 차트로 보고 싶을 때
        - SQL query: 데이터베이스 조회가 필요할 때
        - Vector search: 상품 유사도 검색이나 자연어 상품 검색이 필요할 때

    Always provide helpful answers in Korean, explain which tool you're using and why, and focus on improving production efficiency and reducing manufacturing costs.
"""
   }
)



# 메인 실행함수 정의
# ============================================================================
def main():
    """메인 실행 함수"""
    while True:
        print("\n" + "="*60)
        question = input("질문을 입력하세요(종료 : q) : ")
        print()
        
        if question.lower() == "q":
            print("시스템을 종료합니다.")
            break
        try:
            print("처리 중...")
            result = main_agent.invoke(question)

            clean_output = result['output'].replace('<end_of_turn>', '').strip()
            print(f"\n[답변] \n{clean_output}")
            
        except Exception as e:
            print(f"오류 발생: {str(e)}")



# 메인 실행
# ============================================================================
if __name__ == "__main__":
    main()
