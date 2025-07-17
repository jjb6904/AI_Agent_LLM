# 필요 라이브러리 가져오기
from langchain_ollama.llms import OllamaLLM
from langchain.agents import initialize_agent, AgentType, Tool
import pandas as pd
import numpy as np
import os
from typing import Optional

# Tools
#import Tool1_opti_vrp           # 최적화
import Tool1_opti_vrp
import Tool2_data_analysis      # 데이터 분석
import Tool4_visualization      # plotly기반 시각화
import Tool3_sql                # sql기반 검색
import Tool5_vector_db          # vector_db연결(RAG)


# 경고메세지 제거
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='urllib3')



# 모델 객체 생성 : gemma3:12b qat방식의 양자화 모델(4비트로 축소시킴) -> 디스틸레이션
model = OllamaLLM(model="gemma3:12b-it-qat")



# Toolkits
# ============================================================================
tools = [
    Tool(
        name="dish_optimization",
        description="""
        반찬 생산라인 최적화를 수행합니다. 
        Excel 파일의 반찬 주문 데이터를 분석하여 생산라인의 최적 스케줄을 생성합니다.
        입력 형식: '파일경로' (예: '/path/to/data.xlsx')
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
)
]



# Agent
# ============================================================================
main_agent = initialize_agent(
   tools=tools,
   llm=model,
   agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
   verbose=True,
   max_iterations=5,
   handle_parsing_errors=True,
   agent_kwargs={
    'prefix': """
    You are an intelligent Korean side dish production optimization assistant. You have access to four powerful tools:

    1. data_analysis: For statistical analysis and visualization of production data
        - Use for: production trends, efficiency statistics, charts, graphs, data exploration
        - Analyze production volumes, time patterns, and performance metrics

    2. dish_optimization: For Korean side dish production line optimization
        - Use for: production scheduling, line optimization, efficiency improvement
        - Input format: 'file_path' (example: '/path/to/data.xlsx')
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

# 사용 예시 및 도움말
# ============================================================================
def print_help():
    """사용법 안내"""
    print("""
🍱 생산 최적화 Agent 시스템
================================================

💡 사용 가능한 기능:

1️⃣ 반찬 생산라인 최적화
   예시: "생산전략_비교_분석데이터_전처리.xlsx 파일을 최적화해줘"
   예시: "/path/to/data.xlsx 반찬 생산 최적화 실행"

2️⃣ 데이터 분석 및 시각화  
   예시: "월별 주문 트렌드를 차트로 보여줘"
   예시: "상위 10개 반찬의 생산량 분석해줘"

3️⃣ 최적화 결과 시각화
   예시: "간트차트로 보여줘"
   예시: "효율성 분석해줘"  
   예시: "병목 구간 히트맵 생성"
   예시: "전체 시각화해줘"

4️⃣ SQL 데이터베이스 조회
   예시: "2022년 4월 1일 주문 데이터 조회해줘"
   예시: "특정 반찬의 생산 이력을 보여줘"

5️⃣ 벡터 데이터베이스 통합 검색 (RAG)
   • 상품 검색: "김치와 비슷한 반찬들", "된장 관련 상품들"
   • 최적화 분석: "최적화 결과 어때?", "어느 라인이 효율적?", "성능 분석해줘"

🚀 전체 시스템 특징:
   • 248개 반찬 종류별 유사도 분석
   • AI 기반 전환시간 계산  
   • VRP 알고리즘으로 스케줄 최적화
   • 다중 생산라인 동시 최적화
   • 간트차트, 효율성 분석, 병목구간 히트맵
   • RAG 기반 의미적 상품 검색

📝 명령어: 'help' (도움말), 'q' (종료)
================================================
    """)

# 메인 실행 루프
# ============================================================================
def main():
    """메인 실행 함수"""
    print_help()
    
    while True:
        print("\n" + "="*60)
        question = input("🤖 질문을 입력하세요 (help/q) : ")
        print()
        
        if question.lower() == "q":
            print("👋 시스템을 종료합니다.")
            break
        elif question.lower() == "help":
            print_help()
            continue
        
        try:
            print("🔄 처리 중...")
            result = main_agent.invoke(question)
            print(f"\n✨ 답변:\n{result}")
            
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
            print("💡 다시 시도하거나 'help' 명령어로 사용법을 확인해주세요.")

# ============================================================================
if __name__ == "__main__":
    main()