# Tool2_data_analysis.py - 데이터 분석 전용 모듈
import pandas as pd
from langchain_ollama.llms import OllamaLLM
from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import warnings

# 경고 메시지 제거
warnings.filterwarnings('ignore')


# 모델 객체 생성
model = OllamaLLM(model="gemma3:12b-it-qat")


# 분석할 df 불러오기(2022_04)
df_product = pd.read_excel("/Users/jibaekjang/VS-Code/AI_Agent/product_data_all.xlsx")


def data_analysis_tool(query: str) -> str:
    """
    Agent에서 호출하는 데이터 분석 도구
    create_pandas_dataframe_agent가 알아서 분석 수행
    """
    try:
        # pandas DataFrame 에이전트 생성
        agent = create_pandas_dataframe_agent(
            llm=model,
            df=df_product,
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
        
        # 분석 실행 - 에이전트가 알아서 판단하고 처리
        result = agent.invoke(query)
        return result
        
    except Exception as e:
        return f"❌ 분석 중 오류 발생: {str(e)}"


# 테스트용 (필요시)
if __name__ == "__main__":
    test_query = "데이터 기본 정보를 보여줘"
    result = data_analysis_tool(test_query)
    print(result)