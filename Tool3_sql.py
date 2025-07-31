# Tool3_sql_query.py : SQL 쿼리 전용 Tool
import pandas as pd
import re
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.llms.ollama import Ollama
from sqlalchemy import create_engine, text

# 설정
DB_URL = 'mysql+pymysql://langchain_user:1234@localhost:3306/langchain_db'
EXCEL_FILE = '/Users/jibaekjang/VS-Code/AI_Agent/product_data_all.xlsx'

# 전역 변수
db = None
llm = None


# 데이터베이스 연결 및 초기 설정
# ============================================================================
def setup_database():
    global db, llm
    
    try:
        engine = create_engine(DB_URL)
        
        # 테이블 존재 확인 (text() 함수 사용)
        with engine.connect() as conn:
            result = conn.execute(text("SHOW TABLES LIKE 'products'"))
            table_exists = result.fetchone() is not None
        
        if not table_exists:
            # 테이블이 없을 때만 엑셀 파일 로드
            df = pd.read_excel(EXCEL_FILE)
            df.to_sql('products', engine, if_exists='replace', index=False)
            print(f"테이블 'products' 생성 완료 ({len(df)}행)")
        else:
            print("기존 테이블 'products' 사용")
        
        # LangChain DB 연결
        db = SQLDatabase(engine)
        llm = Ollama(model="gemma3:12b-it-qat")
        print("DB 연결 완료!")
        return True
        
    except Exception as e:
        print(f"DB 설정 실패: {e}")
        return False


# SQL 쿼리 도구 함수 : Agent에서 호출하는 함수
# ============================================================================
def sql_query_tool(question: str) -> str:
    
    # DB 연결 확인
    if db is None or llm is None:
        if not setup_database():
            return "데이터베이스 연결 실패"
    
    try:
        # SQL 쿼리 생성
        chain = create_sql_query_chain(llm, db)
        sql_query = chain.invoke({"question": question})
        
        # SQL 정리 - SQLQuery: 이후 부분만 추출
        if "SQLQuery:" in sql_query:
            clean_sql = sql_query.split("SQLQuery:")[-1].strip()
        else:
            clean_sql = sql_query
        
        # 마크다운, 세미콜론 제거
        clean_sql = re.sub(r'```sql|```|;', '', clean_sql).strip()
        
        print(f"실행 SQL: {clean_sql}")
        
        # 쿼리 실행
        result = db.run(clean_sql)
        
        return f"결과:\n{result}"
        
    except Exception as e:
        return f"쿼리 오류: {str(e)}"