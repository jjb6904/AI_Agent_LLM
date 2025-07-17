import pandas as pd
import re
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.llms.ollama import Ollama
from sqlalchemy import create_engine

# MySQL 연결 설정
DB_URL = 'mysql+pymysql://langchain_user:1234@localhost:3306/langchain_db'
EXCEL_FILE = '/Users/jibaekjang/VS-Code/AI_Agent/product_data_all.xlsx'


def load_excel_to_mysql():
   """엑셀 파일을 MySQL 데이터베이스에 저장"""
   engine = create_engine(DB_URL)
   df = pd.read_excel(EXCEL_FILE)
   df.to_sql('product_order_2022_04', engine, if_exists='replace', index=False)
   print("데이터 로드 완료!")


def initialize_sql_connection():
   """SQL 연결 및 LLM 초기화"""
   engine = create_engine(DB_URL)
   db = SQLDatabase(engine)
   llm = Ollama(model="gemma3:12b-it-qat")
   return db, llm


def clean_sql_query(sql_text: str) -> str:
   """SQL 쿼리에서 마크다운 코드 블록 제거"""
   sql_text = re.sub(r'```sql\s*', '', sql_text) # 마크다운의 첫부분 제거 : sql_text에 들어있는 ''', sql, 공백문자 -> ''로 대체(빈문자열 / 사실상 삭제)
   sql_text = re.sub(r'```\s*', '', sql_text)    # 마크다운의 끝부분 제거
   return sql_text.strip()                       # 문자열의 앞 뒤 공백을 제거하고 반환함


def sql_query_tool(query: str) -> str:
   """MySQL 데이터베이스 조회 도구"""
   try:
       sql_chain = create_sql_query_chain(sql_llm, sql_db)
       sql_query = sql_chain.invoke({"question": query})
       clean_query = clean_sql_query(sql_query)
       result = sql_db.run(clean_query)
       
       return f"SQL 쿼리: {clean_query}\n결과: {result}"
       
   except Exception as e:
       return f"데이터베이스 쿼리 오류: {str(e)}"


# 초기화
load_excel_to_mysql()  # 필요시에만 실행
sql_db, sql_llm = initialize_sql_connection()