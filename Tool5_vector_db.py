import os
import pandas as pd

# 전역 변수로 캐시 저장
_embeddings = None
_product_vector_store = None
_optimization_vector_store = None
_product_retriever = None
_optimization_retriever = None

# 파일 경로들
excel_path = "/Users/jibaekjang/VS-Code/AI_Agent/product_data_all.xlsx"
product_db_location = "./chroma_langchain_product_db"
optimization_db_location = "./chroma_optimization_results_db"


# 임베딩 모델 로드 함수
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_ollama import OllamaEmbeddings
        _embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return _embeddings



# 상품 벡터 스토어 로드/생성
def get_product_vector_store():
    
    global _product_vector_store
    
    if _product_vector_store is None:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        
        embeddings = get_embeddings()
        
        _product_vector_store = Chroma(
            collection_name="order_records",
            persist_directory=product_db_location,
            embedding_function=embeddings
        )
        
        # 처음 실행이면 데이터 추가
        if not os.path.exists(product_db_location):
            df_product_all = pd.read_excel(excel_path)
            
            documents = []
            for i, row in df_product_all.iterrows():
                document = Document(
                    page_content=row["상품명"],
                    metadata={
                        "date": row["주문일자"],
                        "order_id": row["주문번호"],
                        "product_code": row["상품코드"],
                        "item_count": row["수량"]
                    }
                )
                documents.append(document)
            
            _product_vector_store.add_documents(documents)
    
    return _product_vector_store


# 최적화 결과 벡터 스토어 로드
def get_optimization_vector_store():
    
    global _optimization_vector_store
    
    if _optimization_vector_store is None:
        from langchain_chroma import Chroma
        
        embeddings = get_embeddings()
        
        if os.path.exists(optimization_db_location):
            _optimization_vector_store = Chroma(
                collection_name="optimization_results",
                persist_directory=optimization_db_location,
                embedding_function=embeddings
            )
        else:
            return None
    
    return _optimization_vector_store


# 상품정보 검색기
def get_product_retriever():
    
    global _product_retriever
    
    if _product_retriever is None:
        vector_store = get_product_vector_store()
        _product_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    return _product_retriever


# 최적화 결과 검색기
def get_optimization_retriever():
    
    global _optimization_retriever
    
    if _optimization_retriever is None:
        vector_store = get_optimization_vector_store()
        if vector_store is not None:
            _optimization_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    return _optimization_retriever


# 상품 데이터 검색 함수
def search_product_data(query: str) -> str:

    try:
        retriever = get_product_retriever()
        results = retriever.invoke(query)
        
        if not results:
            return f"'{query}' 관련 상품 검색 결과 없음"
        
        # Raw 데이터만 반환
        result_data = []
        for doc in results:
            result_data.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return str(result_data)
        
    except Exception as e:
        return f"상품 검색 오류: {str(e)}"


# 최적화 결과 검색
def search_optimization_results(query: str) -> str:

    try:
        retriever = get_optimization_retriever()
        
        if retriever is None:
            return "최적화 결과가 없습니다. 먼저 반찬 생산 최적화를 실행해주세요."
        
        results = retriever.invoke(query)
        
        if not results:
            return f"'{query}' 관련 최적화 결과 없음"
        
        # Raw 데이터만 반환
        result_data = []
        for doc in results:
            result_data.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return str(result_data)
        
    except Exception as e:
        return f"최적화 결과 검색 오류: {str(e)}"



# 벡터 검색(상품 정보 + 최적화 결과) Tool
def vector_search_tool(query: str) -> str:

    try:
        results = []
        
        # 상품 검색
        try:
            product_result = search_product_data(query)
            results.append(product_result)
        except:
            pass
        
        # 최적화 검색  
        try:
            optimization_result = search_optimization_results(query)
            results.append(optimization_result)
        except:
            pass
        
        if not results:
            return f"'{query}' 관련 검색 결과가 없습니다."
        
        return "\n\n".join(results)
        
    except Exception as e:
        return f"검색 오류: {str(e)}"
