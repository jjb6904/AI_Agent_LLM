# Tool1_opti_vrp.py : 생산 최적화 Tool
import os
import sys
import datetime
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings



# 설정 상수
# =====================================================================
DEFAULT_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
DEFAULT_NUM_LINES = 4
DEFAULT_MAX_TIME = 240
DEFAULT_BASE_CHANGEOVER_TIME = 2
DEFAULT_MAX_ADDITIONAL_TIME = 2
DEFAULT_UNKNOWN_COOKING_TIME = 3
UNIT_TIME_PER_QUANTITY = 0.01
OPTIMIZATION_DB_PATH = "./chroma_optimization_results_db"
OPTIMIZATION_TIME_LIMIT = 60



# 유틸리티 클래스
# =====================================================================
class StdoutCapture:
    """표준 출력 캡처를 위한 클래스"""
    
    def __init__(self):
        self.contents = []
        
    def write(self, data: str) -> None:
        self.contents.append(data)
        
    def flush(self) -> None:
        pass
        
    def get_output(self) -> str:
        return ''.join(self.contents)



# 전역 변수(Agent 연동용)
# =====================================================================
last_optimization_output = None
last_optimization_text = None
current_file_name = None




## 1. 벡터 임베딩 관련 함수 ##
# =====================================================================
# 1-1. 반찬명 벡터 임베딩 생성 함수
def create_dish_embeddings(df: pd.DataFrame, 
                          dish_column: str = '상품명',
                          model_name: str = DEFAULT_MODEL_NAME) -> Dict[str, Any]:
    
    model = SentenceTransformer(model_name)
    
    # 고유한 반찬명 추출
    unique_dishes = df[dish_column].unique().tolist()
    
    # 벡터 임베딩 생성
    embeddings = model.encode(unique_dishes, show_progress_bar=True)
    
    print(f"임베딩 완료 / 차원 : {embeddings.shape}")
    
    return {
        'dish_names': unique_dishes,
        'embeddings': embeddings,
        'embedding_dim': embeddings.shape[1],
        'model': model
    }

# 1-2. 전환시간 계산 함수
def calculate_changeover_matrix(embedding_result: Dict[str, Any],
                               base_time: int = DEFAULT_BASE_CHANGEOVER_TIME,
                               max_additional_time: int = DEFAULT_MAX_ADDITIONAL_TIME) -> pd.DataFrame:

    dish_names = embedding_result['dish_names']
    embeddings = embedding_result['embeddings']
    
    # 코사인 거리 계산 : 코사인 유사도 기반 전환시간 계산용
    cosine_dist_matrix = cosine_distances(embeddings)
    
    # 코사인 거리를 전환시간으로 변환 : 기본시간 + (코사인 거리 * 최대 추가시간)
    changeover_matrix = base_time + (cosine_dist_matrix * max_additional_time)
    
    # 대각선 요소는 0 : 같은 반찬끼리는 전환시간이 존재하지 않음
    np.fill_diagonal(changeover_matrix, 0)
    
    # DataFrame 변환
    changeover_df = pd.DataFrame(
        changeover_matrix,
        index=dish_names,
        columns=dish_names
    )
    
    print(f"전환 시간 범위: {changeover_matrix.min():.1f}분 ~ {changeover_matrix.max():.1f}분")
    
    return changeover_df




## 2. 조리시간 관련 함수 ##
# =====================================================================
# 2-1. 반찬별 조리시간 데이터
def get_dish_cooking_times() -> Dict[str, int]:

    return {
        # 무침류 (1-3분)
        '콩나물무침': 1, '미나리무침': 2, '무생채': 2, '시금치나물 - 90g': 3,
        '새콤달콤 유채나물무침': 3, '새콤달콤 방풍나물무침': 3, '닭가슴살 두부무침': 3,
        '새콤달콤 돌나물무침': 2, '새콤달콤 오징어무침': 3, '새콤달콤 오이달래무침': 2,
        '브로콜리 두부무침 - 100g': 3, '매콤 콩나물무침': 3, '오이부추무침': 2,
        '참깨소스 시금치무침': 3, '(gs재등록) 닭가슴살 참깨무침': 3, '무말랭이무침': 3,
        '오징어무말랭이무침 - 130g': 3, '참나물무침 - 80g': 2, '연근참깨무침': 3,
        '참깨소스 버섯무침 - 100g': 3, '톳두부무침': 3, '가지무침': 3,
        '숙주나물무침 - 90g': 3, '달래김무침': 2, '새콤 꼬시래기무침': 3,
        '오이부추무침 - 100g': 2, '참깨두부무침 - 200g': 3, '새콤 오이무생채': 3,
        '새콤달콤 오징어무침 - 110g': 3, '새콤달콤 도라지무침': 3, '콩나물무침 - 90g': 2,
        '무생채 - 100g': 2, '파래김무침': 2, '무나물 - 100g': 2,
        
        # 김치/절임류 (1-3분)
        '물김치 - 350g': 2, '백김치 - 350g': 2, '양파고추 장아찌 - 150g': 2,
        '유자향 오이무피클 - 240g': 2, '깻잎 장아찌': 2, '셀러리 장아찌': 2,
        '깍두기': 3, '나박김치': 3, '총각김치': 3, '곰취 장아찌': 2,
        '볶음김치': 3, '볶음김치_대용량': 3,
        
        # 국물류 (3-5분)
        '아이들 된장국': 4, '감자국': 5, '계란국(냉동)': 3, '순한 오징어무국': 5,
        '시래기 된장국(냉동)': 5, '달래 된장찌개': 4, '근대 된장국(냉동)': 5,
        '된장찌개': 5, '동태알탕': 5, '맑은 콩나물국(냉동)': 4, '오징어 무국(냉동)': 5,
        '냉이 된장국(냉동)': 4, '한우 소고기 감자국': 5, '우리콩 강된장찌개': 5,
        '맑은 순두부찌개': 4, '계란 황태국(냉동)': 4, '오징어찌개': 5,
        '시금치 된장국(냉동)': 4, '김치콩나물국(냉동)': 5, '한우사골곰탕(냉동) - 600g': 5,
        '한우 소고기 무국(냉동) - 650g': 5, '한우 소고기 미역국(냉동) - 650g': 5,
        '맑은 동태국': 5, '콩나물 황태국(냉동)': 4, '배추 된장국(냉동)': 5,
        
        # 찌개류 (5-8분)
        '한돈 돼지김치찌개': 7, '한돈 청국장찌개': 6, '동태찌개': 6,
        '한돈 돼지돼지 김치찌개_쿠킹박스': 7, '한돈 돼지고추장찌개': 7, '알탕': 8,
        
        # 볶음류 (3-5분)
        '한우 무볶음': 4, '고추장 멸치볶음': 3, '야채 어묵볶음': 4,
        '느타리버섯볶음 - 90g': 3, '풋마늘 어묵볶음': 4, '애호박볶음': 3,
        '새우 애호박볶음 - 110g': 4, '한돈 가지볶음': 4, '들깨머위나물볶음': 3,
        '도라지볶음 - 80g': 3, '감자햄볶음': 4, '느타리버섯볶음': 3,
        '토마토 계란볶음': 3, '미역줄기볶음': 3, '건곤드레볶음': 4,
        '건고사리볶음 - 80g': 3, '호두 멸치볶음_대용량': 4, '미역줄기볶음_대용량': 4,
        '감자채볶음': 3, '건취나물볶음 - 80g': 3, '호두 멸치볶음': 4,
        '꼴뚜기 간장볶음': 5, '새우오이볶음': 3, '소고기 야채볶음_반조리': 5,
        '들깨시래기볶음 - 90g': 4, '보리새우 간장볶음': 4, '소고기 우엉볶음': 5,
        '한우오이볶음': 4, '건가지볶음': 3, '들깨고구마 줄기볶음 - 80g': 3,
        '한우오이볶음 - 100g': 4, '야채 어묵볶음 - 80g': 4, '감자채볶음 - 80g': 3,
        '매콤 어묵볶음': 4, '건피마자볶음': 3, '한우 무볶음 - 110g': 4,
        '감자햄볶음 - 80g': 4, '소고기 우엉볶음 - 80g': 5, '꽈리멸치볶음 - 60g': 3,
        '호두 멸치볶음 - 60g': 4, '미역줄기볶음 - 60g': 3, '꽈리멸치볶음_대용량': 4,
        '소고기 가지볶음': 5, '간장소스 어묵볶음': 4, '건호박볶음': 3,
        '고추장 멸치볶음_대용량': 4, '한돈 냉이 버섯볶음밥 재료': 5,
        '상하농원 케찹 소세지 야채볶음': 4, '상하농원 햄 어묵볶음': 4,
        
        # 제육/고기볶음류 (3-5분)
        '한돈 매콤 제육볶음_반조리 - 500g': 5, '주꾸미 한돈 제육볶음_반조리': 5,
        '한돈 김치두루치기_반조리': 5, '한돈 미나리 고추장불고기_반조리': 5,
        '한돈 대파 제육볶음_반조리': 5, '주꾸미 야채볶음_반조리': 5,
        '오징어 야채볶음_반조리': 4, '간장 오리 주물럭_반조리': 5,
        '한돈 콩나물불고기_반조리': 5, '한돈 간장 콩나물불고기_반조리': 5,
        '한돈 간장불고기_반조리': 4, '오리 주물럭_반조리': 5,
        '한돈 된장불고기_반조리': 5, '한돈 간장불고기_쿠킹박스': 4,
        '한돈 매콤 제육볶음_쿠킹박스': 5, '한돈 풋마늘 두루치기_반조리': 5,
        
        # 조림류 (3-5분)
        '메추리알 간장조림': 5, '소고기 장조림 - 180g': 5, '두부조림': 4,
        '알감자조림': 4, '케찹두부조림': 4, '매콤 닭가슴살 장조림': 5,
        '메추리알 간장조림_대용량': 5, '깻잎조림_대용량': 3, '소고기 장조림_대용량': 5,
        '한입 두부간장조림': 4, '검은콩조림': 5, '한입 두부간장조림 - 110g': 4,
        '표고버섯조림': 5, '케찹두부조림 - 120g': 4, '계란 간장조림': 4,
        '명란 장조림': 3, '국내산 땅콩조림': 5, '깻잎조림': 3,
        '간장 감자조림': 5, '마늘쫑 간장조림': 3, '메추리알 간장조림 - 110g': 5,
        '한우 장조림': 5, '우엉조림 - 100g': 5, '유자견과류조림': 4,
        '한돈 매콤 안심장조림': 5, '촉촉 간장무조림': 5, '미니새송이버섯조림': 4,
        '간장 코다리조림': 5, '매콤 코다리조림': 5, '고등어무조림': 5,
        
        # 찜류 (5-8분)
        '꽈리고추찜': 5, '야채 계란찜': 5, '계란찜': 5, '매운돼지갈비찜': 8,
        '순두부 계란찜': 5, '안동찜닭_반조리': 8,
        
        # 전류 (3-5분)
        '소고기육전과 파채': 5, '참치깻잎전': 5, '냉이전 - 140g': 4,
        '매생이전': 4, '동태전': 5, '달콤 옥수수전 - 140g': 4,
        '반달 계란전': 4, '매콤김치전': 5,
        
        # 구이류 (3-5분)
        '간편화덕 고등어 순살구이': 4, '간편화덕 삼치 순살구이': 4,
        '간편화덕 연어 순살구이': 5, '한돈 너비아니(냉동)': 4,
        '오븐치킨_반조리(냉동)': 5, '한돈등심 치즈가스_반조리(냉동)': 4,
        '통등심 수제돈가스_반조리(냉동)': 4,
        
        # 밥/주먹밥류 (1-3분)
        '한돈 주먹밥': 3, '계란 두부소보로 주먹밥': 3, '멸치 주먹밥': 3,
        '참치마요 주먹밥': 3, '한우 주먹밥': 3, '햇반 발아현미밥': 2, '햇반 백미': 2,
        
        # 덮밥류 (1-3분)
        '한돈 토마토 덮밥': 3, '아이들 두부덮밥': 3, '사색 소보로 덮밥': 3,
        
        # 볶음밥 재료 (3-5분)
        '새우 볶음밥 재료': 4, '닭갈비 볶음밥 재료': 4, '냉이 새우볶음밥 재료': 4,
        '상하농원 소세지 볶음밥 재료': 4, '감자볶음밥 재료': 4, '한돈 불고기볶음밥 재료': 4,
        
        # 비빔밥류 (1-3분)
        '꼬막비빔밥': 3,
        
        # 떡볶이류 (3-5분)
        '궁중 떡볶이_반조리 - 520g': 5, '우리쌀로 만든 기름떡볶이_반조리': 4,
        
        # 불고기/전골류 (5-8분)
        '뚝배기 불고기_반조리': 7, '서울식 불고기버섯전골_반조리': 8,
        '한우 파육개장(냉동)': 8, '소불고기_반조리 - 400g': 7,
        '한우 소불고기_반조리': 8, '모둠버섯 불고기_반조리': 6,
        
        # 계란말이 (3-5분)
        '계란말이': 3, '야채계란말이': 3,
        
        # 장류/소스 (1분)
        '달래장': 1, '맛쌈장': 1, '양배추와 맛쌈장': 1, '사랑담은 돈가스소스': 1,
        
        # 기타 특수 요리 (3분)
        '옥수수 버무리': 3, '상하농원 햄 메추리알 케찹볶음': 3, '무나물': 3,
        '수제비_요리놀이터': 3, '봄나물 샐러드': 3, '황태 보푸리': 3,
        '가지강정_대용량': 3, '가지강정': 3, '낙지젓': 3, '영양과채사라다': 3,
        '시래기 된장지짐': 3, '잡채 - 450g': 3, '해물잡채': 3,
        '바른 간장참치 - 130g': 3, '골뱅이무침_반조리': 3, '참깨소스 버섯무침': 3,
        '한우 계란소보로': 3, '꼬마김밥_요리놀이터': 3, '요리놀이터 꼬꼬마 김발': 3,
        '오징어젓': 3, '황기 닭곰탕(냉동)': 3, '불고기 잡채': 3,
        '우엉잡채 - 80g': 3, '만두속재료_요리놀이터': 3,
    }

# 2-2. 특정 반찬의 총 조리시간 계산 함수
def get_cooking_time(dish_name: str, quantity: int = 1) -> float:
    cooking_times = get_dish_cooking_times()
    
    if dish_name in cooking_times:
        base_time = cooking_times[dish_name]
    else:
        base_time = DEFAULT_UNKNOWN_COOKING_TIME # 기본 조리시간이 없는 경우
        print(f"⚠️ '{dish_name}' 조리시간을 찾을 수 없어 기본값 {base_time}분 사용")
    
    # 수량 비례 시간 추가
    total_time = base_time + (quantity * UNIT_TIME_PER_QUANTITY)
    
    return total_time

# 2-3. 조리시간 DataFrame 생성 함수
def create_cooking_time_dataframe() -> pd.DataFrame:
    """조리시간 데이터를 DataFrame으로 변환"""
    cooking_times = get_dish_cooking_times()
    
    df = pd.DataFrame([
        {'반찬명': dish, '기본조리시간(분)': time}
        for dish, time in cooking_times.items()
    ])
    
    return df.sort_values('기본조리시간(분)')




# 3. VRP 최적화 함수
# =====================================================================
# 3-1. VRP 최적화 함수
def solve_dish_production_vrp(embedding_result: Dict[str, Any],
                             changeover_matrix: pd.DataFrame,
                             orders_df: pd.DataFrame,
                             num_lines: int = DEFAULT_NUM_LINES,
                             max_time: int = DEFAULT_MAX_TIME) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    
    # 주문된 반찬별 총 수량 계산
    dish_demands = orders_df.groupby('상품명')['수량'].sum().to_dict()
    ordered_dishes = list(dish_demands.keys())
    num_dishes = len(ordered_dishes)
    
    print(f"주문된 반찬: {num_dishes}개")
    print(f"총 생산량: {sum(dish_demands.values())}개")
    
    # 각 반찬의 조리시간 계산
    cooking_times = {}
    for dish in ordered_dishes:
        quantity = dish_demands[dish]
        cooking_times[dish] = get_cooking_time(dish, quantity)
    
    print(f"조리 시간 범위: {min(cooking_times.values()):.1f}분 ~ {max(cooking_times.values()):.1f}분")
    
    # 거리 매트릭스 생성
    num_depots = num_lines
    num_nodes = num_depots + num_dishes
    
    # 거리 매트릭스 초기화
    distance_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    
    # depot에서 반찬으로 가는 거리 = 0 (시작 비용 없음)
    for depot in range(num_depots):
        for dish_idx in range(num_dishes):
            node_idx = num_depots + dish_idx
            distance_matrix[depot][node_idx] = 0
    
    # 반찬에서 depot으로 돌아가는 거리 = 0 (종료 비용 없음)
    for dish_idx in range(num_dishes):
        node_idx = num_depots + dish_idx
        for depot in range(num_depots):
            distance_matrix[node_idx][depot] = 0
    
    # 반찬 간 전환시간 설정
    for i in range(num_dishes):
        for j in range(num_dishes):
            dish_i = ordered_dishes[i]
            dish_j = ordered_dishes[j]
            
            if dish_i in changeover_matrix.index and dish_j in changeover_matrix.columns:
                changeover_time = int(changeover_matrix.loc[dish_i, dish_j])
            else:
                changeover_time = DEFAULT_UNKNOWN_COOKING_TIME
                
            node_i = num_depots + i
            node_j = num_depots + j
            distance_matrix[node_i][node_j] = changeover_time
    
    # VRP 모델 생성
    depot_starts = list(range(num_lines))
    depot_ends = list(range(num_lines))
    
    manager = pywrapcp.RoutingIndexManager(
        num_nodes, num_lines, depot_starts, depot_ends
    )
    
    routing = pywrapcp.RoutingModel(manager)
    
    # 거리 콜백 함수
    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # 시간 제약 설정
    def time_callback(from_index: int) -> int:
        # 각 노드에서의 시간 소모량
        from_node = manager.IndexToNode(from_index)
        
        # depot이면 시간 소모 없음
        if from_node < num_depots:
            return 0
            
        # 반찬이면 조리시간 소모
        dish_idx = from_node - num_depots
        dish_name = ordered_dishes[dish_idx]
        return int(cooking_times[dish_name])
    
    time_callback_index = routing.RegisterUnaryTransitCallback(time_callback)
    
    # 각 라인별 시간 제약 추가
    routing.AddDimensionWithVehicleCapacity(
        time_callback_index,
        0,  # slack
        [max_time] * num_lines,  # 각 라인의 최대 시간
        True,  # start cumul을 0으로 고정
        'Time'
    )
    
    # 모든 반찬이 정확히 한 번씩 방문되도록 제약
    for dish_idx in range(num_dishes):
        node_idx = num_depots + dish_idx
        routing.AddDisjunction([manager.NodeToIndex(node_idx)], 1000000)
    
    # 목적함수 설정 (Makespan 최소화)
    time_dimension = routing.GetDimensionOrDie('Time')
    end_time_vars = [time_dimension.CumulVar(routing.End(line)) for line in range(num_lines)]
    
    # 최대 완료시간을 최소화
    max_end_time = routing.AddVariableMinimizedByFinalizer(
        routing.solver().Max(end_time_vars)
    )
    
    # 개별 라인 완료시간도 최적화 대상에 포함
    for var in end_time_vars:
        routing.AddVariableMinimizedByFinalizer(var)
    
    # 솔버 설정
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(OPTIMIZATION_TIME_LIMIT)
    
    # 최적화 실행
    solution = routing.SolveWithParameters(search_parameters)
    
    # 결과 출력
    if solution:
        print_solution(manager, routing, solution, ordered_dishes, cooking_times, num_depots)
        return manager, routing, solution
    else:
        print("해를 찾을 수 없습니다!")
        return None, None, None

# 3-2. 최적화 결과 출력 함수
def print_solution(manager: Any, routing: Any, solution: Any,
                  ordered_dishes: List[str], cooking_times: Dict[str, float],
                  num_depots: int) -> None:
    
    print("\n" + "="*50)
    print("최적화 결과")
    print("="*50)
    
    max_line_time = 0
    
    for line_id in range(routing.vehicles()):
        index = routing.Start(line_id)
        plan_output = f'생산라인 {line_id + 1}: '
        route_time = 0
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            
            # 반찬 노드인 경우
            if node >= num_depots:
                dish_idx = node - num_depots
                dish_name = ordered_dishes[dish_idx]
                cooking_time = cooking_times[dish_name]
                
                plan_output += f'{dish_name}({cooking_time:.1f}분) -> '
                route_time += cooking_time
                
                # 다음 노드로의 전환시간 추가
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                if not routing.IsEnd(index):
                    route_time += routing.GetArcCostForVehicle(previous_index, index, line_id)
            else:
                index = solution.Value(routing.NextVar(index))
        
        plan_output += '완료'
        print(f'{plan_output}')
        print(f'⏱️  총 소요시간: {route_time:.1f}분')
        print('-' * 50)
        
        max_line_time = max(max_line_time, route_time)
    
    print(f"\n 전체 완료 시간 (Makespan): {max_line_time:.1f}분")
    print(f" 제한 시간 대비: {max_line_time/DEFAULT_MAX_TIME*100:.1f}%")
    
    if max_line_time <= DEFAULT_MAX_TIME:
        print("시간 제약 만족!")
    else:
        print("시간 제약 초과!")



# 4. 벡터 데이터베이스 저장 함수
# =====================================================================
# 4-1. 최적화 결과를 벡터 데이터베이스에 저장하는 함수
def save_optimization_to_vectordb(manager: Any, routing: Any, solution: Any,
                                 ordered_dishes: List[str], cooking_times: Dict[str, float],
                                 num_depots: int, file_name: Optional[str] = None) -> None:
    try:
        # 임베딩 모델 로드
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        
        # 벡터 스토어 생성/로드
        vector_store = Chroma(
            collection_name="optimization_results",
            persist_directory=OPTIMIZATION_DB_PATH,
            embedding_function=embeddings
        )
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        documents = []
        max_line_time = 0
        line_info = []
        
        # 라인별 데이터 추출
        for line_id in range(routing.vehicles()):
            index = routing.Start(line_id)
            route_time = 0
            line_dishes = []
            line_times = []
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                
                if node >= num_depots:
                    dish_idx = node - num_depots
                    dish_name = ordered_dishes[dish_idx]
                    cooking_time = cooking_times[dish_name]
                    
                    line_dishes.append(dish_name)
                    line_times.append(round(cooking_time, 1))
                    route_time += cooking_time
                    
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    if not routing.IsEnd(index):
                        route_time += routing.GetArcCostForVehicle(previous_index, index, line_id)
                else:
                    index = solution.Value(routing.NextVar(index))
            
            if route_time > 0:  # 작업이 있는 라인만
                line_info.append({
                    'line_id': line_id + 1,
                    'dishes': line_dishes,
                    'times': line_times,
                    'total_time': round(route_time, 1)
                })
                max_line_time = max(max_line_time, route_time)
        
        # 저장된 정보 1 : 전체 요약 문서
        summary_text = f"""
                        최적화 실행 시간: {timestamp}
                        전체 완료시간: {max_line_time:.1f}분
                        제한시간 대비: {max_line_time/DEFAULT_MAX_TIME*100:.1f}%
                        제약조건 만족: {'예' if max_line_time <= DEFAULT_MAX_TIME else '아니오'}
                        총 반찬 종류: {len(ordered_dishes)}개
                        활성 생산라인: {len(line_info)}개
                        입력파일: {file_name or '알 수 없음'}
                        """.strip()
        
        documents.append(Document(
            page_content=summary_text,
            metadata={
                "type": "summary",
                "timestamp": timestamp,
                "makespan": max_line_time,
                "file_name": file_name or "unknown"
            }
        ))
        
        # 저장된 정보 2 : 라인별 상세 정보
        for line in line_info:
            schedule_text = " → ".join(line['dishes'])
            times_text = " → ".join([f"{t}분" for t in line['times']])
            
            line_text = f"""
                        라인{line['line_id']} 생산스케줄:
                        생산순서: {schedule_text}
                        조리시간: {times_text}
                        총 작업시간: {line['total_time']}분
                        효율성: {line['total_time']/DEFAULT_MAX_TIME*100:.1f}%
                        """.strip()
            
            documents.append(Document(
                page_content=line_text,
                metadata={
                    "type": "line_detail",
                    "line_number": line['line_id'],
                    "total_time": line['total_time'],
                    "timestamp": timestamp
                }
            ))
        
        # 저장된 정보 3 : 개별 반찬 정보
        dish_texts = []
        for line in line_info:
            for dish, time in zip(line['dishes'], line['times']):
                dish_texts.append(f"{dish} 반찬이 라인{line['line_id']}에서 {time}분간 생산됨")
        
        if dish_texts:
            documents.append(Document(
                page_content=". ".join(dish_texts),
                metadata={
                    "type": "dish_details",
                    "timestamp": timestamp,
                    "total_dishes": len(dish_texts)
                }
            ))
        
        # 벡터 DB에 저장
        vector_store.add_documents(documents)
        print(f"최적화 결과 벡터 DB 저장 완료 ({len(documents)}개 문서)")
        
    except Exception as e:
        print(f"벡터 DB 저장 중 오류: {str(e)}")



# 5. 메인 실행 함수들
# =====================================================================
# 5-1. 생산 최적화 실행 함수
def run_vrp_optimization(embedding_result: Dict[str, Any],
                        changeover_matrix: pd.DataFrame,
                        orders_df: pd.DataFrame,
                        num_lines: int = DEFAULT_NUM_LINES,
                        max_time: int = DEFAULT_MAX_TIME) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    
    print("생산 최적화를 시작합니다!")
    
    return solve_dish_production_vrp(
        embedding_result=embedding_result,
        changeover_matrix=changeover_matrix,
        orders_df=orders_df,
        num_lines=num_lines,
        max_time=max_time
    )

# 5-2. 전체 최적화 프로세스 실행 함수
def run_full_optimization(file_path: str,
                         dish_column: str = '상품명',
                         num_lines: int = DEFAULT_NUM_LINES,
                         max_time: int = DEFAULT_MAX_TIME) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:

    global current_file_name
    current_file_name = os.path.basename(file_path)
    
    # 데이터 로드
    df = pd.read_excel(file_path)
    
    # 임베딩 생성
    embedding_result = create_dish_embeddings(df, dish_column)
    
    # 전환시간 매트릭스 계산
    changeover_df = calculate_changeover_matrix(embedding_result)
    
    # VRP 최적화 실행
    manager, routing, solution = run_vrp_optimization(
        embedding_result, changeover_df, df, num_lines, max_time
    )
    
    # 벡터 DB 저장
    if solution:
        # 주문된 반찬과 조리시간 계산
        dish_demands = df.groupby(dish_column)['수량'].sum().to_dict()
        ordered_dishes = list(dish_demands.keys())
        cooking_times = {dish: get_cooking_time(dish, dish_demands[dish]) 
                        for dish in ordered_dishes}
        
        save_optimization_to_vectordb(
            manager, routing, solution, ordered_dishes, cooking_times, 
            DEFAULT_NUM_LINES, current_file_name
        )
    
    return manager, routing, solution



# 6. 생산 최적화 도구
# =====================================================================
# 6-1. 반찬 최적화 도구 함수 : Agent로 전달되는 함수
def dish_optimization_tool(query: str) -> str:

    global last_optimization_output, last_optimization_text

    try:
        # 파일 경로 추출
        file_path = query.split(',')[0].strip()
        
        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            return f"파일을 찾을 수 없습니다: {file_path}"
        
        print(f"반찬 최적화 시작: {file_path}")
        
        # stdout 캡처 시작
        captured_output = StdoutCapture()
        old_stdout = sys.stdout
        
        try:
            # stdout을 캡처로 리다이렉트
            sys.stdout = captured_output
            
            # 최적화 실행
            optimization_result = run_full_optimization(file_path)
            
        finally:
            # stdout 복원
            sys.stdout = old_stdout
        
        # 캡처된 출력 가져오기
        captured_text = captured_output.get_output()
        
        # 전역 변수에 결과 저장
        last_optimization_output = optimization_result
        last_optimization_text = captured_text
        
        # 캡처된 출력을 실제로 출력
        print(captured_text)

        return "반찬 생산 최적화 완료! 위에 상세한 결과가 출력되었으며 벡터 DB에도 저장되었습니다."
        
    except Exception as e:
        return f"최적화 중 오류 발생: {str(e)}"



# 테스트 실행부
# =====================================================================
if __name__ == "__main__":
    test_file = "/Users/jibaekjang/VS-Code/AI_Agent/product_data_2022_04_01.xlsx"
    
    if os.path.exists(test_file):
        run_full_optimization(test_file)
    else:
        print(f"테스트 파일을 찾을 수 없습니다: {test_file}")
        print("파일 경로를 확인해주세요.")