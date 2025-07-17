# ============================================================================
# visualization_tool.py - 최적화 결과 시각화 전용 모듈
# ============================================================================

import re
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
#import Tool1_opti_vrp  # VRP 모듈의 전역 변수 접근을 위해 import
import Tool1_opti_vrp

# ============================================================================
# Tool 3 : plotly 기반 시각화
# ============================================================================
def optimization_visualizer_tool(query: str) -> str:
    """최적화 결과 시각화 도구 (Agent에서 이동)"""
    
    try:
        # 최적화 결과가 있는지 확인
        if Tool1_opti_vrp.last_optimization_text is None:
            return "❌ 먼저 반찬 생산 최적화를 실행해주세요."
        
        # 쿼리가 문자열인지 확인하고 안전하게 처리
        if not isinstance(query, str):
            query = str(query)
        
        # 요청에 따른 차트 타입 결정
        query_lower = query.lower()
        
        if "간트" in query_lower or "타임라인" in query_lower:
            chart_type = "gantt"
        elif "효율" in query_lower or "분석" in query_lower:
            chart_type = "efficiency"  
        elif "병목" in query_lower or "히트맵" in query_lower:
            chart_type = "bottleneck"
        elif "리포트" in query_lower or "보고서" in query_lower:
            chart_type = "summary"
        elif "전체" in query_lower or "모든" in query_lower:
            chart_type = "all"
        else:
            chart_type = "gantt"  # 기본값
        
        print(f"📊 시각화 타입: {chart_type}")
        
        # 시각화 실행 (전역 변수에서 캡처된 텍스트 사용)
        try:
            result = visualize_optimization_result(
                Tool1_opti_vrp.last_optimization_text,  # VRP 모듈의 전역 변수 사용
                chart_type
            )
            return result
        except Exception as viz_error:
            return f"❌ 시각화 도구 실행 중 오류: {str(viz_error)}\n💡 visualization_tool.py의 visualize_optimization_result 함수를 확인해주세요."
        
    except Exception as e:
        return f"❌ 시각화 중 오류 발생: {str(e)}\n📋 오류 타입: {type(e).__name__}"

def parse_optimization_result(optimization_text):
    """
    최적화 결과 텍스트를 파싱하여 구조화된 데이터로 변환
    
    Parameters:
    -----------
    optimization_text : str
        최적화 결과 텍스트
    
    Returns:
    --------
    dict: 파싱된 생산라인 데이터
    """
    
    print(f"📋 파싱할 텍스트 길이: {len(optimization_text)}")
    print(f"📋 텍스트 미리보기:\n{optimization_text[:500]}...")
    
    lines_data = []
    
    # 각 생산라인 데이터 추출 - 더 유연한 패턴 사용
    line_pattern = r'생산라인 (\d+): (.*?) -> 완료.*?총 소요시간: ([\d.]+)분'
    matches = re.findall(line_pattern, optimization_text, re.DOTALL)
    
    print(f"🔍 발견된 생산라인 수: {len(matches)}")
    
    for match in matches:
        line_id = int(match[0])
        dishes_text = match[1]
        total_time = float(match[2])
        
        print(f"📊 라인 {line_id}: {total_time:.1f}분, 반찬: {dishes_text[:50]}...")
        
        # 각 반찬과 시간 추출 - 더 정확한 패턴
        dish_pattern = r'([^(]+?)\(([\d.]+)분\)'
        dish_matches = re.findall(dish_pattern, dishes_text)
        
        dishes = []
        current_time = 0
        
        for dish_name, dish_time in dish_matches:
            dish_name = dish_name.strip()
            dish_time = float(dish_time)
            
            dishes.append({
                'name': dish_name,
                'time': dish_time,
                'start': current_time,
                'end': current_time + dish_time
            })
            current_time += dish_time
            
        # 디버깅 정보
        print(f"   └─ {len(dishes)}개 반찬 파싱됨")
        
        lines_data.append({
            'line_id': line_id,
            'total_time': total_time,
            'dishes': dishes,
            'dish_count': len(dishes)
        })
    
    # Makespan 추출 - 더 유연한 패턴
    makespan_patterns = [
        r'전체 완료 시간 \(Makespan\): ([\d.]+)분',
        r'🏆 전체 완료 시간.*?: ([\d.]+)분',
        r'Makespan.*?: ([\d.]+)분'
    ]
    
    makespan = 0
    for pattern in makespan_patterns:
        makespan_match = re.search(pattern, optimization_text)
        if makespan_match:
            makespan = float(makespan_match.group(1))
            break
    
    print(f"🏆 Makespan: {makespan:.1f}분")
    
    result = {
        'lines': lines_data,
        'makespan': makespan,
        'summary': {
            'total_lines': len(lines_data),
            'avg_time': sum(line['total_time'] for line in lines_data) / len(lines_data) if lines_data else 0,
            'total_dishes': sum(line['dish_count'] for line in lines_data)
        }
    }
    
    print(f"📈 파싱 결과: {result['summary']['total_lines']}개 라인, {result['summary']['total_dishes']}개 반찬")
    
    return result

def create_gantt_chart(parsed_data):
    """간트차트 생성 - 호버 템플릿 완전 수정"""
    
    if not parsed_data['lines']:
        print("❌ 생성할 라인 데이터가 없습니다.")
        return None
    
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#A8E6CF', '#FFD93D']
    
    for i, line_data in enumerate(parsed_data['lines']):
        line_id = line_data['line_id']
        
        if not line_data['dishes']:  # 반찬이 없는 경우 건너뛰기
            continue
            
        for dish_idx, dish in enumerate(line_data['dishes']):
            # 호버 텍스트를 미리 포맷팅
            hover_text = f"""<b>{dish['name']}</b><br>조리시간: {dish['time']:.1f}분<br>시작시간: {dish['start']:.1f}분<br>종료시간: {dish['end']:.1f}분<br>라인: {line_id}"""
            
            # 각 반찬마다 개별 trace 생성
            fig.add_trace(go.Scatter(
                x=[dish['start'], dish['end']],
                y=[line_id, line_id],
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=12),
                name=f"라인 {line_id}" if dish_idx == 0 else "",  # 첫 번째만 범례 표시
                showlegend=(dish_idx == 0),  # 첫 번째만 범례 표시
                hoverinfo='text',  # text 모드 사용
                hovertext=hover_text,  # 미리 포맷된 텍스트 사용
                hoverlabel=dict(
                    bgcolor="white",
                    bordercolor=colors[i % len(colors)],
                    font_size=12,
                    font_family="Arial"
                )
            ))
            
            # 반찬 이름을 라벨로 추가 (선택사항)
            mid_point = (dish['start'] + dish['end']) / 2
            fig.add_annotation(
                x=mid_point,
                y=line_id,
                text=dish['name'][:8] + "..." if len(dish['name']) > 8 else dish['name'],
                showarrow=False,
                font=dict(size=9, color="white"),
                bgcolor=colors[i % len(colors)],
                bordercolor="white",
                borderwidth=1,
                opacity=0.8
            )
    
    fig.update_layout(
        title={
            'text': '🍱 반찬 생산라인 간트차트',
            'x': 0.5,
            'font': {'size': 20}
        },
        xaxis_title="시간 (분)",
        yaxis_title="생산라인",
        yaxis=dict(
            tickmode='array',
            tickvals=[line['line_id'] for line in parsed_data['lines']],
            ticktext=[f"라인 {line['line_id']}" for line in parsed_data['lines']],
            autorange='reversed'  # 위에서부터 1,2,3 순서로
        ),
        height=600,
        width=1200,
        template='plotly_white',
        showlegend=True,
        hovermode='closest'  # 가장 가까운 포인트의 호버 정보 표시
    )
    
    return fig

def create_efficiency_chart(parsed_data):
    """효율성 분석 차트 생성"""
    
    if not parsed_data['lines']:
        return None
    
    line_ids = [line['line_id'] for line in parsed_data['lines']]
    total_times = [line['total_time'] for line in parsed_data['lines']]
    dish_counts = [line['dish_count'] for line in parsed_data['lines']]
    
    # 효율성 계산 (makespan 대비)
    if parsed_data['makespan'] > 0:
        efficiencies = [(parsed_data['makespan'] - time) / parsed_data['makespan'] * 100 for time in total_times]
    else:
        efficiencies = [0] * len(total_times)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('라인별 소요시간', '라인별 반찬 수', '라인별 효율성', '시간 분포'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "pie"}]]
    )
    
    # 소요시간 막대그래프
    fig.add_trace(
        go.Bar(
            x=[f"라인{id}" for id in line_ids],
            y=total_times,
            name="소요시간",
            marker_color='lightblue',
            text=[f"{time:.1f}분" for time in total_times],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # 반찬 수 막대그래프
    fig.add_trace(
        go.Bar(
            x=[f"라인{id}" for id in line_ids],
            y=dish_counts,
            name="반찬 수",
            marker_color='lightgreen',
            text=[f"{count}개" for count in dish_counts],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # 효율성 막대그래프
    fig.add_trace(
        go.Bar(
            x=[f"라인{id}" for id in line_ids],
            y=efficiencies,
            name="효율성",
            marker_color='orange',
            text=[f"{eff:.1f}%" for eff in efficiencies],
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # 시간 분포 파이차트
    fig.add_trace(
        go.Pie(
            labels=[f"라인{id}" for id in line_ids],
            values=total_times,
            name="시간분포"
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title={
            'text': '📊 생산라인 효율성 분석 대시보드',
            'x': 0.5,
            'font': {'size': 20}
        },
        height=800,
        width=1200,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def create_bottleneck_analysis(parsed_data):
    """병목 구간 분석 히트맵"""
    
    if not parsed_data['lines'] or parsed_data['makespan'] <= 0:
        return None
    
    # 시간을 10분 단위로 나누어 분석
    max_time = int(parsed_data['makespan']) + 10
    time_slots = list(range(0, max_time, 10))
    
    heatmap_data = []
    
    for time_slot in time_slots:
        slot_data = []
        for line_data in parsed_data['lines']:
            # 해당 시간대에 작업 중인 반찬 수 계산
            active_dishes = 0
            for dish in line_data['dishes']:
                if dish['start'] <= time_slot < dish['end']:
                    active_dishes = 1
                    break
            slot_data.append(active_dishes)
        heatmap_data.append(slot_data)
    
    heatmap_array = np.array(heatmap_data).T
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_array,
        x=[f"{t}분" for t in time_slots],
        y=[f"라인{line['line_id']}" for line in parsed_data['lines']],
        colorscale='RdYlBu_r',
        showscale=True,
        colorbar=dict(title="작업 상태")
    ))
    
    fig.update_layout(
        title={
            'text': '🔥 시간대별 생산라인 가동 현황',
            'x': 0.5,
            'font': {'size': 20}
        },
        xaxis_title="시간대",
        yaxis_title="생산라인",
        height=500,
        width=1000,
        template='plotly_white'
    )
    
    return fig

def create_summary_report(parsed_data):
    """종합 분석 리포트 생성"""
    
    if not parsed_data['lines']:
        return "❌ 분석할 데이터가 없습니다."
    
    summary = f"""
📊 반찬 생산 최적화 결과 종합 분석
{'='*50}

📈 핵심 지표:
• 총 생산라인: {parsed_data['summary']['total_lines']}개
• 전체 반찬 수: {parsed_data['summary']['total_dishes']}개  
• 최대 소요시간: {parsed_data['makespan']:.1f}분
• 평균 소요시간: {parsed_data['summary']['avg_time']:.1f}분

📋 라인별 상세 정보:
"""
    
    for line in parsed_data['lines']:
        if parsed_data['makespan'] > 0:
            efficiency = (parsed_data['makespan'] - line['total_time']) / parsed_data['makespan'] * 100
        else:
            efficiency = 0
        summary += f"• 라인 {line['line_id']}: {line['total_time']:.1f}분 ({line['dish_count']}개 반찬, 효율성 {efficiency:.1f}%)\n"
    
    # 병목 분석
    if len(parsed_data['lines']) > 1:
        max_line = max(parsed_data['lines'], key=lambda x: x['total_time'])
        min_line = min(parsed_data['lines'], key=lambda x: x['total_time'])
        time_diff = max_line['total_time'] - min_line['total_time']
        
        summary += f"""
🔍 병목 분석:
• 가장 느린 라인: 라인 {max_line['line_id']} ({max_line['total_time']:.1f}분)
• 가장 빠른 라인: 라인 {min_line['line_id']} ({min_line['total_time']:.1f}분)  
• 라인간 편차: {time_diff:.1f}분 ({time_diff/parsed_data['summary']['avg_time']*100:.1f}%)

💡 개선 제안:
"""
        
        if time_diff > 50:
            summary += "• 라인간 편차가 큽니다. 반찬 재배치를 고려해보세요.\n"
        if parsed_data['makespan'] > 480:
            summary += "• 전체 시간이 8시간을 초과합니다. 추가 라인 검토가 필요합니다.\n"
        
        avg_efficiency = sum((parsed_data['makespan'] - line['total_time']) / parsed_data['makespan'] * 100 for line in parsed_data['lines']) / len(parsed_data['lines'])
        if avg_efficiency < 70:
            summary += "• 평균 효율성이 낮습니다. 최적화 파라미터 조정을 권장합니다.\n"
    
    return summary

# Plotly 렌더링 설정 추가
import plotly.io as pio

def setup_plotly_renderer():
    """Plotly 렌더링 환경 설정"""
    try:
        # 사용 가능한 렌더러 확인
        available_renderers = pio.renderers
        print(f"🔧 사용 가능한 렌더러: {list(available_renderers.keys())}")
        
        # 우선순위에 따라 렌더러 설정
        if 'browser' in available_renderers:
            pio.renderers.default = 'browser'
            print("✅ 브라우저 렌더러 설정 완료")
        elif 'plotly_mimetype' in available_renderers:
            pio.renderers.default = 'plotly_mimetype'
            print("✅ plotly_mimetype 렌더러 설정 완료")
        else:
            # 대체 렌더러 설정
            pio.renderers.default = 'png'
            print("✅ PNG 렌더러 설정 완료 (대체)")
            
    except Exception as e:
        print(f"⚠️ 렌더러 설정 실패: {e}")
        # 기본 설정 유지

def safe_show_figure(fig, chart_name="차트"):
    """안전한 차트 표시 함수"""
    try:
        # 첫 번째 시도: 기본 show()
        fig.show()
        return True, f"✅ {chart_name} 표시 완료"
        
    except Exception as e:
        print(f"⚠️ 기본 렌더링 실패: {e}")
        
        try:
            # 두 번째 시도: 브라우저 렌더러
            fig.show(renderer='browser')
            return True, f"✅ {chart_name} 표시 완료 (브라우저)"
            
        except Exception as e2:
            print(f"⚠️ 브라우저 렌더링 실패: {e2}")
            
            try:
                # 세 번째 시도: HTML 파일로 저장
                import tempfile
                import webbrowser
                import os
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                    fig.write_html(f.name)
                    html_path = f.name
                
                # 브라우저에서 열기
                webbrowser.open('file://' + html_path)
                return True, f"✅ {chart_name} HTML 파일로 저장 및 표시 완료"
                
            except Exception as e3:
                print(f"⚠️ HTML 저장 실패: {e3}")
                
                try:
                    # 네 번째 시도: 이미지로 저장
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as f:
                        fig.write_image(f.name)
                        img_path = f.name
                    
                    print(f"📁 {chart_name} 이미지 저장: {img_path}")
                    return True, f"✅ {chart_name} 이미지로 저장 완료: {img_path}"
                    
                except Exception as e4:
                    return False, f"❌ {chart_name} 표시 실패: 모든 렌더링 방법 실패"

def visualize_optimization_result(optimization_text, chart_type="all"):
    """
    최적화 결과 시각화 메인 함수
    
    Parameters:
    -----------
    optimization_text : str
        최적화 결과 텍스트
    chart_type : str
        생성할 차트 타입 ("gantt", "efficiency", "bottleneck", "summary", "all")
        
    Returns:
    --------
    str : 시각화 완료 메시지
    """
    
    try:
        print("📊 시각화 시작...")
        
        # Plotly 렌더러 설정
        setup_plotly_renderer()
        
        print(f"📋 입력 텍스트 타입: {type(optimization_text)}")
        
        # 입력 타입 확인 및 변환
        if not isinstance(optimization_text, str):
            return "❌ 입력 데이터가 문자열이 아닙니다."
        
        if len(optimization_text) < 50:
            return "❌ 최적화 결과 텍스트가 너무 짧습니다. 올바른 최적화 결과인지 확인해주세요."
        
        # 결과 파싱
        print("🔍 결과 파싱 중...")
        parsed_data = parse_optimization_result(optimization_text)
        
        if not parsed_data['lines']:
            return "❌ 최적화 결과를 파싱할 수 없습니다. 생산라인 정보가 없습니다."
        
        print(f"✅ 파싱 완료: {len(parsed_data['lines'])}개 라인 발견")
        
        # 차트 타입에 따른 시각화
        if chart_type == "gantt":
            fig = create_gantt_chart(parsed_data)
            if fig:
                success, message = safe_show_figure(fig, "간트차트")
                return message
            else:
                return "❌ 간트차트 생성에 실패했습니다."
                
        elif chart_type == "efficiency":
            fig = create_efficiency_chart(parsed_data)
            if fig:
                success, message = safe_show_figure(fig, "효율성 분석 대시보드")
                return message
            else:
                return "❌ 효율성 차트 생성에 실패했습니다."
                
        elif chart_type == "bottleneck":
            fig = create_bottleneck_analysis(parsed_data)
            if fig:
                success, message = safe_show_figure(fig, "병목 구간 분석 히트맵")
                return message
            else:
                return "❌ 병목 분석 차트 생성에 실패했습니다."
                
        elif chart_type == "summary":
            report = create_summary_report(parsed_data)
            print(report)
            return "📋 종합 분석 리포트가 생성되었습니다!"
            
        else:  # "all" 또는 기타
            # 모든 차트 생성
            results = []
            
            print("📊 간트차트 생성 중...")
            gantt_fig = create_gantt_chart(parsed_data)
            if gantt_fig:
                success, message = safe_show_figure(gantt_fig, "간트차트")
                results.append(("간트차트", success, message))
            
            print("📈 효율성 분석 차트 생성 중...")
            efficiency_fig = create_efficiency_chart(parsed_data)
            if efficiency_fig:
                success, message = safe_show_figure(efficiency_fig, "효율성 분석")
                results.append(("효율성 분석", success, message))
            
            print("🔥 병목 분석 차트 생성 중...")
            bottleneck_fig = create_bottleneck_analysis(parsed_data)
            if bottleneck_fig:
                success, message = safe_show_figure(bottleneck_fig, "병목 분석")
                results.append(("병목 분석", success, message))
            
            # 종합 리포트도 출력
            print("📋 종합 리포트 생성 중...")
            report = create_summary_report(parsed_data)
            print(report)
            results.append(("종합 리포트", True, "✅ 종합 리포트 생성 완료"))
            
            # 결과 요약
            success_count = sum(1 for _, success, _ in results if success)
            result_summary = f"🎨 전체 시각화 완료! ({success_count}/{len(results)} 성공)\n"
            
            for chart_name, success, message in results:
                status = "✅" if success else "❌"
                result_summary += f"{status} {chart_name}\n"
            
            return result_summary
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"❌ 시각화 중 오류 발생: {str(e)}\n🔍 상세 오류:\n{error_detail}"

# ============================================================================
# 직접 실행할 때만 테스트 실행
# ============================================================================

if __name__ == "__main__":
    # 테스트용 최적화 결과 (더 현실적인 데이터)
    test_result = """
🚀 반찬 생산 최적화를 시작합니다!
=== 데이터 준비 중 ===
총 248개의 고유한 반찬 발견
총 생산량: 1500개

==================================================
🎯 최적화 결과
==================================================
생산라인 1: 무생채(2.2분) -> 달래김무침(2.1분) -> 달래장(1.2분) -> 콩나물무침(1.5분) -> 완료
⏱️  총 소요시간: 312.9분
--------------------------------------------------
생산라인 2: 잡채 - 450g(3.1분) -> 궁중 떡볶이_반조리 - 520g(5.1분) -> 한우 소고기 감자국(4.8분) -> 완료  
⏱️  총 소요시간: 296.5분
--------------------------------------------------
생산라인 3: 한돈 매콤 제육볶음_반조리 - 500g(5.2분) -> 메추리알 간장조림(4.9분) -> 완료
⏱️  총 소요시간: 285.1분
--------------------------------------------------
생산라인 4: 계란찜(5.0분) -> 야채계란말이(3.2분) -> 한우 주먹밥(2.8분) -> 완료
⏱️  총 소요시간: 301.4분
--------------------------------------------------

🏆 전체 완료 시간 (Makespan): 312.9분
⏰ 제한 시간 대비: 130.4%
⚠️  시간 제약 초과!
    """
    
    print("=== 시각화 도구 테스트 ===")
    result = visualize_optimization_result(test_result, "all")
    print(result)