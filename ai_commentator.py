import os
import json
import sys
import urllib.request
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))

def get_kst_now():
    return datetime.now(KST)

def get_session_info():
    hour = get_kst_now().hour
    if 5 <= hour < 11:
        return 'morning', '🌅 모닝 마켓 브리핑 (08:30 KST)', '국내 장 개장 대비 및 글로벌 야간 시장(미 국채·뉴욕 증시) 종합 분석'
    elif 11 <= hour < 18:
        return 'midday', '☀️ 미드데이 장중 점검 (13:00 KST)', '국내 증시(코스피·코스닥) 장중 수급 및 아시아 시장 종합 진단'
    else:
        return 'night', '🌙 나이트 / 뉴욕 본장 브리핑 (23:00 KST)', '미국 본장 개장 동향, 글로벌 매크로 지표 및 야간 선물 분석'

def load_screener_data():
    if os.path.exists("screener_results.json"):
        with open("screener_results.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def build_market_prompt(data: dict, session_name: str, session_desc: str) -> str:
    macro = data.get('_macro', {})
    now_str = get_kst_now().strftime("%Y-%m-%d %H:%M KST")
    
    macro_lines = []
    for k, v in macro.items():
        if isinstance(v, dict):
            macro_lines.append(f"- {k.upper()}: {v.get('price')} (변동: {v.get('change')})")
    
    stocks = [v for k, v in data.items() if isinstance(v, dict) and v.get('ticker')]
    under = sorted(stocks, key=lambda x: x.get('total_undervalued_score', 0), reverse=True)[:3]
    over = sorted(stocks, key=lambda x: x.get('total_overvalued_score', 0), reverse=True)[:3]
    
    under_lines = [f"- {s.get('ticker')} ({s.get('company_name')}): 저평가 {s.get('total_undervalued_score')}점" for s in under]
    over_lines = [f"- {s.get('ticker')} ({s.get('company_name')}): 과열 {s.get('total_overvalued_score')}점" for s in over]
    
    return f"""당신은 월가 헤지펀드의 수석 매크로 전략가이자 퀀트 포트폴리오 매니저입니다.
아래의 실시간 거시경제 지표 및 퀀트 랭킹을 바탕으로 [{session_name}] 해설을 작성해주세요.

[기준 시각]: {now_str}
[세션 성격]: {session_desc}

[실시간 거시경제 지표]:
{chr(10).join(macro_lines)}

[퀀트 저평가 TOP 종목]:
{chr(10).join(under_lines)}

[퀀트 과열/고평가 경계 종목]:
{chr(10).join(over_lines)}

반드시 아래 JSON 형식으로만 답변해주세요 (마크다운 백틱 없이 순수 JSON만 반환):
{{
  "headline": "핵심을 관통하는 한 줄 헤드라인 (30자 내외)",
  "summary": "거시경제 환경과 주식시장 영향을 유기적으로 연결한 2~3문장의 깊이 있는 전문 해설",
  "key_points": [
    "🏛️ 국채/통화: 미국채 금리 흐름과 연준 통화정책에 대한 핵심 분석 1문장",
    "🎯 퀀트/섹터: 저평가/과열 종목 및 주도 섹터 수급 특징 분석 1문장",
    "💡 투자 전략: 지금 투자자가 취해야 할 구체적인 포지셔닝 대응 전략 1문장"
  ],
  "macro_insights": {{
    "us10y": "미국채 10년물 실시간 수치에 기반한 시장 영향 해설",
    "sp500": "S&P 500 증시 흐름 및 밸류에이션 해설",
    "kospi": "코스피 외국인 수급 및 지수 지지력 해설",
    "usdkrw": "원/달러 환율 변동과 국내 증시 상관관계 해설",
    "btc": "가상자산 유동성 및 위험자산 선호 심리 해설"
  }}
}}"""

def call_gemini_api(prompt: str, api_key: str) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4}
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method='POST'
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        res_data = json.loads(resp.read().decode('utf-8'))
        return json.loads(res_data['candidates'][0]['content']['parts'][0]['text'].strip())

def get_heuristic_fallback(session_id: str, data: dict) -> dict:
    u10 = data.get('_macro', {}).get('us10y', {}).get('price', 4.93)
    krw = data.get('_macro', {}).get('usdkrw', {}).get('price', 1335)
    return {
        "headline": f"미 국채 10년물 {u10}% 안정화 속 코스피 외인 수급 분기점",
        "summary": f"미국채 10년물 금리가 {u10}%선에서 숨고르기에 들어가며 성장주 밸류에이션 압박이 완화되었습니다. 원/달러 환율({krw:,.0f}원) 안정 여부와 1년 최대낙폭(MDD)에 도달한 저평가 우량주 중심의 반등 탄력에 주목해야 합니다.",
        "key_points": [
            f"🏛️ 국채/통화: 미국채 10년물({u10}%)이 5% 상단을 제어하며 글로벌 할인율 급등 부담을 완화하고 있습니다.",
            "🎯 퀀트/섹터: 기술적 과매도 구간에 진입한 대형 반도체 및 핵심 ETF 종목군으로 저가 매수세가 유입 중입니다.",
            "💡 투자 전략: 변동성(VIX) 추이를 주시하며 실적 펀더멘털 대비 과도하게 하락한 종목 위주로 선별 분할 매수가 유리합니다."
        ],
        "macro_insights": {
            "us10y": f"10년물 금리가 {u10}% 수준을 유지하며 글로벌 성장주 밸류에이션 할인율에 영향을 주고 있습니다.",
            "sp500": "대형 우량주 중심의 실적 지지력을 시험하며 안착을 시도하고 있습니다.",
            "kospi": f"원/달러 환율({krw:,.0f}원) 추이에 따라 외국인 현선물 순매수 방향성이 결정될 전망입니다.",
            "usdkrw": "수출업체 네고 물량 유입과 외환당국 미세조정이 원화 안정의 핵심 변수입니다.",
            "btc": "24시간 글로벌 유동성 및 위험자산 선호 심리의 선행 척도 역할을 하고 있습니다."
        }
    }

def main():
    session_id, session_name, session_desc = get_session_info()
    now_str = get_kst_now().strftime("%Y-%m-%d %H:%M KST")
    print(f"--- 🤖 AI 마켓 인텔리전스 분석 시작 [{session_name}] ---")
    
    data = load_screener_data()
    if not data:
        print("❌ screener_results.json 파일이 없습니다.")
        sys.exit(1)
        
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    ai_content = None
    
    if api_key:
        try:
            print("🧠 Gemini AI에 심층 분석 요청 중...")
            prompt = build_market_prompt(data, session_name, session_desc)
            ai_content = call_gemini_api(prompt, api_key)
            print("✓ Gemini AI 심층 시황 분석 생성 완료!")
        except Exception as e:
            print(f"⚠️ Gemini API 호출 실패 ({e}). 백업 엔진으로 전환합니다.")
            
    if not ai_content:
        ai_content = get_heuristic_fallback(session_id, data)
        
    data['_ai_commentary'] = {
        "session": session_id,
        "session_name": session_name,
        "updated_at": now_str,
        "headline": ai_content.get("headline", ""),
        "summary": ai_content.get("summary", ""),
        "key_points": ai_content.get("key_points", []),
        "macro_insights": ai_content.get("macro_insights", {})
    }
    
    if '_macro' in data and isinstance(data['_macro'], dict):
        for m_key, m_ins in ai_content.get("macro_insights", {}).items():
            if m_key in data['_macro'] and isinstance(data['_macro'][m_key], dict):
                data['_macro'][m_key]['insight'] = m_ins
                
    with open("screener_results.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"🎉 screener_results.json 갱신 완료! ({ai_content.get('headline')})")

if __name__ == "__main__":
    main()
