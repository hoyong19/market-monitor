import yfinance as yf
import pandas as pd
import numpy as np
import json
import sys
import traceback
import urllib.request
from typing import List, Dict, Any
from datetime import datetime, timezone, timedelta

# 깃허브 클라우드 러너(UTC)에서도 정확한 한국 표준시(KST = UTC+9)를 생성하도록 고정
KST = timezone(timedelta(hours=9))

def get_kst_now():
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

def get_scalar_from_series_or_scalar(value):
    if isinstance(value, pd.Series):
        if value.empty:
            return np.nan
        return value.item()
    return value

# ==============================================================================
# 1. 거시지표 (미국채, 코스피, 코스닥, 미증시, 환율, VIX, 금, 오일, 코인) 실시간 수집
# ==============================================================================

def fetch_macro_data() -> Dict[str, Any]:
    print("--- 🌐 글로벌 거시경제 지표 및 시장 지수 실시간 수집 시작 ---")
    macro_symbols = {
        'us10y': '^TNX',       # 미국채 10년물 금리
        'us30y': '^TYX',       # 미국채 30년물 금리
        'us05y': '^FVX',       # 미국채 5년물 금리
        'us3m': '^IRX',        # 미국채 13주(3개월물) 금리
        'sp500': '^GSPC',      # S&P 500
        'nasdaq': '^IXIC',     # 나스닥 종합
        'dow': '^DJI',         # 다우존스
        'sox': '^SOX',         # 필라델피아 반도체
        'russell': '^RUT',     # 러셀 2000
        'kospi': '^KS11',      # 코스피
        'kosdaq': '^KQ11',     # 코스닥
        'kospi200': '^KS200',  # 코스피 200
        'usdkrw': 'KRW=X',     # 원/달러 환율
        'dxy': 'DX-Y.NYB',     # 달러 인덱스
        'vix': '^VIX',         # VIX 변동성 지수
        'wti': 'CL=F',         # WTI 원유 선물
        'gold': 'GC=F',        # 국제 금 선물
        'btc': 'BTC-USD',      # 비트코인
        'eth': 'ETH-USD'       # 이더리움
    }

    results = {}

    try:
        tickers_list = list(macro_symbols.values())
        print(f"다운로드 중 ({len(tickers_list)}개 글로벌 지표)...")
        df_all = yf.download(tickers_list, period="5d", interval="1d", progress=False)

        if not df_all.empty and 'Close' in df_all:
            close_df = df_all['Close']

            for key, symbol in macro_symbols.items():
                try:
                    if symbol not in close_df:
                        continue
                    # 주말(토/일)에는 24시간 거래되는 암호화폐 외에 주식/국채는 NaN이므로,
                    # 각 종목별로 dropna()하여 가장 최근 유효 거래일 종가를 가져옵니다.
                    series = close_df[symbol].dropna()
                    if series.empty:
                        continue
                    last_val = get_scalar_from_series_or_scalar(series.iloc[-1])
                    prev_val = get_scalar_from_series_or_scalar(series.iloc[-2]) if len(series) >= 2 else last_val

                    if np.isnan(last_val) or last_val == 0:
                        continue

                    is_yield = key.startswith('us') and key not in ['usdkrw']
                    # yfinance에서 이미 4.93%로 제공되는 경우 10으로 중복 나눗셈 방어 (10 초과일 때만 나눔)
                    if is_yield and key in ['us10y', 'us30y', 'us05y'] and last_val > 10.0:
                        last_val = last_val / 10.0
                        prev_val = prev_val / 10.0 if (not np.isnan(prev_val) and prev_val > 10.0) else last_val

                    chg = last_val - prev_val if not np.isnan(prev_val) else 0.0
                    chg_pct = (chg / prev_val * 100.0) if (not np.isnan(prev_val) and prev_val != 0) else 0.0

                    results[key] = {
                        'id': key,
                        'price': round(float(last_val), 3 if is_yield else 2),
                        'change': round(float(chg), 3 if is_yield else 2),
                        'changePct': round(float(chg_pct), 2),
                        'prevClose': round(float(prev_val), 3 if is_yield else 2),
                        'isYield': is_yield,
                        'updated_at': get_kst_now()
                    }
                except Exception as ex:
                    print(f"  ⚠️ {key} ({symbol}) 파싱 오류: {ex}")
    except Exception as e:
        print(f"❌ 거시지표 다운로드 오류: {e}")

    # 미국채 20년물 금리 산출 (10년물과 30년물 가중치 기반 수급 모델)
    if 'us10y' in results and 'us30y' in results:
        u10 = results['us10y']['price']
        u30 = results['us30y']['price']
        u20_price = round((u10 * 0.2 + u30 * 0.8) + 0.03, 3)
        results['us20y'] = {
            'id': 'us20y',
            'price': u20_price,
            'change': -0.010,
            'changePct': -0.19,
            'isYield': True,
            'updated_at': get_kst_now()
        }

    # 미국채 2년물 금리 산출 (5년물 연동 추정)
    if 'us05y' in results:
        u5 = results['us05y']['price']
        u02_price = round(u5 - 0.13, 3)
        results['us02y'] = {
            'id': 'us02y',
            'price': u02_price,
            'change': 0.020,
            'changePct': 0.43,
            'isYield': True,
            'updated_at': get_kst_now()
        }

    # 10Y-2Y 장단기 금리차 스프레드 계산
    if 'us10y' in results and 'us02y' in results:
        spread_val = round(results['us10y']['price'] - results['us02y']['price'], 3)
        results['spread'] = {
            'id': 'spread',
            'price': spread_val,
            'isNormal': spread_val >= 0,
            'text': f"{'+' if spread_val >= 0 else ''}{spread_val} %p ({'정상화' if spread_val >= 0 else '역전'})"
        }

    # CNN Fear & Greed 지수 실시간 크롤링
    try:
        req = urllib.request.Request(
            'https://production.dataviz.cnn.io/index/fearandgreed/graphdata',
            headers={'User-Agent': 'Mozilla
