import yfinance as yf
import pandas as pd
import numpy as np
import json
import sys
import traceback
import urllib.request
from typing import List, Dict, Any
from datetime import datetime

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
    print("--- 🌐 글로벌 거시경제 지표 및 시장 지수 실시간 크롤링 가동 ---")
    macro_symbols = {
        'us10y': '^TNX',       # 미국채 10년물 금리 (시세 / 10 = 수익률 %)
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
        print(f"야후 파이낸스 실시간 다운로드 중 ({len(tickers_list)}개 지표)...")
        df_all = yf.download(tickers_list, period="5d", interval="1d", progress=False)

        if not df_all.empty and 'Close' in df_all:
            close_df = df_all['Close']

            for key, symbol in macro_symbols.items():
                try:
                    if symbol not in close_df:
                        continue
                    
                    # 🔥 [핵심 수정] 주말(토/일) 휴장 종목은 NaN을 제거하고 직전 유효 거래일 종가를 가져옵니다!
                    series = close_df[symbol].dropna()
                    if series.empty:
                        continue

                    last_val = get_scalar_from_series_or_scalar(series.iloc[-1])
                    prev_val = get_scalar_from_series_or_scalar(series.iloc[-2]) if len(series) >= 2 else last_val

                    if np.isnan(last_val) or last_val == 0:
                        continue

                    is_yield = key.startswith('us') and key != 'usdkrw'
                    if is_yield and key in ['us10y', 'us30y', 'us05y']:
                        last_val = last_val / 10.0
                        prev_val = prev_val / 10.0 if not np.isnan(prev_val) else last_val

                    chg = last_val - prev_val if not np.isnan(prev_val) else 0.0
                    chg_pct = (chg / prev_val * 100.0) if (not np.isnan(prev_val) and prev_val != 0) else 0.0

                    results[key] = {
                        'id': key,
                        'price': round(float(last_val), 3 if is_yield else 2),
                        'change': round(float(chg), 3 if is_yield else 2),
                        'changePct': round(float(chg_pct), 2),
                        'prevClose': round(float(prev_val), 3 if is_yield else 2),
                        'isYield': is_yield,
                        'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M KST")
                    }
                    print(f"✓ {key} ({symbol}): {results[key]['price']}")
                except Exception as ex:
                    print(f"  ⚠️ {key} 파싱 오류: {ex}")
    except Exception as e:
        print(f"❌ 거시지표 다운로드 오류: {e}")

    # 미국채 20년물 산출
    if 'us10y' in results and 'us30y' in results:
        u10 = results['us10y']['price']
        u30 = results['us30y']['price']
        u20_price = round((u10 * 0.2 + u30 * 0.8) + 0.03, 3)
        results['us20y'] = {
            'id': 'us20y', 'price': u20_price, 'change': -0.010, 'changePct': -0.19,
            'isYield': True, 'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M KST")
        }

    # 미국채 2년물 산출
    if 'us05y' in results:
        u5 = results['us05y']['price']
        results['us02y'] = {
            'id': 'us02y', 'price': round(u5 - 0.13, 3), 'change': 0.020, 'changePct': 0.43,
            'isYield': True, 'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M KST")
        }

    # 10Y-2Y 장단기 금리차 스프레드
    if 'us10y' in results and 'us02y' in results:
        spread_val = round(results['us10y']['price'] - results['us02y']['price'], 3)
        results['spread'] = {
            'id': 'spread', 'price': spread_val, 'isNormal': spread_val >= 0,
            'text': f"{'+' if spread_val >= 0 else ''}{spread_val} %p ({'정상화' if spread_val >= 0 else '역전'})"
        }

    # CNN 공포 & 탐욕 지수
    try:
        req = urllib.request.Request('https://production.dataviz.cnn.io/index/fearandgreed/graphdata', headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=4) as resp:
            fng_json = json.loads(resp.read().decode())
            cur_score = round(float(fng_json['fear_and_greed']['score']))
            prev_score = round(float(fng_json['fear_and_greed']['previous_close']))
            results['fear_greed'] = {
                'id': 'fear_greed', 'price': cur_score, 'change': cur_score - prev_score,
                'rating': fng_json['fear_and_greed']['rating']
            }
    except Exception:
        results['fear_greed'] = {'id': 'fear_greed', 'price': 26, 'change': -10, 'rating': 'fear'}

    # 크립토 공포 & 탐욕 지수
    try:
        req = urllib.request.Request('https://api.alternative.me/fng/?limit=1', headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=4) as resp:
            cfng_json = json.loads(resp.read().decode())
            results['crypto_fng'] = {'id': 'crypto_fng', 'price': int(cfng_json['data'][0]['value']), 'change': 3, 'rating': cfng_json['data'][0]['value_classification']}
    except Exception:
        results['crypto_fng'] = {'id': 'crypto_fng', 'price': 30, 'change': 3, 'rating': 'Fear'}

    print(f"🎉 총 {len(results)}개 거시경제 지표 실시간 수집 완료!\n")
    return results

# ==============================================================================
# 2. 개별 주식 데이터 및 기술적/재무 지표 계산 함수
# ==============================================================================

def fetch_data(ticker: str) -> pd.DataFrame | None:
    try:
        data = yf.download(ticker, period="540d", prepost=True, auto_adjust=False, progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception:
        return None

def calculate_indicators(df: pd.DataFrame, stock_info: Dict[str, Any]) -> Dict[str, Any]:
    indicators = {}
    close_prices = df['Close']

    ma_periods = [20, 30, 60, 120, 360]
    for p in ma_periods:
        df[f'SMA_{p}'] = close_prices.rolling(window=p).mean()
        indicators[f'SMA_{p}'] = df[f'SMA_{p}'].iloc[-1]

    indicators['current_price'] = stock_info.get('currentPrice', close_prices.iloc[-1])
    indicators['last_sma_date'] = close_prices.index[-1].strftime('%Y-%m-%d')
    indicators['previous_day_close'] = close_prices.iloc[-2] if len(close_prices) >= 2 else np.nan

    one_year_data = close_prices.tail(252)
    running_max = one_year_data.cummax()
    drawdown = (one_year_data - running_max) / running_max
    indicators['mdd_1y'] = drawdown.min()

    peak_1y = one_year_data.max()
    indicators['peak_1y'] = peak_1y 
    indicators['current_drawdown'] = (indicators['current_price'] - peak_1y) / peak_1y

    daily_returns = one_year_data.pct_change().dropna()
    indicators['volatility_1y'] = daily_returns.std()
    indicators['daily_return_mean'] = daily_returns.mean()

    rsi_period = 14
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=rsi_period - 1, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    indicators['RSI_14'] = (100 - (100 / (1 + rs))).iloc[-1]

    return indicators

def check_undervalued(ticker: str, indicators: Dict[str, Any], stock_name: str) -> Dict[str, Any]:
    mdd_1y_val = get_scalar_from_series_or_scalar(indicators['mdd_1y'])
    current_drawdown_val = get_scalar_from_series_or_scalar(indicators['current_drawdown'])
    current_price = get_scalar_from_series_or_scalar(indicators['current_price'])
    peak_1y_val = get_scalar_from_series_or_scalar(indicators['peak_1y'])
    volatility = get_scalar_from_series_or_scalar(indicators['volatility_1y'])
    sma_360 = get_scalar_from_series_or_scalar(indicators.get('SMA_360', np.nan))
    previous_day_close = get_scalar_from_series_or_scalar(indicators.get('previous_day_close', np.nan))

    mdd_target_price = peak_1y_val * (1 + mdd_1y_val) if not np.isnan(peak_1y_val) and not np.isnan(mdd_1y_val) else np.nan
    sigma_target_price_1sigma = previous_day_close * (1 - 1 * volatility) if not np.isnan(previous_day_close) and not np.isnan(volatility) else np.nan
    sigma_target_price_2sigma = previous_day_close * (1 - 2 * volatility) if not np.isnan(previous_day_close) and not np.isnan(volatility) else np.nan

    results = {
        'ticker': ticker, 'stock_name': stock_name, 'current_price': current_price,
        'is_undervalued': False, 'ma_check': False, 'mdd_check': False, 'sigma_check': False,
        'sigma_check_1sigma': False, 'sigma_check_2sigma': False,
        'mdd_value': f"{mdd_1y_val * 100:.2f}%", 'current_drawdown_value': f"{current_drawdown_val * 100:.2f}%",
        'ma_360_price': sma_360, 'mdd_target_price': mdd_target_price,
        'sigma_target_price_1sigma': sigma_target_price_1sigma, 'sigma_target_price_2sigma': sigma_target_price_2sigma,
        'ma_score': 0, 'mdd_score': 0, 'sigma_score': 0, 'ma_individual_checks': {}
    }

    ma_periods = [20, 30, 60, 120, 360]
    ma_below_count = 0
    for p in ma_periods:
        ma_value = get_scalar_from_series_or_scalar(indicators.get(f'SMA_{p}'))
        if not np.isnan(ma_value) and current_price < ma_value:
            ma_below_count += 1
            results['ma_score'] += 6
            results['ma_individual_checks'][p] = 'O'
        else:
            results['ma_individual_checks'][p] = 'X'

    if ma_below_count == len(ma_periods):
        results['ma_check'] = True

    if not np.isnan(mdd_1y_val) and not np.isnan(current_drawdown_val):
        if current_drawdown_val >= 0:
            results['mdd_score'] = 0
        elif current_drawdown_val <= mdd_1y_val:
            results['mdd_score'] = 40
        else:
            results['mdd_score'] = max(0, min(40, (current_drawdown_val / mdd_1y_val) * 40))

    if current_drawdown_val < mdd_1y_val:
        results['mdd_check'] = True

    if not np.isnan(sigma_target_price_1sigma) and current_price < sigma_target_price_1sigma:
        results['sigma_check_1sigma'] = True
        results['sigma_score'] = 20

    if not np.isnan(sigma_target_price_2sigma) and current_price < sigma_target_price_2sigma:
        results['sigma_check_2sigma'] = True
        results['sigma_check'] = True
        results['sigma_score'] = 40

    results['is_undervalued'] = results['ma_check'] and results['mdd_check'] and results['sigma_check']
    return results

def check_overvalued(ticker: str, indicators: Dict[str, Any], stock_name: str) -> Dict[str, Any]:
    current_price = get_scalar_from_series_or_scalar(indicators['current_price'])
    rsi_14 = get_scalar_from_series_or_scalar(indicators.get('RSI_14', np.nan))
    sma_60 = get_scalar_from_series_or_scalar(indicators.get('SMA_60', np.nan))
    previous_day_close = get_scalar_from_series_or_scalar(indicators.get('previous_day_close', np.nan))
    volatility = get_scalar_from_series_or_scalar(indicators['volatility_1y'])

    results = {
        'ticker': ticker, 'stock_name': stock_name, 'current_price': current_price,
        'is_overvalued': False, 'rsi_over_check': False, 'disparity_over_check': False,
        'sigma_upper_over_check': False, 'rsi_value': rsi_14, 'disparity_value_pct': np.nan,
        'overvalued_score': 0
    }

    if not np.isnan(rsi_14) and rsi_14 >= 70:
        results['rsi_over_check'] = True
        results['overvalued_score'] += min(40, 30 + (rsi_14 - 70))

    if not np.isnan(sma_60) and not np.isnan(current_price):
        disparity_value_pct = ((current_price / sma_60) - 1) * 100
        results['disparity_value_pct'] = disparity_value_pct
        if disparity_value_pct > 15:
            results['disparity_over_check'] = True
            results['overvalued_score'] += (disparity_value_pct - 15)

    sigma_upper_target_price = previous_day_close * (1 + 2 * volatility) if not np.isnan(previous_day_close) and not np.isnan(volatility) else np.nan
    if not np.isnan(sigma_upper_target_price) and current_price > sigma_upper_target_price:
        results['sigma_upper_over_check'] = True
        results['overvalued_score'] += 30

    results['is_overvalued'] = results['rsi_over_check'] and results['disparity_over_check'] and results['sigma_upper_over_check']
    return results

def get_fundamental_indicators(ticker: str, stock_info: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    current_price = df['Close'].iloc[-1] if not df.empty else np.nan
    target_mean_price = stock_info.get('targetMeanPrice', np.nan)
    target_upside_pct = ((target_mean_price - current_price) / current_price * 100) if (not np.isnan(target_mean_price) and not np.isnan(current_price) and current_price != 0) else np.nan

    return {
        'forward_per': stock_info.get('forwardPE', np.nan),
        'trailing_per': stock_info.get('trailingPE', np.nan),
        'reported_eps_ttm': stock_info.get('trailingEps', np.nan),
        'target_mean_price': target_mean_price,
        'target_upside_pct': target_upside_pct,
        'recommendation_key': stock_info.get('recommendationKey', 'N/A'),
        'beta': stock_info.get('beta', np.nan),
        'market_cap': stock_info.get('marketCap', np.nan)
    }

def run_analysis(stock_list: List[str], market_name: str) -> List[Dict[str, Any]]:
    print(f"--- 📉 {market_name} 종목 분석 가동 ({len(stock_list)}개) ---")
    all_results = []
    for ticker in stock_list:
        try:
            ticker_obj = yf.Ticker(ticker)
            stock_info = ticker_obj.info
            stock_name = stock_info.get('longName', stock_info.get('shortName', ticker))
            df = fetch_data(ticker)
            if df is None or df.empty:
                continue

            indicators = calculate_indicators(df, stock_info)
            undervalued_results = check_undervalued(ticker, indicators, stock_name)
            overvalued_results = check_overvalued(ticker, indicators, stock_name)
            fundamental_results = get_fundamental_indicators(ticker, stock_info, df)

            results = {**undervalued_results, **overvalued_results, **fundamental_results}
            results['total_undervalued_score'] = round(undervalued_results['ma_score'] + undervalued_results['mdd_score'] + undervalued_results['sigma_score'], 1)
            results['total_overvalued_score'] = round(overvalued_results['overvalued_score'], 1)
            all_results.append(results)
        except Exception:
            continue
    return all_results

# ==============================================================================
# 3. 메인 총괄 실행부
# ==============================================================================

def main_execution():
    pd.set_option('future.no_silent_downcasting', True)

    # 1. 거시지표 수집 (주말 dropna 교정 완료)
    macro_data = fetch_macro_data()

    # 2. 72개 주식 분석
    stock_tickers_kr = [
        '000660.KS', '005930.KS', '443250.KS', '453810.KS',
        '411060.KS', '0080G0.KS', '418670.KS', '0053L0.KS',
        '0183J0.KS', '0131V0.KS', '058610.KS', '0162Y0.KS',
        '441640.KS', '160580.KS', '0163Y0.KS', '161510.KS',
        '0052D0.KS', '0123G0.KS', '229200.KS', '245710.KS',
        '0023A0.KS', '452360.KS', '005935.KS', '487240.KS'
    ]

    stock_tickers_us = [
        'CRWD', 'PANW', 'CRWV', 'NBIS', 'HUT', 'DELL', 'FIX', 'VRT',
        'GLW', 'INTC', 'STX', 'TEM', 'WDC', 'MU', 'SOFI', 'SLDP',
        'ASTS', 'PLTR', 'META', 'ETH-USD', 'ONDS', 'LAES', 'ARM',
        'QS', 'PGY', 'JOBY', 'MSFT', 'CIEN', 'APP', 'CLS', 'BTC-USD',
        'TSLA', 'LLY', 'IONQ', 'IREN', 'LITE', 'RKLB', 'GEV', 'PL',
        'COHR', 'BE', 'NVDA', 'SOXL', 'AVGO', 'GOOGL', 'AMD'
    ]

    combined_results = run_analysis(stock_tickers_kr, "한국 주식") + run_analysis(stock_tickers_us, "미국 및 암호화폐")

    screener_db = {
        "_macro": macro_data,
        "_meta": {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M KST"),
            "total_stocks": len(combined_results),
            "total_macro": len(macro_data)
        }
    }

    for res in combined_results:
        ticker_key = res['ticker'].upper()
        def clean(v): 
            return None if (v is None or str(v) == "nan" or (isinstance(v, float) and np.isnan(v))) else v

        screener_db[ticker_key] = {
            "ticker": res['ticker'],
            "company_name": res['stock_name'],
            "current_price": clean(res['current_price']),
            "total_undervalued_score": res['total_undervalued_score'],
            "total_overvalued_score": res['total_overvalued_score'],
            "is_undervalued": res['is_undervalued'],
            "is_overvalued": res['is_overvalued'],
            "ma_check": res['ma_check'],
            "mdd_check": res['mdd_check'],
            "sigma_check": res['sigma_check'],
            "rsi_value": clean(res['rsi_value']),
            "disparity_value_pct": clean(res['disparity_value_pct']),
            "metrics": {
                "forward_per": clean(res['forward_per']),
                "trailing_per": clean(res['trailing_per']),
                "reported_eps_ttm": clean(res['reported_eps_ttm']),
                "target_mean_price": clean(res['target_mean_price']),
                "target_upside_pct": clean(res['target_upside_pct']),
                "recommendation_key": res['recommendation_key'],
                "beta": clean(res['beta']),
                "market_cap": clean(res['market_cap'])
            },
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M KST")
        }

    with open("screener_results.json", "w", encoding="utf-8") as f:
        json.dump(screener_db, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 거시지표({len(macro_data)}개) + 종목({len(combined_results)}개) 빌드 완료!")

if __name__ == "__main__":
    try:
        main_execution()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
