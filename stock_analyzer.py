import yfinance as yf
import pandas as pd
import numpy as np
import json
import sys
import traceback
from typing import List, Dict, Any
from datetime import datetime

def get_scalar_from_series_or_scalar(value):
    if isinstance(value, pd.Series):
        if value.empty:
            return np.nan
        return value.item()
    return value

def fetch_data(ticker: str) -> pd.DataFrame | None:
    try:
        data = yf.download(ticker, period="540d", prepost=True, auto_adjust=False, progress=False)
        if data.empty:
            print(f"⚠️ [오류] {ticker}: 데이터를 찾을 수 없습니다.")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        print(f"❌ [오류] {ticker}: 데이터 다운로드 실패 ({e})")
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

    if len(close_prices) >= 2:
        indicators['previous_day_close'] = close_prices.iloc[-2]
    else:
        indicators['previous_day_close'] = np.nan

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
    rsi = 100 - (100 / (1 + rs))
    indicators['RSI_14'] = rsi.iloc[-1]

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

    sigma_pct_1sigma = 1 * volatility * 100 if not np.isnan(volatility) else np.nan
    sigma_pct_2sigma = 2 * volatility * 100 if not np.isnan(volatility) else np.nan

    results = {
        'ticker': ticker,
        'stock_name': stock_name,
        'current_price': current_price,
        'is_undervalued': False,
        'ma_check': False,
        'mdd_check': False,
        'sigma_check': False,
        'sigma_check_1sigma': False,
        'sigma_check_2sigma': False,
        'mdd_value': f"{mdd_1y_val * 100:.2f}%",
        'current_drawdown_value': f"{current_drawdown_val * 100:.2f}%",
        'ma_360_price': sma_360,
        'mdd_target_price': mdd_target_price,
        'sigma_target_price_1sigma': sigma_target_price_1sigma,
        'sigma_target_price_2sigma': sigma_target_price_2sigma,
        'sigma_pct_1sigma': sigma_pct_1sigma,
        'sigma_pct_2sigma': sigma_pct_2sigma,
        'ma_score': 0,
        'mdd_score': 0,
        'sigma_score': 0,
        'ma_individual_checks': {}
    }

    ma_periods = [20, 30, 60, 120, 360]
    ma_below_count = 0

    for p in ma_periods:
        ma_key = f'SMA_{p}'
        ma_value = get_scalar_from_series_or_scalar(indicators.get(ma_key))
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
            results['mdd_score'] = (current_drawdown_val / mdd_1y_val) * 40
            results['mdd_score'] = min(results['mdd_score'], 40)
            results['mdd_score'] = max(results['mdd_score'], 0)
    else:
        results['mdd_score'] = 0

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
        'ticker': ticker,
        'stock_name': stock_name,
        'current_price': current_price,
        'is_overvalued': False,
        'rsi_over_check': False,
        'disparity_over_check': False,
        'sigma_upper_over_check': False,
        'rsi_value': rsi_14,
        'disparity_value_pct': np.nan,
        'disparity_target_price': np.nan,
        'sigma_upper_target_price': np.nan,
        'sigma_upper_pct': np.nan,
        'overvalued_score': 0,
        'disparity_score': 0,
        'rsi_score': 0
    }

    if not np.isnan(rsi_14):
        if rsi_14 >= 70:
            results['rsi_over_check'] = True
            rsi_score_calculated = 30 + (rsi_14 - 70)
            results['rsi_score'] = min(40, rsi_score_calculated)
            results['overvalued_score'] += results['rsi_score']

    if not np.isnan(sma_60) and not np.isnan(current_price):
        disparity_value_pct = ((current_price / sma_60) - 1) * 100
        results['disparity_value_pct'] = disparity_value_pct
        results['disparity_target_price'] = sma_60 * 1.15

        if disparity_value_pct > 15:
            results['disparity_over_check'] = True
            results['disparity_score'] = disparity_value_pct - 15
            results['overvalued_score'] += results['disparity_score']

    sigma_upper_target_price = previous_day_close * (1 + 2 * volatility) if not np.isnan(previous_day_close) and not np.isnan(volatility) else np.nan
    results['sigma_upper_target_price'] = sigma_upper_target_price
    results['sigma_upper_pct'] = 2 * volatility * 100 if not np.isnan(volatility) else np.nan

    if not np.isnan(sigma_upper_target_price) and current_price > sigma_upper_target_price:
        results['sigma_upper_over_check'] = True
        results['overvalued_score'] += 30

    results['is_overvalued'] = results['rsi_over_check'] and results['disparity_over_check'] and results['sigma_upper_over_check']
    return results

def get_fundamental_indicators(ticker: str, stock_info: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    fundamental_indicators = {
        'forward_per': np.nan, 'trailing_per': np.nan, 'reported_eps_ttm': np.nan,
        'target_mean_price': np.nan, 'target_upside_pct': np.nan,
        'recommendation_key': 'N/A', 'beta': np.nan, 'market_cap': np.nan
    }

    fundamental_indicators['forward_per'] = stock_info.get('forwardPE', np.nan)
    fundamental_indicators['trailing_per'] = stock_info.get('trailingPE', np.nan)
    fundamental_indicators['reported_eps_ttm'] = stock_info.get('trailingEps', np.nan)

    current_price = df['Close'].iloc[-1] if not df.empty else np.nan
    target_mean_price = stock_info.get('targetMeanPrice', np.nan)
    fundamental_indicators['target_mean_price'] = target_mean_price

    if not np.isnan(target_mean_price) and not np.isnan(current_price) and current_price != 0:
        fundamental_indicators['target_upside_pct'] = ((target_mean_price - current_price) / current_price) * 100

    fundamental_indicators['recommendation_key'] = stock_info.get('recommendationKey', 'N/A')
    fundamental_indicators['beta'] = stock_info.get('beta', np.nan)
    fundamental_indicators['market_cap'] = stock_info.get('marketCap', np.nan)

    return fundamental_indicators

def run_analysis(stock_list: List[str], market_name: str) -> List[Dict[str, Any]]:
    print(f"--- 📉 {market_name} 스크리닝 시작 ({len(stock_list)}개 종목) ---")
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
            print(f"✓ {ticker} 완료: 저평가 {results['total_undervalued_score']}점 / 과열 {results['total_overvalued_score']}점")
        except Exception as e:
            print(f"❌ {ticker} 오류: {e}")
            continue

    return all_results

def main_execution():
    pd.set_option('future.no_silent_downcasting', True)

    # 453810.開設 오타를 453810.KS로 정상 수정 완료
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

    all_results_kr = run_analysis(stock_tickers_kr, "한국 주식")
    all_results_us = run_analysis(stock_tickers_us, "미국 및 암호화폐")
    combined_results = all_results_kr + all_results_us

    screener_db = {}
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

    # 깃허브 웹 대시보드가 읽을 수 있도록 같은 폴더에 저장
    with open("screener_results.json", "w", encoding="utf-8") as f:
        json.dump(screener_db, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 screener_results.json 생성 완료 (총 {len(screener_db)}개 종목)")

if __name__ == "__main__":
    try:
        main_execution()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
