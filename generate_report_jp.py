#!/usr/bin/env python3
"""
generate_report_jp.py
日本株専用クオンツ分析レポートジェネレーター
Usage: python3 generate_report_jp.py [TICKER]  # default: 7203.T
Output: reports/{TICKER}/index.html
"""

import sys
import os
import math
import warnings
import datetime
import traceback

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────
TICKER_SYM = sys.argv[1] if len(sys.argv) > 1 else "7203.T"
BB_WIN = 52        # ボリンジャーバンド期間（週足換算）
N_SIM  = 2000      # モンテカルロ本数
USD_JPY = 158.94

PEER_MAP = {
    "7203.T": ["7267.T", "7201.T", "7261.T", "7270.T", "7269.T"],
    "6701.T": ["6702.T", "6501.T", "6326.T", "9984.T"],  # 富士通・日立・クボタ・ソフトバンクG
    "7267.T": ["7203.T", "7201.T", "7261.T", "7270.T", "7269.T"],  # トヨタ・日産・マツダ・スバル・スズキ
}
PEER_LABEL_MAP = {
    "7203.T": "国内自動車大手",
    "6701.T": "国内IT・電機大手",
    "7267.T": "国内自動車大手",
}

# ─────────────────────────────────────────
# 共通カラーパレット
# ─────────────────────────────────────────
BG   = "#0a0a1a"
CARD = "#0f0f23"
AC   = "#00d4ff"
AC2  = "#ffd700"
AC3  = "#ff6b35"
GR   = "#00ff88"
TX   = "#e0e0f0"
DIM  = "#8888aa"

CHART_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color=TX, family="'Noto Sans JP', 'Helvetica Neue', Arial, sans-serif"),
    margin=dict(l=52, r=20, t=36, b=36),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zeroline=False,
        showline=False,
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zeroline=False,
        showline=False,
    ),
)


def apply_layout(fig, height=400, title="", yaxis_title=""):
    layout = dict(**CHART_LAYOUT)
    layout["height"] = height
    if title:
        layout["title"] = dict(text=title, font=dict(size=13, color=TX), x=0.01)
    if yaxis_title:
        layout["yaxis"] = dict(**CHART_LAYOUT["yaxis"], title=yaxis_title)
    fig.update_layout(**layout)


def safe_html(fig, div_id):
    """Plotly figure → HTML div string. Returns '' on error."""
    try:
        return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)
    except Exception as e:
        print(f"  [WARN] to_html failed for {div_id}: {e}")
        return ""


def fmt_oku(val):
    """円を億円表記に変換"""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    oku = val / 1e8
    if abs(oku) >= 10000:
        return f"{oku/10000:.1f}兆円"
    elif abs(oku) >= 1:
        return f"{oku:.0f}億円"
    else:
        return f"{val:,.0f}円"


def fmt_num(val, digits=2, suffix=""):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return f"{val:.{digits}f}{suffix}"


def fmt_pct(val, digits=1):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return f"{val*100:.{digits}f}%"


def bollinger(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    pct_b = (series - lower) / (upper - lower + 1e-9)
    return sma, upper, lower, pct_b


# ─────────────────────────────────────────
# Section 1: データ取得
# ─────────────────────────────────────────
print(f"[1] Fetching data for {TICKER_SYM} ...")

ticker = yf.Ticker(TICKER_SYM)

# 株価3年分
try:
    hist = ticker.history(period="3y")
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    close = hist["Close"].dropna()
    volume = hist["Volume"].dropna()
    high   = hist["High"].dropna()
    low    = hist["Low"].dropna()
    print(f"  Price rows: {len(close)}")
except Exception as e:
    print(f"  [ERROR] history: {e}")
    close = pd.Series(dtype=float)
    volume = pd.Series(dtype=float)
    high   = pd.Series(dtype=float)
    low    = pd.Series(dtype=float)

# ticker.info
try:
    info = ticker.info
except Exception as e:
    print(f"  [ERROR] info: {e}")
    info = {}

cur_price          = info.get("currentPrice") or info.get("regularMarketPrice") or (float(close.iloc[-1]) if len(close) else None)
trailing_pe        = info.get("trailingPE")
forward_pe         = info.get("forwardPE")
trailing_eps       = info.get("trailingEps")
forward_eps        = info.get("forwardEps")
price_to_book      = info.get("priceToBook")
book_value         = info.get("bookValue")
dividend_yield     = info.get("dividendYield")
dividend_rate      = info.get("dividendRate")
roe                = info.get("returnOnEquity")
op_margin          = info.get("operatingMargins")
net_margin         = info.get("profitMargins")
target_mean        = info.get("targetMeanPrice")
target_high        = info.get("targetHighPrice")
target_low         = info.get("targetLowPrice")
n_analysts         = info.get("numberOfAnalystOpinions")
earnings_growth    = info.get("earningsGrowth")
market_cap         = info.get("marketCap")
ebitda_info        = info.get("ebitda")
total_debt_info    = info.get("totalDebt")
total_cash_info    = info.get("totalCash")
short_name         = info.get("shortName") or info.get("longName") or TICKER_SYM

print(f"  cur_price={cur_price}, PE={trailing_pe}, PBR={price_to_book}, bookValue={book_value}")

# 株式数
shares_count = None
if market_cap and cur_price:
    shares_count = market_cap / cur_price

# 年次損益計算書
try:
    inc = ticker.income_stmt
    print(f"  income_stmt cols: {list(inc.columns)[:4]}")
except Exception as e:
    print(f"  [ERROR] income_stmt: {e}")
    inc = pd.DataFrame()

# 四半期損益計算書
try:
    q_inc = ticker.quarterly_income_stmt
    print(f"  quarterly_income_stmt cols: {list(q_inc.columns)[:4]}")
except Exception as e:
    print(f"  [ERROR] quarterly_income_stmt: {e}")
    q_inc = pd.DataFrame()

# 年次バランスシート
try:
    bs = ticker.balance_sheet
except Exception as e:
    print(f"  [ERROR] balance_sheet: {e}")
    bs = pd.DataFrame()

# 四半期バランスシート
try:
    q_bs = ticker.quarterly_balance_sheet
except Exception as e:
    print(f"  [ERROR] quarterly_balance_sheet: {e}")
    q_bs = pd.DataFrame()

# 配当履歴
try:
    divs = ticker.dividends
    divs.index = pd.to_datetime(divs.index).tz_localize(None)
except Exception as e:
    print(f"  [ERROR] dividends: {e}")
    divs = pd.Series(dtype=float)

# 決算発表日・EPSサプライズ
try:
    earn_dates = ticker.earnings_dates
    if earn_dates is not None:
        earn_dates.index = pd.to_datetime(earn_dates.index).tz_localize(None)
except Exception as e:
    print(f"  [ERROR] earnings_dates: {e}")
    earn_dates = None

# EPS修正トレンド
try:
    eps_trend = ticker.eps_trend
except Exception as e:
    print(f"  [ERROR] eps_trend: {e}")
    eps_trend = None

# EPS修正方向性
try:
    eps_rev = ticker.eps_revisions
except Exception as e:
    print(f"  [ERROR] eps_revisions: {e}")
    eps_rev = None

# カレンダーイベント取得
calendar_events = []  # list of {'date': str, 'label': str, 'color': str, 'impact': str, 'detail': str}
try:
    cal = ticker.calendar
    if cal:
        # 決算発表日
        ed_list = cal.get('Earnings Date') or []
        if not isinstance(ed_list, list):
            ed_list = [ed_list]
        rev_avg = cal.get('Revenue Average')
        rev_lo  = cal.get('Revenue Low')
        rev_hi  = cal.get('Revenue High')
        rev_txt = ""
        if rev_avg:
            rev_txt = f"売上高コンセンサス: ¥{rev_avg/1e12:.2f}兆（¥{rev_lo/1e12:.2f}〜¥{rev_hi/1e12:.2f}兆）" if rev_lo and rev_hi else f"売上高コンセンサス: ¥{rev_avg/1e12:.2f}兆"
        for ed in ed_list:
            if ed:
                calendar_events.append({
                    'date': str(ed), 'label': '決算発表',
                    'type': 'earnings', 'color': AC2,
                    'impact': '±5〜15%', 'detail': rev_txt or '通期業績・ガイダンス発表'
                })
        # 配当落ち日
        ex_div = cal.get('Ex-Dividend Date')
        if ex_div:
            dy_txt = f"配当 ¥{dividend_rate:,.0f}/株" if dividend_rate else "配当落ち"
            calendar_events.append({
                'date': str(ex_div), 'label': '配当落ち日',
                'type': 'dividend', 'color': GR,
                'impact': f'−{dividend_rate/cur_price*100:.1f}%' if (dividend_rate and cur_price) else '−配当相当',
                'detail': dy_txt
            })
    # 直近の過去サプライズから決算インパクト推定
    avg_surprise_abs = None
    if earn_dates is not None and not earn_dates.empty:
        past = earn_dates.dropna(subset=['Surprise(%)']) if 'Surprise(%)' in earn_dates.columns else pd.DataFrame()
        if not past.empty:
            avg_surprise_abs = float(past['Surprise(%)'].abs().mean())
    calendar_events.sort(key=lambda e: e['date'])
    print(f"  カレンダーイベント: {len(calendar_events)}件")
except Exception as e:
    print(f"  [WARN] calendar: {e}")

# ─── OBV計算 ───
obv = pd.Series(dtype=float)
obv_ma20 = pd.Series(dtype=float)
try:
    if len(close) > 0 and len(volume) > 0:
        obv_vals = []
        obv_running = 0.0
        cl = close.reindex(volume.index).ffill()
        for i, (dt, vol) in enumerate(volume.items()):
            if i == 0:
                obv_running += float(vol)
            else:
                prev_close = float(cl.iloc[i - 1])
                cur_close  = float(cl.loc[dt])
                if cur_close > prev_close:
                    obv_running += float(vol)
                elif cur_close < prev_close:
                    obv_running -= float(vol)
            obv_vals.append(obv_running)
        obv = pd.Series(obv_vals, index=volume.index)
        obv_ma20 = obv.rolling(20).mean()
except Exception as e:
    print(f"  [WARN] OBV: {e}")

# ─── EV/EBITDA データ取得 ───
print("  EV/EBITDA データ取得中...")
ebitda_q = pd.Series(dtype=float)
try:
    ebitda_keys = ['EBITDA', 'Normalized EBITDA']
    for k in ebitda_keys:
        if k in q_inc.index:
            raw = q_inc.loc[k].dropna()
            ebitda_q = pd.Series(
                {pd.to_datetime(c).tz_localize(None): abs(float(v)) for c, v in raw.items()}
            ).sort_index()
            break
    if ebitda_q.empty:
        q_cf = ticker.quarterly_cashflow
        da_keys = ['Depreciation And Amortization', 'Depreciation Amortization Depletion']
        oi_keys = ['Operating Income', 'EBIT']
        oi = da = None
        for k in oi_keys:
            if k in q_inc.index: oi = q_inc.loc[k]; break
        for k in da_keys:
            if k in q_cf.index: da = q_cf.loc[k]; break
        if oi is not None and da is not None:
            for col in oi.index:
                dt = pd.to_datetime(col).tz_localize(None)
                o_val = oi.get(col, np.nan)
                d_val = da.get(col, np.nan)
                if pd.notna(o_val) and pd.notna(d_val):
                    ebitda_q[dt] = abs(float(o_val)) + abs(float(d_val))
            ebitda_q = ebitda_q.sort_index()
    print(f"  四半期EBITDA: {len(ebitda_q)}件")
except Exception as e:
    print(f"  EBITDA取得エラー: {e}")

ebitda_annual = {}
try:
    ebitda_keys = ['EBITDA', 'Normalized EBITDA']
    for k in ebitda_keys:
        if k in inc.index:
            for col, val in inc.loc[k].dropna().items():
                ebitda_annual[pd.to_datetime(col).tz_localize(None)] = abs(float(val))
            break
    if not ebitda_annual:
        a_cf = ticker.cashflow
        da_keys = ['Depreciation And Amortization', 'Depreciation Amortization Depletion']
        oi_keys = ['Operating Income', 'EBIT']
        oi = da = None
        for k in oi_keys:
            if k in inc.index: oi = inc.loc[k]; break
        for k in da_keys:
            if k in a_cf.index: da = a_cf.loc[k]; break
        if oi is not None and da is not None:
            for col in oi.index:
                dt = pd.to_datetime(col).tz_localize(None)
                o_val = oi.get(col, np.nan)
                d_val = da.get(col, np.nan)
                if pd.notna(o_val) and pd.notna(d_val):
                    ebitda_annual[dt] = abs(float(o_val)) + abs(float(d_val))
    print(f"  年次EBITDA: {len(ebitda_annual)}件")
except Exception as e:
    print(f"  年次EBITDA取得エラー: {e}")

net_debt_q = {}
try:
    debt_keys = ['Total Debt', 'Long Term Debt', 'Current Debt',
                 'Long Term Capital Lease Obligation', 'Current Capital Lease Obligation']
    cash_keys = ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments']
    for col in q_bs.columns:
        dt = pd.to_datetime(col).tz_localize(None)
        debt = cash = 0.0
        for k in debt_keys:
            if k in q_bs.index and pd.notna(q_bs.loc[k, col]):
                debt = abs(float(q_bs.loc[k, col])); break
        for k in cash_keys:
            if k in q_bs.index and pd.notna(q_bs.loc[k, col]):
                cash = abs(float(q_bs.loc[k, col])); break
        net_debt_q[dt] = debt - cash
    net_debt_q = dict(sorted(net_debt_q.items()))
    # 最新の純負債が大きく負（= 債務が取れていない）の場合 → info.totalDebt でフォールバック
    if net_debt_q and total_debt_info:
        latest_nd = list(net_debt_q.values())[-1]
        info_nd = total_debt_info - (total_cash_info or 0)
        # info側の純負債が正なのに BS側が負 → BSのdebtキー不足と判断
        if info_nd > 0 and latest_nd < 0:
            net_debt_q = {dt: info_nd for dt in net_debt_q}
            print(f"  純負債: BS純負債が負 → info fallback {info_nd/1e12:.2f}兆円")
except Exception as e:
    print(f"  純負債取得エラー: {e}")

# 決算発表日補正
def get_release_dates(fiscal_dates):
    release_map = {}
    try:
        if earn_dates is not None and not earn_dates.empty:
            ed_idx = earn_dates.index.normalize()
            rel_dates = sorted(ed_idx.tolist())
            for fdate in fiscal_dates:
                cands = [d for d in rel_dates
                         if pd.Timedelta(days=30) <= (d - fdate) <= pd.Timedelta(days=120)]
                if cands:
                    release_map[fdate] = min(cands)
    except Exception as e:
        print(f"  発表日マップエラー: {e}")
    for fdate in fiscal_dates:
        if fdate not in release_map:
            release_map[fdate] = fdate + pd.Timedelta(days=45)
    return release_map

fiscal_dates = list(ebitda_q.index)
release_map = get_release_dates(fiscal_dates)
ebitda_q_released = pd.Series(
    {release_map[fd]: val for fd, val in ebitda_q.items()}, dtype=float
).sort_index()

def build_ttm_ebitda(close_index, ebitda_q_released, ebitda_annual):
    result = {}
    ann_dates = sorted(ebitda_annual.keys()) if ebitda_annual else []
    # 空や非DatetimeIndexの場合は四半期パスをスキップ
    has_q = (not ebitda_q_released.empty and
             isinstance(ebitda_q_released.index, pd.DatetimeIndex))
    for date in close_index:
        past_q = ebitda_q_released[ebitda_q_released.index <= date] if has_q else pd.Series(dtype=float)
        if len(past_q) >= 4:
            last_4 = past_q.iloc[-4:]
            valid_count = last_4.notna().sum()
            if valid_count >= 3:
                ttm = last_4.sum() * (4 / valid_count)
                if ttm > 0:
                    result[date] = ttm
                    continue
        prev_d = [d for d in ann_dates if d <= date]
        next_d = [d for d in ann_dates if d > date]
        if prev_d and next_d:
            d0, d1 = max(prev_d), min(next_d)
            e0, e1 = ebitda_annual[d0], ebitda_annual[d1]
            ratio = (date - d0).days / max((d1 - d0).days, 1)
            ttm = e0 + (e1 - e0) * ratio
            if ttm > 0: result[date] = ttm
        elif prev_d:
            ttm = ebitda_annual[max(prev_d)]
            if ttm > 0: result[date] = ttm
    return pd.Series(result)

ttm_ebitda = build_ttm_ebitda(close.index, ebitda_q_released, ebitda_annual)

ev_series = pd.Series(dtype=float)
if shares_count and not close.empty:
    mktcap_series = close * shares_count
    if net_debt_q:
        nd_series = pd.Series(net_debt_q).reindex(close.index, method='ffill')
        ev_series = mktcap_series + nd_series
    else:
        ev_series = mktcap_series

ev_ebitda_all = pd.Series(dtype=float)
if not ev_series.empty and not ttm_ebitda.empty:
    ev_ebitda_all = (ev_series / ttm_ebitda).dropna()
    ev_ebitda_all = ev_ebitda_all[(ev_ebitda_all > 0) & (ev_ebitda_all < 200)]

has_ev_ebitda = False
ee_plot = pd.DataFrame()
cur_ee = cur_ee_ma = cur_ee_up = cur_ee_lo = cur_ee_pctb = 0.0
if len(ev_ebitda_all) > 60:
    ee_df = ev_ebitda_all.to_frame('ee')
    ee_df['ma']    = ee_df['ee'].rolling(52).mean()
    ee_df['std']   = ee_df['ee'].rolling(52).std()
    ee_df['upper'] = ee_df['ma'] + 2 * ee_df['std']
    ee_df['lower'] = ee_df['ma'] - 2 * ee_df['std']
    ee_df['pct_b'] = ((ee_df['ee'] - ee_df['lower']) / (ee_df['upper'] - ee_df['lower'])).clip(-0.5, 1.5)
    ee_plot = ee_df.dropna()
    if len(ee_plot) > 0:
        cur_ee      = float(ee_plot['ee'].iloc[-1])
        cur_ee_ma   = float(ee_plot['ma'].iloc[-1])
        cur_ee_up   = float(ee_plot['upper'].iloc[-1])
        cur_ee_lo   = float(ee_plot['lower'].iloc[-1])
        cur_ee_pctb = float(ee_plot['pct_b'].iloc[-1])
        has_ev_ebitda = True
        print(f"  EV/EBITDA={cur_ee:.1f}x  MA={cur_ee_ma:.1f}x  %B={cur_ee_pctb:.2f}")
    else:
        print("  EV/EBITDAデータ不足（dropna後空）")
else:
    print(f"  EV/EBITDAデータ不足（{len(ev_ebitda_all)}点）")

# ─── 年次EPS系列を構築（PERシリーズ用）───
annual_eps_series = pd.Series(dtype=float)
try:
    if not inc.empty and shares_count:
        ni_row = None
        for k in ["Net Income", "Net Income Common Stockholders", "Net Income From Continuing Operation Net Minority Interest"]:
            if k in inc.index:
                ni_row = inc.loc[k]
                break
        if ni_row is not None:
            # 各年次の日付とEPSを辞書に
            eps_dict = {}
            for col in ni_row.index:
                dt = pd.to_datetime(col).tz_localize(None)
                val = ni_row[col]
                if pd.notna(val):
                    eps_dict[dt] = float(val) / shares_count
            if eps_dict:
                # close の各日付に対してその時点で最新の年次EPSを割り当て
                eps_dates = sorted(eps_dict.keys())
                def get_eps_at(dt):
                    # dt 以前で最も新しい決算日のEPSを返す
                    # 年次決算は通常3〜4ヶ月後に発表されるので、決算日+120日以降に反映
                    available = [d for d in eps_dates if d + pd.Timedelta(days=120) <= dt]
                    if not available:
                        # 最も古いデータで代替
                        return eps_dict[eps_dates[-1]] if eps_dates else None
                    return eps_dict[max(available)]
                annual_eps_series = pd.Series(
                    [get_eps_at(dt) for dt in close.index],
                    index=close.index,
                    dtype=float
                )
except Exception as e:
    print(f"  [WARN] annual_eps_series: {e}")

# ─── 四半期BPSシリーズを構築（PBRシリーズ用）───
bps_series = pd.Series(dtype=float)
try:
    if shares_count and shares_count > 0:
        bps_dict = {}
        if not q_bs.empty:
            for col in q_bs.columns:
                dt = pd.to_datetime(col).tz_localize(None)
                for k in ["Common Stock Equity", "Stockholders Equity", "Total Equity Gross Minority Interest"]:
                    if k in q_bs.index and pd.notna(q_bs.loc[k, col]):
                        bps_val = abs(float(q_bs.loc[k, col])) / shares_count
                        bps_dict[dt] = bps_val
                        break
        if not bps_dict and book_value:
            # フォールバック: info['bookValue'] を固定で使用
            bps_dict = {close.index[0]: float(book_value)} if len(close) else {}

        if bps_dict:
            bps_dates = sorted(bps_dict.keys())
            def get_bps_at(dt):
                available = [d for d in bps_dates if d + pd.Timedelta(days=45) <= dt]
                if not available:
                    return bps_dict[bps_dates[-1]] if bps_dates else None
                return bps_dict[max(available)]
            bps_series = pd.Series(
                [get_bps_at(dt) for dt in close.index],
                index=close.index,
                dtype=float
            )
    elif book_value:
        bps_series = pd.Series(float(book_value), index=close.index, dtype=float)
except Exception as e:
    print(f"  [WARN] bps_series: {e}")
    if book_value and len(close):
        bps_series = pd.Series(float(book_value), index=close.index, dtype=float)

print(f"  annual_eps_series non-nan: {annual_eps_series.notna().sum()}")
print(f"  bps_series non-nan: {bps_series.notna().sum()}")

# ─────────────────────────────────────────
# Section 2: チャート生成
# ─────────────────────────────────────────
print("[2] Generating charts ...")

# ─── Chart 1: 株価チャート ───
html_price = ""
try:
    sma20, bb_upper, bb_lower, bb_pctb = bollinger(close, 20, 2)
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    fig_price = go.Figure()
    # BB 塗りつぶし
    fig_price.add_trace(go.Scatter(
        x=list(bb_upper.index), y=list(bb_upper),
        name="BB Upper", line=dict(color="rgba(255,165,100,0.4)", width=1),
        showlegend=True
    ))
    fig_price.add_trace(go.Scatter(
        x=list(bb_lower.index), y=list(bb_lower),
        name="BB Lower", line=dict(color="rgba(255,165,100,0.4)", width=1),
        fill="tonexty", fillcolor="rgba(255,165,100,0.05)",
        showlegend=True
    ))
    # SMA20
    fig_price.add_trace(go.Scatter(
        x=list(sma20.index), y=list(sma20),
        name="SMA20", line=dict(color=AC2, width=1, dash="dash"),
    ))
    # SMA50
    fig_price.add_trace(go.Scatter(
        x=list(sma50.index), y=list(sma50),
        name="SMA50", line=dict(color=GR, width=1, dash="dash"),
    ))
    # SMA200
    fig_price.add_trace(go.Scatter(
        x=list(sma200.index), y=list(sma200),
        name="SMA200", line=dict(color="#9b59b6", width=1, dash="dash"),
    ))
    # 終値（最前面）
    fig_price.add_trace(go.Scatter(
        x=list(close.index), y=list(close),
        name="終値", line=dict(color=AC, width=1.5),
    ))
    apply_layout(fig_price, height=420, yaxis_title="株価（円）")
    html_price = safe_html(fig_price, "chart-price")
    print("  Chart 1 (price) OK")
except Exception as e:
    print(f"  [WARN] Chart 1 (price): {e}")
    traceback.print_exc()

# ─── Chart 2: 出来高 ───
html_vol = ""
try:
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=list(volume.index), y=list(volume),
        name="出来高", marker_color="rgba(100,100,120,0.6)"
    ))
    apply_layout(fig_vol, height=200, yaxis_title="出来高")
    html_vol = safe_html(fig_vol, "chart-vol")
    print("  Chart 2 (volume) OK")
except Exception as e:
    print(f"  [WARN] Chart 2 (volume): {e}")

# ─── Chart 2b: MACDチャート ───
html_macd = ""
try:
    if len(close) > 40:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal    = macd_line.ewm(span=9, adjust=False).mean()
        histogram  = macd_line - signal

        hist_colors = [GR if float(v) >= 0 else "#ff4444" for v in histogram]
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Bar(
            x=list(histogram.index), y=[float(v) for v in histogram],
            name='ヒストグラム', marker_color=hist_colors, opacity=0.7,
            hovertemplate='ヒストグラム: %{y:.2f}<extra></extra>'
        ))
        fig_macd.add_trace(go.Scatter(
            x=list(macd_line.index), y=[float(v) for v in macd_line],
            name='MACD', line=dict(color=AC, width=1.8),
            hovertemplate='MACD: %{y:.2f}<extra></extra>'
        ))
        fig_macd.add_trace(go.Scatter(
            x=list(signal.index), y=[float(v) for v in signal],
            name='シグナル(9)', line=dict(color=AC3, width=1.5, dash='dash'),
            hovertemplate='Signal: %{y:.2f}<extra></extra>'
        ))
        fig_macd.add_hline(y=0, line=dict(color='rgba(255,255,255,0.2)', width=1))
        cur_macd = float(macd_line.iloc[-1])
        cur_sig  = float(signal.iloc[-1])
        macd_cross = "ゴールデンクロス（買いシグナル）" if cur_macd > cur_sig else "デッドクロス（売りシグナル）"
        apply_layout(fig_macd, height=260, title='MACD（12/26/9）', yaxis_title='MACD')
        html_macd = safe_html(fig_macd, "chart-macd")
        print(f"  Chart 2b (MACD) OK: {macd_cross}")
    else:
        print("  Chart 2b (MACD): skipped (insufficient data)")
except Exception as e:
    print(f"  [WARN] Chart 2b (MACD): {e}")
    traceback.print_exc()

# ─── Chart 3: PER BBチャート ───
html_per = ""
per_pctb_now = None
try:
    if len(annual_eps_series.dropna()) > BB_WIN:
        per_series = close / annual_eps_series
        per_series = per_series.replace([np.inf, -np.inf], np.nan).dropna()
        # 負のPERは除外
        per_series = per_series[per_series > 0]
        if len(per_series) > BB_WIN:
            sma_per, ub_per, lb_per, pctb_per = bollinger(per_series, BB_WIN, 2)
            per_pctb_now = float(pctb_per.iloc[-1]) if len(pctb_per.dropna()) else None

            fig_per = go.Figure()
            fig_per.add_trace(go.Scatter(
                x=list(ub_per.index), y=list(ub_per),
                name="BB Upper", line=dict(color="rgba(255,165,100,0.4)", width=1),
            ))
            fig_per.add_trace(go.Scatter(
                x=list(lb_per.index), y=list(lb_per),
                name="BB Lower", line=dict(color="rgba(255,165,100,0.4)", width=1),
                fill="tonexty", fillcolor="rgba(255,165,100,0.05)",
            ))
            fig_per.add_trace(go.Scatter(
                x=list(sma_per.index), y=list(sma_per),
                name="SMA", line=dict(color=AC2, width=1, dash="dash"),
            ))
            fig_per.add_trace(go.Scatter(
                x=list(per_series.index), y=list(per_series),
                name="PER", line=dict(color=AC, width=1.5),
            ))
            apply_layout(fig_per, height=380, yaxis_title="PER（倍）")
            pctb_label = f"{per_pctb_now*100:.0f}%" if per_pctb_now is not None else "N/A"
            fig_per.add_annotation(
                x=0.01, y=0.95, xref="paper", yref="paper",
                text=f"%B: {pctb_label}",
                showarrow=False, font=dict(size=12, color=AC2),
                align="left"
            )
            html_per = safe_html(fig_per, "chart-per")
            print("  Chart 3 (PER BB) OK")
        else:
            print("  Chart 3 (PER BB): not enough data after filter")
    else:
        print("  Chart 3 (PER BB): skipped (no annual EPS data)")
except Exception as e:
    print(f"  [WARN] Chart 3 (PER BB): {e}")
    traceback.print_exc()

# ─── Chart 4: PBR BBチャート ───
html_pbr = ""
pbr_pctb_now = None
try:
    if len(bps_series.dropna()) > BB_WIN:
        pbr_series = close / bps_series
        pbr_series = pbr_series.replace([np.inf, -np.inf], np.nan).dropna()
        pbr_series = pbr_series[pbr_series > 0]
        if len(pbr_series) > BB_WIN:
            sma_pbr, ub_pbr, lb_pbr, pctb_pbr = bollinger(pbr_series, BB_WIN, 2)
            pbr_pctb_now = float(pctb_pbr.iloc[-1]) if len(pctb_pbr.dropna()) else None

            fig_pbr = go.Figure()
            fig_pbr.add_trace(go.Scatter(
                x=list(ub_pbr.index), y=list(ub_pbr),
                name="BB Upper", line=dict(color="rgba(255,165,100,0.4)", width=1),
            ))
            fig_pbr.add_trace(go.Scatter(
                x=list(lb_pbr.index), y=list(lb_pbr),
                name="BB Lower", line=dict(color="rgba(255,165,100,0.4)", width=1),
                fill="tonexty", fillcolor="rgba(255,165,100,0.05)",
            ))
            fig_pbr.add_trace(go.Scatter(
                x=list(sma_pbr.index), y=list(sma_pbr),
                name="SMA", line=dict(color=AC2, width=1, dash="dash"),
            ))
            fig_pbr.add_trace(go.Scatter(
                x=list(pbr_series.index), y=list(pbr_series),
                name="PBR", line=dict(color=GR, width=1.5),
            ))
            # TSE改革基準: PBR 1.0倍ライン
            fig_pbr.add_hline(
                y=1.0,
                line=dict(color="rgba(255,0,0,0.5)", width=1, dash="dash"),
                annotation_text="1.0x (TSE基準)",
                annotation_font=dict(color="rgba(255,100,100,0.8)", size=10),
            )
            apply_layout(fig_pbr, height=380, yaxis_title="PBR（倍）")
            pctb_label = f"{pbr_pctb_now*100:.0f}%" if pbr_pctb_now is not None else "N/A"
            fig_pbr.add_annotation(
                x=0.01, y=0.95, xref="paper", yref="paper",
                text=f"%B: {pctb_label}",
                showarrow=False, font=dict(size=12, color=GR),
                align="left"
            )
            html_pbr = safe_html(fig_pbr, "chart-pbr")
            print("  Chart 4 (PBR BB) OK")
        else:
            print("  Chart 4 (PBR BB): not enough data after filter")
    else:
        # フォールバック: info['bookValue'] で単純計算
        if book_value and len(close) > BB_WIN:
            pbr_series = close / float(book_value)
            sma_pbr, ub_pbr, lb_pbr, pctb_pbr = bollinger(pbr_series, BB_WIN, 2)
            pbr_pctb_now = float(pctb_pbr.iloc[-1]) if len(pctb_pbr.dropna()) else None

            fig_pbr = go.Figure()
            fig_pbr.add_trace(go.Scatter(
                x=list(ub_pbr.index), y=list(ub_pbr),
                name="BB Upper", line=dict(color="rgba(255,165,100,0.4)", width=1),
            ))
            fig_pbr.add_trace(go.Scatter(
                x=list(lb_pbr.index), y=list(lb_pbr),
                name="BB Lower", line=dict(color="rgba(255,165,100,0.4)", width=1),
                fill="tonexty", fillcolor="rgba(255,165,100,0.05)",
            ))
            fig_pbr.add_trace(go.Scatter(
                x=list(sma_pbr.index), y=list(sma_pbr),
                name="SMA", line=dict(color=AC2, width=1, dash="dash"),
            ))
            fig_pbr.add_trace(go.Scatter(
                x=list(pbr_series.index), y=list(pbr_series),
                name="PBR", line=dict(color=GR, width=1.5),
            ))
            fig_pbr.add_hline(
                y=1.0,
                line=dict(color="rgba(255,0,0,0.5)", width=1, dash="dash"),
                annotation_text="1.0x (TSE基準)",
                annotation_font=dict(color="rgba(255,100,100,0.8)", size=10),
            )
            apply_layout(fig_pbr, height=380, yaxis_title="PBR（倍）")
            html_pbr = safe_html(fig_pbr, "chart-pbr")
            print("  Chart 4 (PBR BB) OK (fallback bookValue)")
        else:
            print("  Chart 4 (PBR BB): skipped (no BPS data)")
except Exception as e:
    print(f"  [WARN] Chart 4 (PBR BB): {e}")
    traceback.print_exc()

# ─── Chart 5: 配当利回りチャート ───
html_div = ""
avg_div_yield = None
try:
    if dividend_rate and dividend_rate > 0 and len(close) > 0:
        div_yield_series = (dividend_rate / close) * 100
        avg_div_yield = float(div_yield_series.mean())

        fig_div = go.Figure()
        fig_div.add_trace(go.Scatter(
            x=list(div_yield_series.index), y=list(div_yield_series),
            name="配当利回り", line=dict(color=AC2, width=1.5),
            fill="tozeroy", fillcolor="rgba(255,215,0,0.08)",
        ))
        # 過去平均ライン
        fig_div.add_hline(
            y=avg_div_yield,
            line=dict(color="rgba(255,215,0,0.5)", width=1, dash="dash"),
            annotation_text=f"平均 {avg_div_yield:.2f}%",
            annotation_font=dict(color=AC2, size=10),
        )
        apply_layout(fig_div, height=280, yaxis_title="%")
        html_div = safe_html(fig_div, "chart-div")
        print("  Chart 5 (dividend yield) OK")
    else:
        print("  Chart 5 (dividend yield): skipped (no dividend data)")
except Exception as e:
    print(f"  [WARN] Chart 5 (dividend yield): {e}")

# ─── Chart 6: 四半期業績チャート ───
html_earnings = ""
html_progress = ""
try:
    # 四半期データに Revenue/Operating Income が含まれるか確認
    Q_INC_KEYS = ["Total Revenue", "Revenue", "Operating Income", "Operating Revenue",
                  "Net Income", "Net Income Common Stockholders"]
    has_quarterly_income = not q_inc.empty and any(k in q_inc.index for k in Q_INC_KEYS)

    if has_quarterly_income:
        cols = sorted(q_inc.columns, key=lambda c: pd.to_datetime(c))
        cols = cols[-12:]

        # ── FY判定（トヨタは4月〜3月決算）
        def get_fy(dt):
            return dt.year if dt.month <= 3 else dt.year + 1

        # 4月始まりFYの四半期番号: Jun=Q1, Sep=Q2, Dec=Q3, Mar=Q4
        FY_Q_MAP = {6: 1, 9: 2, 12: 3, 3: 4}

        labels, rev_vals, op_vals, ni_vals = [], [], [], []
        q_meta = []  # (label, fy, q_in_fy, op_val_raw, rev_val_raw, ni_val_raw)

        for col in cols:
            dt = pd.to_datetime(col)
            fy = get_fy(dt)
            q_in_fy = FY_Q_MAP.get(dt.month, ((dt.month - 1) // 3) % 4 + 1)
            label = f"FY{str(fy)[2:]} Q{q_in_fy}"
            labels.append(label)

            rev = op = ni = None
            for k in ["Total Revenue", "Revenue"]:
                if k in q_inc.index and pd.notna(q_inc.loc[k, col]):
                    rev = float(q_inc.loc[k, col]); break
            for k in ["Operating Income", "Operating Revenue"]:
                if k in q_inc.index and pd.notna(q_inc.loc[k, col]):
                    op = float(q_inc.loc[k, col]); break
            for k in ["Net Income", "Net Income Common Stockholders"]:
                if k in q_inc.index and pd.notna(q_inc.loc[k, col]):
                    ni = float(q_inc.loc[k, col]); break

            rev_vals.append(rev / 1e12 if rev else None)
            op_vals.append(op / 1e11 if op else None)
            ni_vals.append(ni / 1e11 if ni else None)
            q_meta.append((label, fy, q_in_fy, op, rev, ni))

        fig_earnings = go.Figure()
        if any(v is not None for v in rev_vals):
            fig_earnings.add_trace(go.Bar(x=labels, y=rev_vals, name="売上高（兆円）", marker_color=AC))
        if any(v is not None for v in op_vals):
            fig_earnings.add_trace(go.Bar(x=labels, y=op_vals, name="営業利益（千億円）", marker_color=GR))
        if any(v is not None for v in ni_vals):
            fig_earnings.add_trace(go.Bar(x=labels, y=ni_vals, name="純利益（千億円）", marker_color=AC3))
        fig_earnings.update_layout(barmode="group")
        apply_layout(fig_earnings, height=360, yaxis_title="金額（兆/千億円）")
        html_earnings = safe_html(fig_earnings, "chart-earnings")
        print("  Chart 6 (earnings) OK")

        # ── 目標達成率チャート ─────────────────────────────────────────
        # 通期目標: 過去FYは income_stmt 実績、今期FYはアナリスト推計
        fy_targets = {}  # fy → {'op': 兆円, 'rev': 兆円, 'ni': 兆円}
        try:
            if not inc.empty:
                for col in inc.columns:
                    dt = pd.to_datetime(col)
                    fy = get_fy(dt)
                    op_a = rev_a = ni_a = None
                    for k in ["Operating Income"]:
                        if k in inc.index and pd.notna(inc.loc[k, col]):
                            op_a = float(inc.loc[k, col]); break
                    for k in ["Total Revenue", "Revenue"]:
                        if k in inc.index and pd.notna(inc.loc[k, col]):
                            rev_a = float(inc.loc[k, col]); break
                    for k in ["Net Income", "Net Income Common Stockholders"]:
                        if k in inc.index and pd.notna(inc.loc[k, col]):
                            ni_a = float(inc.loc[k, col]); break
                    if op_a or rev_a:
                        fy_targets[fy] = {'op': op_a, 'rev': rev_a, 'ni': ni_a}
        except Exception:
            pass

        # 今期FY（最新四半期のFY）のアナリスト推計を追加
        current_fy = max(m[1] for m in q_meta) if q_meta else None
        if current_fy and current_fy not in fy_targets:
            try:
                est = ticker.earnings_estimate
                rev_est = ticker.revenue_estimate
                shares_count = info.get('marketCap', 0) / info.get('currentPrice', 1)
                if est is not None and '0y' in est.index:
                    eps_est = float(est.loc['0y', 'avg']) if pd.notna(est.loc['0y', 'avg']) else None
                    ni_est = eps_est * shares_count if eps_est else None
                else:
                    ni_est = None
                rev_est_val = None
                if rev_est is not None and '0y' in rev_est.index:
                    rev_est_val = float(rev_est.loc['0y', 'avg']) if pd.notna(rev_est.loc['0y', 'avg']) else None
                # 営業利益推計 = 売上高推計 × 直近実績営業利益率
                op_margin = info.get('operatingMargins', None)
                op_est = rev_est_val * op_margin if (rev_est_val and op_margin) else None
                if op_est or rev_est_val:
                    fy_targets[current_fy] = {'op': op_est, 'rev': rev_est_val, 'ni': ni_est}
                    print(f"  今期FY{current_fy}目標: 売上¥{rev_est_val/1e12:.1f}兆 営業利益¥{op_est/1e12:.1f}兆" if op_est and rev_est_val else "  今期FY推計取得")
            except Exception as e:
                print(f"  今期FY推計取得エラー: {e}")

        # 達成率を計算（FYごとに累積）
        if fy_targets:
            # FYごとに累積op/revを集計
            fy_cumul = {}  # fy → [(q_label, cum_op_pct, cum_rev_pct)]
            fy_op_acc = {}; fy_rev_acc = {}
            for label, fy, q_in_fy, op_raw, rev_raw, ni_raw in q_meta:
                if fy not in fy_targets:
                    continue
                target = fy_targets[fy]
                fy_op_acc.setdefault(fy, 0.0)
                fy_rev_acc.setdefault(fy, 0.0)
                if op_raw:
                    fy_op_acc[fy] += op_raw
                if rev_raw:
                    fy_rev_acc[fy] += rev_raw
                op_pct  = (fy_op_acc[fy]  / target['op']  * 100) if (target.get('op')  and target['op']  > 0) else None
                rev_pct = (fy_rev_acc[fy] / target['rev'] * 100) if (target.get('rev') and target['rev'] > 0) else None
                fy_cumul.setdefault(fy, []).append((label, op_pct, rev_pct, q_in_fy))

            if fy_cumul:
                fig_progress = go.Figure()
                colors_fy = [AC, AC2, AC3, GR, '#9b59b6']
                for i, (fy, pts) in enumerate(sorted(fy_cumul.items())):
                    lbs   = [p[0] for p in pts]
                    op_ps = [p[1] for p in pts]
                    rv_ps = [p[2] for p in pts]
                    col_  = colors_fy[i % len(colors_fy)]
                    is_current = (fy == current_fy)
                    lw = 2.5 if is_current else 1.5
                    dash_ = 'solid' if is_current else 'dot'
                    label_suffix = '（今期・予想ベース）' if is_current else '（実績ベース）'
                    if any(v is not None for v in op_ps):
                        fig_progress.add_trace(go.Scatter(
                            x=[f"Q{p[3]}" for p in pts], y=op_ps,
                            mode='lines+markers',
                            name=f'FY{str(fy)[2:]} 営業利益{label_suffix}',
                            line=dict(color=col_, width=lw, dash=dash_),
                            marker=dict(size=8),
                            connectgaps=True,
                            hovertemplate=f'FY{str(fy)[2:]} %{{x}}<br>累計達成率: %{{y:.1f}}%<extra></extra>'
                        ))

                # ペース基準線（25%/50%/75%/100%）
                for q, pct in [(1, 25), (2, 50), (3, 75), (4, 100)]:
                    fig_progress.add_annotation(
                        x=f"Q{q}", y=pct,
                        text=f'{pct}%', showarrow=False,
                        font=dict(color='rgba(255,255,255,0.3)', size=10),
                        xanchor='left', yanchor='bottom'
                    )
                for pct in [25, 50, 75, 100]:
                    fig_progress.add_hline(
                        y=pct, line=dict(color='rgba(255,255,255,0.12)', width=1, dash='dot')
                    )
                fig_progress.update_layout(
                    xaxis=dict(categoryorder='array', categoryarray=['Q1','Q2','Q3','Q4'])
                )
                apply_layout(fig_progress, height=360,
                             title="営業利益 通期目標達成率（累計進捗）",
                             yaxis_title="達成率（%）")
                html_progress = safe_html(fig_progress, "chart-progress")
                print(f"  Chart 6b (progress) OK: FY={sorted(fy_cumul.keys())}")
        else:
            print("  Chart 6b (progress): 目標データ不足でスキップ")
    else:
        # ── 年次データでフォールバック ──────────────────────────────────
        print("  Chart 6 (earnings): quarterly income missing, falling back to annual")
        if not inc.empty:
            ann_cols = sorted(inc.columns, key=lambda c: pd.to_datetime(c))
            ann_labels, ann_rev, ann_op, ann_ni = [], [], [], []
            for col in ann_cols:
                dt = pd.to_datetime(col)
                fy = dt.year if dt.month <= 3 else dt.year + 1
                ann_labels.append(f"FY{str(fy)[2:]}")
                rv = op = ni = None
                for k in ["Total Revenue", "Revenue"]:
                    if k in inc.index and pd.notna(inc.loc[k, col]):
                        rv = float(inc.loc[k, col]); break
                for k in ["Operating Income"]:
                    if k in inc.index and pd.notna(inc.loc[k, col]):
                        op = float(inc.loc[k, col]); break
                for k in ["Net Income", "Net Income Common Stockholders"]:
                    if k in inc.index and pd.notna(inc.loc[k, col]):
                        ni = float(inc.loc[k, col]); break
                ann_rev.append(rv / 1e12 if rv else None)
                ann_op.append(op / 1e11 if op else None)
                ann_ni.append(ni / 1e11 if ni else None)

            fig_earnings = go.Figure()
            if any(v is not None for v in ann_rev):
                fig_earnings.add_trace(go.Bar(x=ann_labels, y=ann_rev, name="売上高（兆円）", marker_color=AC))
            if any(v is not None for v in ann_op):
                fig_earnings.add_trace(go.Bar(x=ann_labels, y=ann_op, name="営業利益（千億円）", marker_color=GR))
            if any(v is not None for v in ann_ni):
                fig_earnings.add_trace(go.Bar(x=ann_labels, y=ann_ni, name="純利益（千億円）", marker_color=AC3))
            fig_earnings.update_layout(barmode="group")
            apply_layout(fig_earnings, height=360, yaxis_title="金額（兆/千億円）")
            html_earnings = safe_html(fig_earnings, "chart-earnings")
            print("  Chart 6 (earnings) OK: annual fallback")
except Exception as e:
    print(f"  [WARN] Chart 6 (earnings): {e}")
    traceback.print_exc()

# ─── Chart 7: アナリスト目標株価 ───
html_target = ""
upside_pct = None
try:
    if cur_price and (target_low or target_mean or target_high):
        labels = []
        values = []
        colors = []
        if cur_price:
            labels.append("現在株価"); values.append(cur_price); colors.append(AC)
        if target_low:
            labels.append("目標下限"); values.append(target_low); colors.append("#ff6b6b")
        if target_mean:
            labels.append("目標平均"); values.append(target_mean); colors.append(AC2)
            upside_pct = (target_mean - cur_price) / cur_price * 100
        if target_high:
            labels.append("目標上限"); values.append(target_high); colors.append(GR)

        fig_target = go.Figure()
        fig_target.add_trace(go.Bar(
            x=labels, y=values,
            marker_color=colors,
            text=[f"¥{v:,.0f}" for v in values],
            textposition="outside",
            textfont=dict(size=11, color=TX),
        ))
        apply_layout(fig_target, height=300, yaxis_title="株価（円）")
        if n_analysts:
            fig_target.add_annotation(
                x=0.99, y=0.95, xref="paper", yref="paper",
                text=f"アナリスト数: {n_analysts}名",
                showarrow=False, font=dict(size=11, color=DIM),
                align="right"
            )
        html_target = safe_html(fig_target, "chart-target")
        print("  Chart 7 (target) OK")
    else:
        print("  Chart 7 (target): skipped (no analyst data)")
except Exception as e:
    print(f"  [WARN] Chart 7 (target): {e}")

# ─── Chart 8: 決算EPSサプライズ ───
html_surprise = ""
try:
    if earn_dates is not None and len(earn_dates) > 0:
        # Reported EPS と Estimated EPS からサプライズ率を計算
        ed = earn_dates.copy()
        # カラム名を確認
        rep_col = None
        est_col = None
        for c in ed.columns:
            cl = str(c).lower()
            if "reported" in cl:
                rep_col = c
            elif "estimate" in cl or "eps estimate" in cl.replace(" ", ""):
                est_col = c

        if rep_col and est_col:
            ed2 = ed[[rep_col, est_col]].dropna()
            ed2 = ed2[ed2[est_col] != 0]
            ed2["surprise_pct"] = (ed2[rep_col] - ed2[est_col]) / ed2[est_col].abs() * 100
            ed2 = ed2.sort_index()
            # 過去N件
            ed2 = ed2.tail(12)

            colors_s = [GR if v >= 0 else "#ff6b6b" for v in ed2["surprise_pct"]]
            fig_surprise = go.Figure()
            fig_surprise.add_trace(go.Bar(
                x=[str(d.date()) for d in ed2.index],
                y=list(ed2["surprise_pct"]),
                marker_color=colors_s,
                name="EPSサプライズ率",
                text=[f"{v:.1f}%" for v in ed2["surprise_pct"]],
                textposition="outside",
                textfont=dict(size=10, color=TX),
            ))
            fig_surprise.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))
            apply_layout(fig_surprise, height=320, yaxis_title="サプライズ率（%）")
            html_surprise = safe_html(fig_surprise, "chart-surprise")
            print("  Chart 8 (EPS surprise) OK")
        else:
            print(f"  Chart 8: columns not found: {list(ed.columns)}")
    else:
        print("  Chart 8 (EPS surprise): skipped (no earnings dates)")
except Exception as e:
    print(f"  [WARN] Chart 8 (EPS surprise): {e}")
    traceback.print_exc()

# ─── Chart 9: EPS修正トレンド ───
html_eps_trend = ""
try:
    if eps_trend is not None and not eps_trend.empty:
        fig_eps_trend = go.Figure()
        periods = list(eps_trend.columns) if hasattr(eps_trend, "columns") else []
        rows = list(eps_trend.index) if hasattr(eps_trend, "index") else []
        # 典型的な行名: '0q', '+1q', '0y', '+1y' など
        colors_et = [AC, GR, AC2, AC3, "#9b59b6"]
        for i, row_label in enumerate(rows[:5]):
            vals = eps_trend.loc[row_label]
            if pd.notna(vals).any():
                fig_eps_trend.add_trace(go.Scatter(
                    x=[str(c) for c in vals.index],
                    y=list(vals),
                    name=str(row_label),
                    line=dict(color=colors_et[i % len(colors_et)], width=1.5),
                    mode="lines+markers",
                    marker=dict(size=6),
                ))
        apply_layout(fig_eps_trend, height=360, yaxis_title="EPS予想（円）")
        html_eps_trend = safe_html(fig_eps_trend, "chart-eps-trend")
        print("  Chart 9 (EPS trend) OK")
    else:
        print("  Chart 9 (EPS trend): skipped (no data)")
except Exception as e:
    print(f"  [WARN] Chart 9 (EPS trend): {e}")

# ─── Chart 10: ピアーEV/EBITDA比較 ───
html_peer = ""
try:
    peers = PEER_MAP.get(TICKER_SYM, [])
    peer_tickers = [TICKER_SYM] + peers
    peer_names  = []
    peer_ev_ebitda = []

    for pt in peer_tickers:
        try:
            pi = yf.Ticker(pt).info
            mc  = pi.get("marketCap") or 0
            td  = pi.get("totalDebt") or 0
            tc  = pi.get("totalCash") or 0
            ebt = pi.get("ebitda")
            nm  = pi.get("shortName") or pi.get("longName") or pt
            if ebt and ebt != 0:
                ev = mc + td - tc
                ev_ebitda = ev / ebt
                if 0 < ev_ebitda < 100:
                    peer_names.append(nm[:12])
                    peer_ev_ebitda.append(round(ev_ebitda, 2))
        except Exception:
            pass

    if peer_names:
        bar_colors = [AC if n == (short_name[:12] if short_name else TICKER_SYM) else "#4466aa" for n in peer_names]
        # 主銘柄は強調
        if peer_names:
            bar_colors[0] = AC

        fig_peer = go.Figure()
        fig_peer.add_trace(go.Bar(
            x=peer_ev_ebitda, y=peer_names,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.1f}x" for v in peer_ev_ebitda],
            textposition="outside",
            textfont=dict(size=11, color=TX),
        ))
        apply_layout(fig_peer, height=max(300, 60 * len(peer_names)), yaxis_title="")
        fig_peer.update_layout(xaxis_title="EV/EBITDA（倍）")
        html_peer = safe_html(fig_peer, "chart-peer")
        print(f"  Chart 10 (peer EV/EBITDA) OK: {len(peer_names)} peers")
    else:
        print("  Chart 10 (peer): skipped (no peer data)")
except Exception as e:
    print(f"  [WARN] Chart 10 (peer EV/EBITDA): {e}")
    traceback.print_exc()

# ─── Chart 11: モンテカルロ予測 ───
html_mc = ""
mc_up10_prob = None
try:
    if len(close) > 30 and cur_price:
        log_ret = np.log(close / close.shift(1)).dropna()
        hv30 = float(log_ret.tail(30).std()) * math.sqrt(252)

        # ドリフト: アナリスト目標から逆算、なければ0
        if target_mean and cur_price:
            annual_drift = math.log(target_mean / cur_price)  # 1年で目標に到達と仮定
        else:
            annual_drift = float(log_ret.mean()) * 252

        T = 30      # 日数
        dt = 1/252
        S0 = cur_price
        sigma = hv30

        np.random.seed(42)
        paths = np.zeros((N_SIM, T + 1))
        paths[:, 0] = S0
        mu = annual_drift / 252
        sig_dt = sigma * math.sqrt(dt)

        for t in range(1, T + 1):
            z = np.random.standard_normal(N_SIM)
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2 * dt) + sig_dt * z)

        # 最終価格分布
        final_prices = paths[:, -1]
        mc_up10_prob = float(np.mean(final_prices >= S0 * 1.10)) * 100

        # パーセンタイルパス
        pcts = [5, 25, 50, 75, 95]
        pct_paths = {p: np.percentile(paths, p, axis=0) for p in pcts}

        future_dates = pd.date_range(start=close.index[-1], periods=T+1, freq="B")

        fig_mc = go.Figure()
        pct_colors = {
            5:  "rgba(255,107,53,0.6)",
            25: "rgba(255,165,0,0.5)",
            50: "rgba(0,212,255,0.9)",
            75: "rgba(0,255,136,0.5)",
            95: "rgba(0,255,136,0.6)",
        }
        pct_labels = {5: "5%ile", 25: "25%ile", 50: "中央値", 75: "75%ile", 95: "95%ile"}
        for p in pcts:
            fig_mc.add_trace(go.Scatter(
                x=list(future_dates), y=list(pct_paths[p]),
                name=pct_labels[p],
                line=dict(color=pct_colors[p], width=1.5 if p == 50 else 1, dash="dot" if p != 50 else "solid"),
            ))

        # 過去90日の終値
        hist_90 = close.tail(90)
        fig_mc.add_trace(go.Scatter(
            x=list(hist_90.index), y=list(hist_90),
            name="実績", line=dict(color=TX, width=1.5),
        ))

        # +10% ライン
        fig_mc.add_hline(
            y=S0 * 1.10,
            line=dict(color="rgba(0,255,136,0.4)", width=1, dash="dash"),
            annotation_text=f"+10% (¥{S0*1.10:,.0f})",
            annotation_font=dict(color=GR, size=10),
        )

        apply_layout(fig_mc, height=420, yaxis_title="株価（円）")
        fig_mc.add_annotation(
            x=0.01, y=0.04, xref="paper", yref="paper",
            text=f"30日後+10%超の確率: {mc_up10_prob:.1f}%  |  HV30: {hv30*100:.1f}%  |  N={N_SIM:,}本",
            showarrow=False, font=dict(size=11, color=AC2),
            align="left",
            bgcolor="rgba(0,0,0,0.5)",
        )
        html_mc = safe_html(fig_mc, "chart-mc")
        print(f"  Chart 11 (Monte Carlo) OK: +10% prob = {mc_up10_prob:.1f}%")
    else:
        print("  Chart 11 (MC): skipped (insufficient data)")
except Exception as e:
    print(f"  [WARN] Chart 11 (MC): {e}")
    traceback.print_exc()

# ─── Chart 12: OBV需給 ───
html_obv = ""
try:
    if len(obv) > 0:
        fig_obv = go.Figure()
        fig_obv.add_trace(go.Scatter(
            x=list(obv.index), y=[float(v) for v in obv],
            name='OBV', line=dict(color=GR, width=2),
            hovertemplate='OBV: %{y:,.0f}<extra></extra>'
        ))
        fig_obv.add_trace(go.Scatter(
            x=list(obv_ma20.index), y=[float(v) for v in obv_ma20.dropna()],
            name='OBV MA20', line=dict(color=AC2, width=1.5, dash='dash')
        ))
        apply_layout(fig_obv, height=280, title='OBV（On-Balance Volume）— 需給トレンド', yaxis_title='OBV')
        html_obv = safe_html(fig_obv, "chart-obv")
        obv_trend = '上昇' if float(obv.iloc[-1]) > float(obv.iloc[-20]) else '下落'
        obv_vs_ma  = float(obv.iloc[-1]) > float(obv_ma20.dropna().iloc[-1]) if len(obv_ma20.dropna()) > 0 else None
        print("  Chart 12 (OBV) OK")
    else:
        print("  Chart 12 (OBV): skipped (no data)")
except Exception as e:
    print(f"  [WARN] Chart 12 (OBV): {e}")
    traceback.print_exc()

# ─── Chart 13: EV/EBITDA BBバンド ───
html_ee = ""
html_ee_pctb = ""
html_ee_matrix = ""
try:
    if has_ev_ebitda:
        fig_ee = go.Figure()
        fig_ee.add_trace(go.Scatter(
            x=list(ee_plot.index), y=list(ee_plot['upper']),
            mode='lines', line=dict(color='rgba(255,107,53,0.4)', width=1),
            name=f'BB上限 {cur_ee_up:.1f}x'))
        fig_ee.add_trace(go.Scatter(
            x=list(ee_plot.index), y=list(ee_plot['lower']),
            mode='lines', line=dict(color='rgba(255,107,53,0.4)', width=1),
            fill='tonexty', fillcolor='rgba(255,107,53,0.15)',
            name=f'BB下限 {cur_ee_lo:.1f}x'))
        fig_ee.add_trace(go.Scatter(
            x=list(ee_plot.index), y=list(ee_plot['ma']),
            mode='lines', line=dict(color=AC2, width=1.5, dash='dash'),
            name=f'MA52 {cur_ee_ma:.1f}x'))
        fig_ee.add_trace(go.Scatter(
            x=list(ee_plot.index), y=list(ee_plot['ee']),
            mode='lines', line=dict(color=AC, width=2),
            name=f'EV/EBITDA {cur_ee:.1f}x',
            hovertemplate='EV/EBITDA: %{y:.1f}x<extra></extra>'))
        fig_ee.add_annotation(
            x=ee_plot.index[-1], y=cur_ee,
            text=f"現在 {cur_ee:.1f}x", showarrow=False,
            font=dict(color=AC, size=11), xanchor='right', yanchor='bottom')
        apply_layout(fig_ee, height=380, title=f'EV/EBITDA ボリンジャーバンド（52日±2σ）', yaxis_title='EV/EBITDA（倍）')
        html_ee = safe_html(fig_ee, "chart-ev-ebitda")

        # %B チャート
        fig_ee_pctb = go.Figure()
        fig_ee_pctb.add_trace(go.Scatter(
            x=list(ee_plot.index), y=list(ee_plot['pct_b']),
            mode='lines', line=dict(color=AC3, width=2),
            name='%B', hovertemplate='%B: %{y:.2f}<extra></extra>'))
        for lvl, lc in [(1.0, 'rgba(255,68,68,0.5)'), (0.8, 'rgba(255,107,53,0.4)'),
                        (0.5, 'rgba(255,255,255,0.2)'), (0.2, 'rgba(0,255,136,0.4)'), (0.0, 'rgba(0,212,255,0.5)')]:
            fig_ee_pctb.add_hline(y=lvl, line=dict(color=lc, width=0.8, dash='dash'))
        apply_layout(fig_ee_pctb, height=240, title='EV/EBITDA %B（BBバンド内の位置）', yaxis_title='%B')
        html_ee_pctb = safe_html(fig_ee_pctb, "chart-ev-ebitda-pctb")
        print("  Chart 13 (EV/EBITDA BB) OK")
    else:
        print("  Chart 13 (EV/EBITDA BB): skipped")
except Exception as e:
    print(f"  [WARN] Chart 13 (EV/EBITDA BB): {e}")
    traceback.print_exc()

# ─── Chart 14: PER × EV/EBITDA マトリクス ───
# 月次リサンプル: 日次では PER・EV/EBITDA 両方が株価に支配され対角線になるため
# 月次にすることで四半期ごとの EPS / EBITDA 更新タイミング差が散布図に現れる
try:
    if has_ev_ebitda and len(ee_plot) > 0:
        per_series = close / annual_eps_series
        per_series = per_series[(per_series > 0) & (per_series < 200)].dropna()
        if len(per_series) > 12:
            # 月次末でリサンプル（四半期更新タイミング差を可視化）
            per_mo  = per_series.resample('ME').last().dropna()
            ee_mo   = ev_ebitda_all.resample('ME').last().dropna()
            # 共通月次インデックス
            common_mo = per_mo.index.intersection(ee_mo.index)
            if len(common_mo) > 6:
                # 月次 BB (12ヶ月ウィンドウ)
                per_mo_df = per_mo.reindex(common_mo).to_frame('per')
                ee_mo_df  = ee_mo.reindex(common_mo).to_frame('ee')
                WIN_M = min(24, len(common_mo) // 2)
                for df_, col_ in [(per_mo_df, 'per'), (ee_mo_df, 'ee')]:
                    df_['ma']    = df_[col_].rolling(WIN_M).mean()
                    df_['std']   = df_[col_].rolling(WIN_M).std()
                    df_['upper'] = df_['ma'] + 2 * df_['std']
                    df_['lower'] = df_['ma'] - 2 * df_['std']
                    df_['pct_b'] = ((df_[col_] - df_['lower']) / (df_['upper'] - df_['lower'])).clip(-0.5, 1.5)
                per_mo_df = per_mo_df.dropna()
                ee_mo_df  = ee_mo_df.dropna()
                common = per_mo_df.index.intersection(ee_mo_df.index)
            else:
                common = pd.DatetimeIndex([])
            per_plot_m = per_mo_df  # 現在値計算用に保持
            if len(common) > 6:
                x_per  = [float(per_mo_df.loc[d, 'pct_b']) for d in common]
                y_ee   = [float(ee_mo_df.loc[d, 'pct_b']) for d in common]
                colors = list(range(len(common)))

                fig_matrix = go.Figure()
                fig_matrix.add_trace(go.Scatter(
                    x=x_per, y=y_ee,
                    mode='markers',
                    marker=dict(
                        color=colors, colorscale='Plasma', size=5,
                        colorbar=dict(title='時系列（古→新）', tickfont=dict(color=TX)),
                        opacity=0.7
                    ),
                    hovertemplate='PER %%B: %{x:.2f}<br>EV/EBITDA %%B: %{y:.2f}<extra></extra>'
                ))
                # 現在値
                cur_per_pctb = float(per_plot_m['pct_b'].iloc[-1])
                fig_matrix.add_trace(go.Scatter(
                    x=[cur_per_pctb], y=[cur_ee_pctb],
                    mode='markers+text',
                    marker=dict(color=GR, size=14, symbol='star'),
                    text=['現在'], textposition='top center',
                    textfont=dict(color=GR, size=11), name='現在'
                ))
                for lv in [0.0, 0.5, 1.0]:
                    fig_matrix.add_hline(y=lv, line=dict(color='rgba(255,255,255,0.12)', width=0.8, dash='dot'))
                    fig_matrix.add_vline(x=lv, line=dict(color='rgba(255,255,255,0.12)', width=0.8, dash='dot'))
                # 象限ラベル
                for qx, qy, qt, qc in [
                    (-0.3, 1.3, '割安PER\n割高EV', 'rgba(255,215,0,0.7)'),
                    (1.3, 1.3, '割高PER\n割高EV', 'rgba(255,68,68,0.7)'),
                    (-0.3, -0.3, '割安PER\n割安EV', 'rgba(0,255,136,0.7)'),
                    (1.3, -0.3, '割高PER\n割安EV', 'rgba(0,212,255,0.7)'),
                ]:
                    fig_matrix.add_annotation(x=qx, y=qy, text=qt, showarrow=False,
                        font=dict(size=9, color=qc), align='center')
                apply_layout(fig_matrix, height=400,
                    title='PER × EV/EBITDA マトリクス（%B 二軸）')
                fig_matrix.update_layout(
                    xaxis=dict(**CHART_LAYOUT['xaxis'], title='PER %B（← 割安 | 割高 →）'),
                    yaxis=dict(**CHART_LAYOUT['yaxis'], title='EV/EBITDA %B（← 割安 | 割高 →）'),
                    showlegend=False
                )
                html_ee_matrix = safe_html(fig_matrix, "chart-matrix")
                print("  Chart 14 (matrix) OK")
        else:
            print("  Chart 14 (matrix): skipped (insufficient monthly data)")
    else:
        print("  Chart 14 (matrix): skipped")
except Exception as e:
    print(f"  [WARN] Chart 14 (matrix): {e}")
    traceback.print_exc()

# ─── Chart 15: 3ヶ月株価予測（90日MC + イベント）───
html_mc90 = ""
mc90_table = []   # [(label, p10, p25, p50, p75, p90)]
mc90_up10_prob = None
try:
    if len(close) > 30 and cur_price:
        log_ret = np.log(close / close.shift(1)).dropna()
        hv30    = float(log_ret.tail(30).std()) * math.sqrt(252)

        if target_mean and cur_price:
            annual_drift = math.log(target_mean / cur_price)
        else:
            annual_drift = float(log_ret.mean()) * 252

        T90 = 90
        dt_  = 1 / 252
        S0   = cur_price
        sigma = hv30
        mu90  = annual_drift / 252
        sig_dt90 = sigma * math.sqrt(dt_)

        np.random.seed(42)
        paths90 = np.zeros((N_SIM, T90 + 1))
        paths90[:, 0] = S0
        for step in range(1, T90 + 1):
            z = np.random.standard_normal(N_SIM)
            paths90[:, step] = paths90[:, step-1] * np.exp(
                (mu90 - 0.5 * sigma**2 * dt_) + sig_dt90 * z)

        future_dates90 = pd.date_range(start=close.index[-1], periods=T90+1, freq="B")

        pcts90 = [10, 25, 50, 75, 90]
        pct_paths90 = {p: np.percentile(paths90, p, axis=0) for p in pcts90}
        mc90_up10_prob = float(np.mean(paths90[:, -1] >= S0 * 1.10)) * 100

        # 30/60/90日時点のパーセンタイル価格テーブル
        for days, label in [(30, '1ヶ月後'), (60, '2ヶ月後'), (90, '3ヶ月後')]:
            idx = min(days, T90)
            row = [label] + [float(np.percentile(paths90[:, idx], p)) for p in [10, 25, 50, 75, 90]]
            mc90_table.append(row)

        fig_mc90 = go.Figure()

        # 過去90日実績
        hist_90 = close.tail(90)
        fig_mc90.add_trace(go.Scatter(
            x=list(hist_90.index), y=list(hist_90),
            name='実績株価', line=dict(color=TX, width=2),
            hovertemplate='実績: ¥%{y:,.0f}<extra></extra>'
        ))

        # 信頼帯（10〜90%ile 塗りつぶし）
        fig_mc90.add_trace(go.Scatter(
            x=list(future_dates90) + list(future_dates90)[::-1],
            y=list(pct_paths90[10]) + list(pct_paths90[90])[::-1],
            fill='toself', fillcolor='rgba(0,212,255,0.06)',
            line=dict(color='rgba(0,0,0,0)'), name='10〜90%ile帯', showlegend=True
        ))
        fig_mc90.add_trace(go.Scatter(
            x=list(future_dates90) + list(future_dates90)[::-1],
            y=list(pct_paths90[25]) + list(pct_paths90[75])[::-1],
            fill='toself', fillcolor='rgba(0,212,255,0.12)',
            line=dict(color='rgba(0,0,0,0)'), name='25〜75%ile帯', showlegend=True
        ))
        # 中央値
        fig_mc90.add_trace(go.Scatter(
            x=list(future_dates90), y=list(pct_paths90[50]),
            name='中央値', line=dict(color=AC, width=2),
            hovertemplate='中央値: ¥%{y:,.0f}<extra></extra>'
        ))
        # ±10%ライン
        for mult, lbl, col, col_dim in [
            (1.10, '+10%', GR, 'rgba(0,255,136,0.4)'),
            (0.90, '−10%', '#ff4444', 'rgba(255,68,68,0.4)')
        ]:
            fig_mc90.add_hline(
                y=S0 * mult,
                line=dict(color=col_dim, width=1, dash='dash'),
                annotation_text=f'{lbl} ¥{S0*mult:,.0f}',
                annotation_font=dict(color=col, size=10)
            )

        # イベントラインを追加
        today = pd.Timestamp.today().normalize()
        end_date = future_dates90[-1]
        for ev in calendar_events:
            try:
                ev_dt = pd.Timestamp(ev['date'])
                if today <= ev_dt <= end_date:
                    fig_mc90.add_vline(
                        x=ev_dt.timestamp() * 1000,
                        line=dict(color=ev['color'], width=1.5, dash='dot'),
                        annotation_text=ev['label'],
                        annotation_position='top',
                        annotation_font=dict(color=ev['color'], size=10)
                    )
            except Exception:
                pass

        apply_layout(fig_mc90, height=460, yaxis_title='株価（円）')
        fig_mc90.update_layout(
            title=dict(text='3ヶ月株価予測（GBM モンテカルロ N=2,000）', font=dict(color=TX, size=14)),
            legend=dict(orientation='h', y=1.08, bgcolor='rgba(0,0,0,0)')
        )
        fig_mc90.add_annotation(
            x=0.01, y=0.03, xref='paper', yref='paper',
            text=f"3ヶ月後+10%超の確率: {mc90_up10_prob:.1f}%  |  HV30: {hv30*100:.1f}%  |  ドリフト: {annual_drift*100:.1f}%/年",
            showarrow=False, font=dict(size=11, color=AC2),
            align='left', bgcolor='rgba(0,0,0,0.5)'
        )
        html_mc90 = safe_html(fig_mc90, "chart-mc90")
        print(f"  Chart 15 (MC 90day) OK: 3m+10% prob={mc90_up10_prob:.1f}%")
    else:
        print("  Chart 15 (MC 90day): skipped")
except Exception as e:
    print(f"  [WARN] Chart 15 (MC 90day): {e}")
    traceback.print_exc()

# ─────────────────────────────────────────
# Section 3: HTML生成
# ─────────────────────────────────────────
print("[3] Generating HTML ...")

now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M JST")

# KPI 計算値
kpi_price = f"¥{cur_price:,.0f}" if cur_price else "N/A"
kpi_per   = fmt_num(trailing_pe, 1, "倍") if trailing_pe else (fmt_num(forward_pe, 1, "倍(FWD)") if forward_pe else "N/A")
kpi_pbr   = fmt_num(price_to_book, 2, "倍") if price_to_book else "N/A"
# 配当利回り: dividendRate/currentPrice が最も信頼できる
if dividend_rate and cur_price:
    kpi_div = f"{dividend_rate/cur_price*100:.2f}%"
elif dividend_yield:
    # yfinanceは日本株で%形式(2.82)を返すケースがある
    kpi_div = f"{dividend_yield:.2f}%" if dividend_yield > 1 else f"{dividend_yield*100:.2f}%"
else:
    kpi_div = "N/A"
# ROE: infoにない場合は財務諸表から計算
if roe:
    kpi_roe = fmt_pct(roe)
else:
    try:
        ni_keys = ['Net Income Common Stockholders', 'Net Income']
        eq_keys = ['Common Stock Equity', 'Stockholders Equity']
        ni = next((float(inc.loc[k].iloc[0]) for k in ni_keys if k in inc.index and pd.notna(inc.loc[k].iloc[0])), None)
        eq = next((float(bs.loc[k].iloc[0]) for k in eq_keys if k in bs.index and pd.notna(bs.loc[k].iloc[0])), None)
        kpi_roe = f"{ni/eq*100:.1f}%" if ni and eq and eq > 0 else "N/A"
        if ni and eq and eq > 0:
            roe = ni / eq  # サマリー評価でも使えるよう更新
    except Exception:
        kpi_roe = "N/A"
kpi_target = f"¥{target_mean:,.0f}" if target_mean else "N/A"

upside_str = f"（アップサイド {upside_pct:+.1f}%）" if upside_pct is not None else ""
peer_label = PEER_LABEL_MAP.get(TICKER_SYM, "同業他社")

# 各セクションのHTML
def section_html(label):
    return f'<div class="sec">{label}</div>'


def card(title, desc, content, insight=""):
    ins = f'<div class="ib">{insight}</div>' if insight else ""
    return f'''<div class="cc">
  <div class="ct">{title}</div>
  <div class="cd">{desc}</div>
  {content}
  {ins}
</div>'''


# KPIカード
def kpi_card(label, value, sub=""):
    return f'''<div class="kpi">
  <div class="kl">{label}</div>
  <div class="kv">{value}</div>
  {"<div class='ks'>" + sub + "</div>" if sub else ""}
</div>'''


kpi_row = "".join([
    kpi_card("現在株価", kpi_price, "JPY"),
    kpi_card("PER (TTM)", kpi_per, "株価収益率"),
    kpi_card("PBR", kpi_pbr, "TSE基準: 1.0倍"),
    kpi_card("配当利回り", kpi_div, f"配当額 ¥{dividend_rate:,.0f}" if dividend_rate else ""),
    kpi_card("ROE", kpi_roe, "自己資本利益率"),
    kpi_card("目標株価(平均)", kpi_target, upside_str if upside_str else f"アナリスト {n_analysts}名" if n_analysts else ""),
])

# チャートカード組み立て
body_sections = []

# テクニカル
body_sections.append(section_html("テクニカル分析"))
if html_price:
    body_sections.append(card(
        "株価チャート（3年）",
        "終値・ボリンジャーバンド（20日）・SMA50・SMA200",
        html_price,
        "ボリンジャーバンドの収縮（スクイーズ）はブレイクアウトの前兆。SMA200上方での推移は長期上昇トレンドの維持を示す。"
    ))
if html_vol:
    body_sections.append(card(
        "出来高チャート",
        "日次出来高",
        html_vol,
    ))
if html_macd:
    try:
        macd_insight = f"現在: MACD={cur_macd:.2f} / Signal={cur_sig:.2f} — <strong>{macd_cross}</strong>"
    except NameError:
        macd_insight = ""
    body_sections.append(card(
        "MACD（12/26/9）",
        "EMA12 − EMA26 のモメンタム指標。MACDがシグナルを上抜けでゴールデンクロス（買い）、下抜けでデッドクロス（売り）。",
        html_macd,
        macd_insight,
    ))

# バリュエーション
body_sections.append(section_html("バリュエーション分析"))
per_pbr_grid = ""
if html_per or html_pbr:
    left = ""
    right = ""
    if html_per:
        left = f'''<div class="cc">
  <div class="ct">PER ボリンジャーバンド（{BB_WIN}日）</div>
  <div class="cd">過去PERの統計的レンジ内での位置を確認</div>
  {html_per}
</div>'''
    if html_pbr:
        right = f'''<div class="cc">
  <div class="ct">PBR ボリンジャーバンド（{BB_WIN}日）</div>
  <div class="cd">東証改革によりPBR1.0倍割れは改善圧力がかかる</div>
  {html_pbr}
</div>'''
    if left and right:
        per_pbr_grid = f'<div class="grid2">{left}{right}</div>'
    else:
        per_pbr_grid = left + right
    body_sections.append(per_pbr_grid)

if html_div:
    body_sections.append(card(
        "配当利回りトレンド",
        f"配当利回りの推移（dividendRate = ¥{dividend_rate:,.0f} / 株価）",
        html_div,
        f"過去平均利回り: {avg_div_yield:.2f}%" if avg_div_yield else "",
    ))

# 業績
body_sections.append(section_html("業績分析"))
if html_earnings:
    body_sections.append(card(
        "四半期業績推移",
        "売上高（兆円）・営業利益・純利益（千億円）",
        html_earnings,
        "売上高と利益の方向性が一致し、利益率が改善傾向にあるかを確認。"
    ))
if html_progress:
    body_sections.append(card(
        "営業利益 通期目標達成率",
        "四半期ごとの累計営業利益を通期目標（過去FYは実績、今期FYはアナリスト予想）で割った進捗率。Q3時点で75%超なら概ね順調。",
        html_progress,
        "実線＝今期（予想ベース）、点線＝過去FY（実績ベース）。Q1〜Q4の達成ペースを過去と比較。"
    ))

# 需給分析
if html_obv:
    body_sections.append(section_html("需給分析 — 出来高 & OBV"))
    obv_insight = ""
    try:
        obv_dir = "上昇" if float(obv.iloc[-1]) > float(obv.iloc[-20]) else "下落"
        obv_ma_txt = "OBV > MA20 → 買い需要が優勢。株価の上昇継続を示唆。" if obv_vs_ma else "OBV < MA20 → 売り圧力継続。短期的に慎重が必要。"
        obv_insight = f"<strong>OBVリーディング</strong>：OBVが{obv_dir}トレンド。{obv_ma_txt}"
    except Exception:
        pass
    body_sections.append(card(
        "OBV（On-Balance Volume）— 需給トレンド",
        "OBVは上昇日の出来高を累積加算・下落日を減算。上昇トレンド＝機関投資家の買い集め示唆。",
        html_obv,
        obv_insight,
    ))

# EV/EBITDA 時系列
if html_ee or html_ee_pctb or html_ee_matrix:
    body_sections.append(section_html("EV/EBITDA 分析"))
if html_ee:
    ee_judge = ('割安ゾーン — EV/EBITDAベースでも買いシグナル点灯' if cur_ee_pctb < 0.2
                else '割高ゾーン — バリュエーション過熱注意' if cur_ee_pctb > 0.8
                else '中立ゾーン — バンド中央付近')
    body_sections.append(card(
        f"EV/EBITDA ボリンジャーバンド（52日±2σ）",
        "Enterprise Value ÷ EBITDA の時系列BBバンド。PERより資本構造の影響を受けにくく、設備産業の比較に有効。",
        html_ee + (html_ee_pctb if html_ee_pctb else ""),
        f"現在 {cur_ee:.1f}x（%B={cur_ee_pctb:.2f}）: {ee_judge}",
    ))
if html_ee_matrix:
    try:
        matrix_insight = f"現在の位置: PER%B={cur_per_pctb:.2f} / EV/EBITDA%B={cur_ee_pctb:.2f}"
    except NameError:
        matrix_insight = f"現在のEV/EBITDA%B={cur_ee_pctb:.2f}"
    body_sections.append(card(
        "PER × EV/EBITDA マトリクス（%B 二軸）",
        "両指標の%B（BBバンド内の位置）を二軸散布図で可視化。左下＝双方割安、右上＝双方割高。色は古（青）→新（黄）の時系列。",
        html_ee_matrix,
        matrix_insight,
    ))

# ピアー
if html_peer:
    body_sections.append(section_html(f"ピアー比較 ({peer_label})"))
    body_sections.append(card(
        "EV/EBITDA ピアー比較",
        "Enterprise Value / EBITDA — 低いほど割安。業界平均との乖離を確認。",
        html_peer,
    ))

# フォワード分析
body_sections.append(section_html("フォワード分析"))
target_surprise_grid = ""
if html_target or html_surprise:
    left2 = ""
    right2 = ""
    if html_target:
        left2 = f'''<div class="cc">
  <div class="ct">アナリスト目標株価</div>
  <div class="cd">{n_analysts}名のアナリストによる目標株価レンジ</div>
  {html_target}
</div>'''
    if html_surprise:
        right2 = f'''<div class="cc">
  <div class="ct">決算EPSサプライズ履歴</div>
  <div class="cd">実績EPS vs 予想EPS（サプライズ率）</div>
  {html_surprise}
</div>'''
    if left2 and right2:
        target_surprise_grid = f'<div class="grid2">{left2}{right2}</div>'
    else:
        target_surprise_grid = left2 + right2
    body_sections.append(target_surprise_grid)

if html_eps_trend:
    body_sections.append(card(
        "EPS修正トレンド",
        "アナリストのEPS予想修正履歴（期間別）",
        html_eps_trend,
        "アナリスト予想の上方修正継続は強気シグナル。",
    ))

# モンテカルロ
body_sections.append(section_html("株価予測（モンテカルロシミュレーション）"))
if html_mc:
    mc_insight = (
        f"30日後に+10%以上となる確率: <strong style='color:{GR}'>{mc_up10_prob:.1f}%</strong>  |  "
        f"HV30ベースのGBM。ドリフトはアナリスト目標株価から逆算。N={N_SIM:,}本。"
        if mc_up10_prob is not None else ""
    )
    body_sections.append(card(
        "モンテカルロ株価シミュレーション（30日）",
        "HV30（30日ヒストリカルボラティリティ）を使用したGBMによる確率分布",
        html_mc,
        mc_insight,
    ))

# 3ヶ月予測・イベントカレンダー
if calendar_events or html_mc90:
    body_sections.append(section_html("3ヶ月株価予測 &amp; イベントカレンダー"))

# イベントテーブル
if calendar_events:
    today_str = pd.Timestamp.today().strftime('%Y-%m-%d')
    rows_html = ""
    for ev in calendar_events:
        ev_dt = pd.Timestamp(ev['date'])
        days_until = (ev_dt - pd.Timestamp.today()).days
        days_txt = f"<span style='color:{ev['color']};font-size:11px'>あと{days_until}日</span>" if days_until >= 0 else f"<span style='color:{DIM};font-size:11px'>{abs(days_until)}日前</span>"
        rows_html += f"""<tr>
  <td style='padding:8px 12px;color:{ev['color']};font-weight:700'>{ev['date']}</td>
  <td style='padding:8px 12px'>{days_txt}</td>
  <td style='padding:8px 12px;font-weight:600'>{ev['label']}</td>
  <td style='padding:8px 12px;color:{DIM}'>{ev.get('impact','')}</td>
  <td style='padding:8px 12px;color:{DIM};font-size:12px'>{ev.get('detail','')}</td>
</tr>"""
    event_table = f"""<div style='overflow-x:auto'>
<table style='width:100%;border-collapse:collapse;font-size:13px'>
  <thead><tr style='border-bottom:1px solid rgba(255,255,255,0.1)'>
    <th style='padding:8px 12px;text-align:left;color:{DIM};font-size:11px;text-transform:uppercase'>日付</th>
    <th style='padding:8px 12px;text-align:left;color:{DIM};font-size:11px;text-transform:uppercase'>残り</th>
    <th style='padding:8px 12px;text-align:left;color:{DIM};font-size:11px;text-transform:uppercase'>イベント</th>
    <th style='padding:8px 12px;text-align:left;color:{DIM};font-size:11px;text-transform:uppercase'>予想影響</th>
    <th style='padding:8px 12px;text-align:left;color:{DIM};font-size:11px;text-transform:uppercase'>詳細</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table></div>"""
    body_sections.append(card("イベントカレンダー（今後3ヶ月）",
        "yfinance calendar より取得。決算発表・配当落ち日など株価に影響するイベント。",
        event_table))

# 3ヶ月MCチャート
if html_mc90:
    # 価格シナリオテーブル
    tbl_rows = ""
    for row in mc90_table:
        lbl, p10, p25, p50, p75, p90 = row
        chg50 = (p50 / cur_price - 1) * 100 if cur_price else 0
        c50 = GR if chg50 > 0 else '#ff4444'
        tbl_rows += f"""<tr style='border-bottom:1px solid rgba(255,255,255,0.05)'>
  <td style='padding:8px 12px;font-weight:600'>{lbl}</td>
  <td style='padding:8px 12px;color:#ff6666'>¥{p10:,.0f}</td>
  <td style='padding:8px 12px;color:{AC3}'>¥{p25:,.0f}</td>
  <td style='padding:8px 12px;color:{AC};font-weight:700'>¥{p50:,.0f} <span style='font-size:11px;color:{c50}'>({chg50:+.1f}%)</span></td>
  <td style='padding:8px 12px;color:{GR}'>¥{p75:,.0f}</td>
  <td style='padding:8px 12px;color:#66ff99'>¥{p90:,.0f}</td>
</tr>"""
    scenario_table = f"""<div style='overflow-x:auto;margin-bottom:16px'>
<table style='width:100%;border-collapse:collapse;font-size:13px'>
  <thead><tr style='border-bottom:1px solid rgba(255,255,255,0.1)'>
    <th style='padding:8px 12px;text-align:left;color:{DIM};font-size:11px;text-transform:uppercase'>期間</th>
    <th style='padding:8px 12px;text-align:left;color:#ff6666;font-size:11px'>悲観(10%ile)</th>
    <th style='padding:8px 12px;text-align:left;color:{AC3};font-size:11px'>弱気(25%ile)</th>
    <th style='padding:8px 12px;text-align:left;color:{AC};font-size:11px'>中央値(50%ile)</th>
    <th style='padding:8px 12px;text-align:left;color:{GR};font-size:11px'>強気(75%ile)</th>
    <th style='padding:8px 12px;text-align:left;color:#66ff99;font-size:11px'>楽観(90%ile)</th>
  </tr></thead>
  <tbody>{tbl_rows}</tbody>
</table></div>"""
    mc90_insight = (f"3ヶ月後に+10%以上となる確率: <strong style='color:{GR}'>{mc90_up10_prob:.1f}%</strong>  "
                    f"| ドリフト: アナリスト目標株価から逆算。HV30ベースGBM N=2,000本。"
                    if mc90_up10_prob is not None else "")
    body_sections.append(card(
        "3ヶ月株価予測（モンテカルロ）",
        "HV30（30日ヒストリカルボラティリティ）を使用したGBMによる90日シミュレーション。縦線＝今後のイベント。",
        scenario_table + html_mc90,
        mc90_insight,
    ))

# ─────────────────────────────────────────
# 分析結果サマリー
# ─────────────────────────────────────────
summary_rows = []

def score_tag(label, value, color, sub=""):
    s = f"<div class='sr-sub'>{sub}</div>" if sub else ""
    return f'''<div class="sr-item">
  <div class="sr-label">{label}</div>
  <div class="sr-value" style="color:{color}">{value}</div>
  {s}
</div>'''

def verdict_tag(text, cls):
    return f'<span class="verdict {cls}">{text}</span>'

# ── バリュエーション評価 ──
val_items = []
# PER
if trailing_pe:
    if trailing_pe < 10:
        val_items.append(score_tag("PER (TTM)", f"{trailing_pe:.1f}倍", GR, "割安水準"))
    elif trailing_pe < 20:
        val_items.append(score_tag("PER (TTM)", f"{trailing_pe:.1f}倍", AC2, "適正水準"))
    else:
        val_items.append(score_tag("PER (TTM)", f"{trailing_pe:.1f}倍", AC3, "やや割高"))
# PBR
if price_to_book:
    if price_to_book < 1.0:
        val_items.append(score_tag("PBR", f"{price_to_book:.2f}倍", GR, "1倍割れ（TSE改革圧力）"))
    elif price_to_book < 2.0:
        val_items.append(score_tag("PBR", f"{price_to_book:.2f}倍", AC2, "適正水準"))
    else:
        val_items.append(score_tag("PBR", f"{price_to_book:.2f}倍", AC3, "やや割高"))
# EV/EBITDA
if has_ev_ebitda:
    if cur_ee_pctb < 0.2:
        val_items.append(score_tag("EV/EBITDA", f"{cur_ee:.1f}x", GR, f"%B={cur_ee_pctb:.2f} 割安ゾーン"))
    elif cur_ee_pctb > 0.8:
        val_items.append(score_tag("EV/EBITDA", f"{cur_ee:.1f}x", AC3, f"%B={cur_ee_pctb:.2f} 割高ゾーン"))
    else:
        val_items.append(score_tag("EV/EBITDA", f"{cur_ee:.1f}x", AC2, f"%B={cur_ee_pctb:.2f} 中立"))
# 配当利回り
if dividend_rate and cur_price:
    dy = dividend_rate / cur_price * 100
    dy_color = GR if dy >= 3.0 else (AC2 if dy >= 1.5 else TX)
    val_items.append(score_tag("配当利回り", f"{dy:.2f}%", dy_color, "高配当" if dy >= 3.0 else ""))

if val_items:
    summary_rows.append(("バリュエーション評価", val_items))

# ── テクニカル評価 ──
tech_items = []
# 株価 vs SMA200
try:
    sma200 = close.rolling(200).mean()
    price_vs_sma200 = float(close.iloc[-1]) / float(sma200.iloc[-1]) - 1
    c = GR if price_vs_sma200 > 0 else AC3
    tech_items.append(score_tag("SMA200比", f"{price_vs_sma200:+.1%}", c,
        "長期上昇トレンド維持" if price_vs_sma200 > 0 else "長期トレンド下方"))
except Exception:
    pass
# OBV
try:
    if len(obv) > 20:
        obv_chg = (float(obv.iloc[-1]) / float(obv.iloc[-20]) - 1) if float(obv.iloc[-20]) != 0 else 0
        obv_c = GR if obv_chg > 0 else AC3
        tech_items.append(score_tag("OBV (20日変化)", f"{obv_chg:+.1%}", obv_c,
            "買い需要優勢" if obv_chg > 0 else "売り圧力継続"))
except Exception:
    pass

if tech_items:
    summary_rows.append(("テクニカル評価", tech_items))

# ── フォワード評価 ──
fwd_items = []
# アップサイド
if upside_pct is not None:
    c = GR if upside_pct > 15 else (AC2 if upside_pct > 0 else AC3)
    fwd_items.append(score_tag("目標株価アップサイド", f"{upside_pct:+.1f}%", c,
        f"平均 ¥{target_mean:,.0f}" if target_mean else ""))
# モンテカルロ
if mc_up10_prob is not None:
    c = GR if mc_up10_prob >= 40 else (AC2 if mc_up10_prob >= 25 else AC3)
    fwd_items.append(score_tag("+10%超の確率 (30日)", f"{mc_up10_prob:.1f}%", c, "モンテカルロ N=2,000"))
# EPS修正
if eps_trend is not None:
    try:
        if '0y' in eps_trend.columns or 'current' in eps_trend.columns:
            col = '0y' if '0y' in eps_trend.columns else 'current'
            if 'consensusEpsEstimate' in eps_trend.index and '7daysAgo' in eps_trend.index:
                cur_est = float(eps_trend.loc['consensusEpsEstimate', col])
                wk1 = float(eps_trend.loc['7daysAgo', col])
                if wk1 and cur_est:
                    chg = (cur_est - wk1) / abs(wk1) * 100
                    c = GR if chg > 0 else (AC3 if chg < -2 else AC2)
                    fwd_items.append(score_tag("EPS修正 (7日)", f"{chg:+.1f}%", c,
                        "上方修正トレンド" if chg > 0 else "下方修正注意"))
    except Exception:
        pass

if fwd_items:
    summary_rows.append(("フォワード評価", fwd_items))

# ── 総合判定 ──
buy_score = 0
total_score = 0
if trailing_pe and trailing_pe < 15: buy_score += 1
if trailing_pe: total_score += 1
if price_to_book and price_to_book < 1.5: buy_score += 1
if price_to_book: total_score += 1
if has_ev_ebitda and cur_ee_pctb < 0.5: buy_score += 1
if has_ev_ebitda: total_score += 1
if upside_pct is not None and upside_pct > 10: buy_score += 1
if upside_pct is not None: total_score += 1
if mc_up10_prob is not None and mc_up10_prob >= 35: buy_score += 1
if mc_up10_prob is not None: total_score += 1
try:
    if len(obv) > 20 and float(obv.iloc[-1]) > float(obv.iloc[-20]): buy_score += 1
    total_score += 1
except Exception:
    pass

if total_score > 0:
    ratio = buy_score / total_score
    if ratio >= 0.6:
        verdict_cls = "buy"
        verdict_text = "買い / 中期強気"
        verdict_color = GR
        verdict_comment = "複数指標がポジティブを示唆。押し目での買い検討。"
    elif ratio >= 0.4:
        verdict_cls = "watch"
        verdict_text = "様子見 / 中立"
        verdict_color = AC2
        verdict_comment = "強弱混在。トレンド確認後のエントリーが無難。"
    else:
        verdict_cls = "pass"
        verdict_text = "見送り"
        verdict_color = "#ff4444"
        verdict_comment = "複数指標が弱気を示唆。リスク管理を優先。"
else:
    verdict_cls = "watch"
    verdict_text = "データ不足"
    verdict_color = AC2
    verdict_comment = ""

# サマリーHTMLを構築
summary_html_parts = [section_html("分析結果サマリー")]
summary_html_parts.append(f'<div class="summary-verdict">{verdict_tag(verdict_text, verdict_cls)}<span class="sv-comment">{verdict_comment}</span></div>')

for title, items in summary_rows:
    items_html = "".join(items)
    summary_html_parts.append(f'''<div class="cc">
  <div class="ct">{title}</div>
  <div class="sr-grid">{items_html}</div>
</div>''')

body_sections.extend(summary_html_parts)

body_html = "\n".join(body_sections)

# ─── HTML テンプレート ───
html_output = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{short_name} ({TICKER_SYM}) — クオンツ分析レポート</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:{BG};--card:{CARD};--ac:{AC};--ac2:{AC2};--ac3:{AC3};--gr:{GR};--tx:{TX};--dim:{DIM}}}
body{{background:var(--bg);color:var(--tx);font-family:'Noto Sans JP','Helvetica Neue',Arial,sans-serif;font-size:14px;line-height:1.6;min-height:100vh}}
a{{color:var(--ac);text-decoration:none}}
.wrap{{max-width:1240px;margin:0 auto;padding:32px 24px}}
.header{{border-bottom:1px solid rgba(255,255,255,.06);padding-bottom:24px;margin-bottom:28px}}
.ticker-badge{{display:inline-block;background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.3);border-radius:6px;padding:3px 10px;font-size:12px;font-weight:700;color:var(--ac);margin-bottom:8px}}
.company-name{{font-size:26px;font-weight:700;margin-bottom:4px}}
.price-big{{font-size:36px;font-weight:700;color:var(--ac);margin-right:12px}}
.update-time{{font-size:11px;color:var(--dim);margin-top:6px}}
.sec{{font-size:12px;color:var(--dim);text-transform:uppercase;letter-spacing:2px;margin:32px 0 16px;padding-bottom:8px;border-bottom:1px solid rgba(255,255,255,.06)}}
.cc{{background:var(--card);border:1px solid rgba(255,255,255,.07);border-radius:14px;padding:20px;margin-bottom:20px}}
.ct{{font-size:15px;font-weight:700;margin-bottom:4px}}
.cd{{font-size:12px;color:var(--dim);margin-bottom:12px}}
.ib{{background:rgba(0,212,255,.06);border-left:3px solid var(--ac);padding:10px 14px;border-radius:0 8px 8px 0;font-size:13px;margin-top:12px;color:var(--dim)}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
.kpi-row{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:28px}}
.kpi{{background:var(--card);border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:16px 20px;min-width:130px;flex:1}}
.kl{{font-size:11px;color:var(--dim);margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px}}
.kv{{font-size:22px;font-weight:700;color:var(--tx)}}
.ks{{font-size:11px;color:var(--dim);margin-top:3px}}
.verdict{{display:inline-block;font-size:11px;font-weight:700;padding:3px 10px;border-radius:5px;margin-bottom:12px}}
.verdict.buy{{background:rgba(0,255,136,.12);color:var(--gr);border:1px solid rgba(0,255,136,.3)}}
.verdict.watch{{background:rgba(255,215,0,.1);color:var(--ac2);border:1px solid rgba(255,215,0,.3)}}
.tag{{background:rgba(255,255,255,.06);border-radius:4px;padding:2px 8px;font-size:10px;margin-right:6px}}
footer{{margin-top:60px;padding-top:24px;border-top:1px solid rgba(255,255,255,.06);font-size:11px;color:var(--dim);text-align:center;padding-bottom:32px}}
.summary-verdict{{margin-bottom:20px;display:flex;align-items:center;gap:14px}}
.sv-comment{{font-size:13px;color:var(--dim)}}
.sr-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-top:8px}}
.sr-item{{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:10px;padding:12px 16px}}
.sr-label{{font-size:11px;color:var(--dim);margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px}}
.sr-value{{font-size:20px;font-weight:700}}
.sr-sub{{font-size:11px;color:var(--dim);margin-top:2px}}
@media(max-width:768px){{.grid2{{grid-template-columns:1fr}}.price-big{{font-size:28px}}.company-name{{font-size:20px}}}}
</style>
</head>
<body>
<div class="wrap">

  <!-- ヘッダー -->
  <div class="header">
    <div class="ticker-badge">{TICKER_SYM}</div>
    <div class="company-name">{short_name}</div>
    <div>
      <span class="price-big">{kpi_price}</span>
      <span style="color:var(--dim);font-size:13px">JPY</span>
    </div>
    <div class="update-time">更新: {now_str} &nbsp;|&nbsp; データソース: yfinance &nbsp;|&nbsp; MC: {N_SIM:,}本 &nbsp;|&nbsp; BB期間: {BB_WIN}日</div>
  </div>

  <!-- KPIカード -->
  <div class="kpi-row">
    {kpi_row}
  </div>

  <!-- 本文 -->
  {body_html}

  <footer>
    本レポートは情報提供目的のみであり、投資助言ではありません。<br>
    Generated by generate_report_jp.py &nbsp;|&nbsp; {now_str}
  </footer>

</div>
</body>
</html>
"""

# ─────────────────────────────────────────
# Section 4: ファイル書き出し
# ─────────────────────────────────────────
out_dir = f"/home/like_rapid/Quant_analysis/reports/{TICKER_SYM}"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "index.html")

with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_output)

print(f"\n[DONE] Report saved to: {out_path}")
print(f"       File size: {os.path.getsize(out_path)/1024:.1f} KB")
