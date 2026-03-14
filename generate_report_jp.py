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
}
PEER_LABEL_MAP = {
    "7203.T": "国内自動車大手",
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
try:
    if not q_inc.empty:
        cols = sorted(q_inc.columns, key=lambda c: pd.to_datetime(c))
        # 最新12四半期まで
        cols = cols[-12:]
        labels = []
        rev_vals = []
        op_vals  = []
        ni_vals  = []
        for col in cols:
            dt = pd.to_datetime(col)
            label = f"{str(dt.year)[2:]}Q{(dt.month-1)//3+1}"
            labels.append(label)
            # 売上高
            rev = None
            for k in ["Total Revenue", "Revenue"]:
                if k in q_inc.index and pd.notna(q_inc.loc[k, col]):
                    rev = float(q_inc.loc[k, col]) / 1e12  # 兆円
                    break
            rev_vals.append(rev)
            # 営業利益
            op = None
            for k in ["Operating Income", "Operating Revenue"]:
                if k in q_inc.index and pd.notna(q_inc.loc[k, col]):
                    op = float(q_inc.loc[k, col]) / 1e11  # 千億円
                    break
            op_vals.append(op)
            # 純利益
            ni = None
            for k in ["Net Income", "Net Income Common Stockholders"]:
                if k in q_inc.index and pd.notna(q_inc.loc[k, col]):
                    ni = float(q_inc.loc[k, col]) / 1e11
                    break
            ni_vals.append(ni)

        fig_earnings = go.Figure()
        if any(v is not None for v in rev_vals):
            fig_earnings.add_trace(go.Bar(
                x=labels, y=rev_vals, name="売上高（兆円）",
                marker_color=AC,
            ))
        if any(v is not None for v in op_vals):
            fig_earnings.add_trace(go.Bar(
                x=labels, y=op_vals, name="営業利益（千億円）",
                marker_color=GR,
            ))
        if any(v is not None for v in ni_vals):
            fig_earnings.add_trace(go.Bar(
                x=labels, y=ni_vals, name="純利益（千億円）",
                marker_color=AC3,
            ))
        fig_earnings.update_layout(barmode="group")
        apply_layout(fig_earnings, height=360, yaxis_title="金額（兆/千億円）")
        html_earnings = safe_html(fig_earnings, "chart-earnings")
        print("  Chart 6 (earnings) OK")
    else:
        print("  Chart 6 (earnings): skipped (no quarterly data)")
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

# ─────────────────────────────────────────
# Section 3: HTML生成
# ─────────────────────────────────────────
print("[3] Generating HTML ...")

now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M JST")

# KPI 計算値
kpi_price = f"¥{cur_price:,.0f}" if cur_price else "N/A"
kpi_per   = fmt_num(trailing_pe, 1, "倍") if trailing_pe else (fmt_num(forward_pe, 1, "倍(FWD)") if forward_pe else "N/A")
kpi_pbr   = fmt_num(price_to_book, 2, "倍") if price_to_book else "N/A"
kpi_div   = f"{dividend_yield*100:.2f}%" if dividend_yield else "N/A"
kpi_roe   = fmt_pct(roe) if roe else "N/A"
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
