"""
統合クオンツ分析レポート ジェネレーター
lly_report_v2.py + lly_forecast_report.py を統合した1本のHTMLを生成する。

使い方:
    python3 generate_report.py [TICKER]   # デフォルト: LLY
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ─── 設定 ────────────────────────────────────────────────────────────
TICKER_SYM = sys.argv[1] if len(sys.argv) > 1 else "LLY"
USD_JPY    = 158.94
OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", TICKER_SYM)
OUT_FILE   = os.path.join(OUT_DIR, "index.html")

os.makedirs(OUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# 1. データ取得
# ════════════════════════════════════════════════════════════════════════
print(f"[{TICKER_SYM}] データ取得中...")
ticker_obj = yf.Ticker(TICKER_SYM)

# 3年分（v2用）
price_3y = ticker_obj.history(period="3y", interval="1d")
price_3y.index = pd.to_datetime(price_3y.index).tz_localize(None)
price_3y = price_3y[['Close', 'Volume', 'High', 'Low']].dropna()
close_3y = price_3y['Close']

# 1年分（forecast用）
price_1y = ticker_obj.history(period="1y", interval="1d")
price_1y.index = pd.to_datetime(price_1y.index).tz_localize(None)
close_1y = price_1y['Close']
volume_1y = price_1y['Volume']

info = ticker_obj.info
cur_price = float(close_3y.iloc[-1])
print(f"  現在株価: ${cur_price:.2f}")

# ════════════════════════════════════════════════════════════════════════
# 2. 年次 EPS 取得（PER BB 長期化用）
# ════════════════════════════════════════════════════════════════════════
print("年次 EPS 取得...")
annual_eps_dict = {}

try:
    ann = ticker_obj.income_stmt
    for key in ['Diluted EPS', 'Basic EPS', 'Diluted Normalized EPS', 'Normalized Diluted EPS']:
        if key in ann.index:
            eps_ann = ann.loc[key].dropna()
            for col, val in eps_ann.items():
                yr = pd.to_datetime(col).tz_localize(None)
                annual_eps_dict[yr] = abs(float(val))
            print(f"  年次EPS ({key}): 取得済み {len(annual_eps_dict)}件")
            break

    if not annual_eps_dict:
        ni = shares = None
        for k in ['Net Income', 'Net Income Common Stockholders']:
            if k in ann.index:
                ni = ann.loc[k]
                break
        for k in ['Diluted Average Shares', 'Ordinary Shares Number']:
            if k in ann.index:
                shares = ann.loc[k]
                break
        if ni is not None and shares is not None:
            for col in ni.index:
                yr = pd.to_datetime(col).tz_localize(None)
                if shares.get(col, 0):
                    annual_eps_dict[yr] = abs(float(ni[col]) / float(shares[col]))
            print(f"  年次EPS (Net/Shares): 取得済み {len(annual_eps_dict)}件")
except Exception as e:
    print(f"  年次EPS取得エラー: {e}")

# ════════════════════════════════════════════════════════════════════════
# 3. 四半期 EPS 取得
# ════════════════════════════════════════════════════════════════════════
print("四半期 EPS 取得...")
eps_q = pd.Series(dtype=float)

try:
    q = ticker_obj.quarterly_income_stmt
    ni = shares = None
    for k in ['Net Income', 'Net Income Common Stockholders']:
        if k in q.index:
            ni = q.loc[k]
            break
    for k in ['Diluted Average Shares', 'Ordinary Shares Number']:
        if k in q.index:
            shares = q.loc[k]
            break
    if ni is not None and shares is not None:
        eps_q_raw = {}
        for col in ni.index:
            dt = pd.to_datetime(col).tz_localize(None)
            if shares.get(col, 0):
                eps_q_raw[dt] = abs(float(ni[col]) / float(shares[col]))
        eps_q = pd.Series(eps_q_raw).sort_index()
        print(f"  四半期EPS: {len(eps_q)}件")
except Exception as e:
    print(f"  四半期EPS取得エラー: {e}")

if len(eps_q) == 0:
    try:
        eh = ticker_obj.earnings_history
        if eh is not None and 'epsActual' in eh.columns:
            for idx, row in eh.iterrows():
                dt = pd.to_datetime(idx).tz_localize(None)
                eps_q[dt] = abs(float(row['epsActual']))
            eps_q = eps_q.sort_index()
            print(f"  四半期EPS (earnings_history): {len(eps_q)}件")
    except Exception as e:
        print(f"  earnings_history エラー: {e}")

# ════════════════════════════════════════════════════════════════════════
# 4. TTM PER 系列構築
# ════════════════════════════════════════════════════════════════════════
print("TTM PER 系列構築...")

def build_ttm_eps_series(close_index, eps_q, annual_eps_dict):
    result = {}
    ann_dates_sorted = sorted(annual_eps_dict.keys()) if annual_eps_dict else []
    for date in close_index:
        past_q = eps_q[eps_q.index <= date]
        if len(past_q) >= 4:
            last_4 = past_q.iloc[-4:]
            valid_count = last_4.notna().sum()
            if valid_count >= 3:
                ttm = last_4.sum()
                if ttm > 0:
                    result[date] = ttm
                    continue
        if len(ann_dates_sorted) >= 2:
            prev_dates = [d for d in ann_dates_sorted if d <= date]
            next_dates = [d for d in ann_dates_sorted if d >  date]
            if prev_dates and next_dates:
                d0, d1 = max(prev_dates), min(next_dates)
                e0, e1 = annual_eps_dict[d0], annual_eps_dict[d1]
                ratio = (date - d0).days / max((d1 - d0).days, 1)
                ttm = e0 + (e1 - e0) * ratio
                if ttm > 0:
                    result[date] = ttm
                    continue
            elif prev_dates:
                ttm = annual_eps_dict[max(prev_dates)]
                if ttm > 0:
                    result[date] = ttm
    return pd.Series(result)

ttm_eps  = build_ttm_eps_series(close_3y.index, eps_q, annual_eps_dict)
per_all  = (close_3y / ttm_eps).dropna()
per_all  = per_all[per_all > 0]
per_all  = per_all[per_all < 200]
print(f"  TTM PER 系列: {len(per_all)} 行")

# ─── PER ボリンジャーバンド
n = len(per_all)
if n >= 252:
    bb_win = 52
elif n >= 100:
    bb_win = 20
else:
    bb_win = max(10, n // 3)

per_df           = per_all.to_frame('per')
per_df['ma']     = per_df['per'].rolling(bb_win).mean()
per_df['std']    = per_df['per'].rolling(bb_win).std()
per_df['upper']  = per_df['ma'] + 2 * per_df['std']
per_df['lower']  = per_df['ma'] - 2 * per_df['std']
per_df['pct_b']  = ((per_df['per'] - per_df['lower'])
                    / (per_df['upper'] - per_df['lower'])).clip(-0.5, 1.5)
per_plot = per_df.dropna()

cur_per   = float(per_plot['per'].iloc[-1])
cur_ma    = float(per_plot['ma'].iloc[-1])
cur_upper = float(per_plot['upper'].iloc[-1])
cur_lower = float(per_plot['lower'].iloc[-1])
cur_pctb  = float(per_plot['pct_b'].iloc[-1])

# ════════════════════════════════════════════════════════════════════════
# 5. テクニカル指標計算
# ════════════════════════════════════════════════════════════════════════
def calc_macd(s, f=12, sl=26, sig=9):
    m      = s.ewm(span=f, adjust=False).mean() - s.ewm(span=sl, adjust=False).mean()
    signal = m.ewm(span=sig, adjust=False).mean()
    return m, signal, m - signal

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100 - 100 / (1 + g / l)

# v2 用（3年データ）
macd_line, sig_line, hist = calc_macd(close_3y)
rsi_series = calc_rsi(close_3y)
sma20_3y   = close_3y.rolling(20).mean()
sma50_3y   = close_3y.rolling(50).mean()
sma200_3y  = close_3y.rolling(200).mean()

cur_rsi   = float(rsi_series.iloc[-1])
high_52w  = float(close_3y.iloc[-252:].max())
low_52w   = float(close_3y.iloc[-252:].min())
cur_macd  = float(macd_line.iloc[-1])
cur_sig   = float(sig_line.iloc[-1])
cur_hist  = float(hist.iloc[-1])

# forecast 用（1年データ）
sma50_1y  = close_1y.rolling(50).mean()
sma200_1y = close_1y.rolling(200).mean()

high_1y = price_1y['High']
low_1y  = price_1y['Low']
tr_1y   = pd.concat([high_1y - low_1y,
                     (high_1y - close_1y.shift()).abs(),
                     (low_1y  - close_1y.shift()).abs()], axis=1).max(axis=1)
atr14   = float(tr_1y.rolling(14).mean().iloc[-1])

log_ret = np.log(close_1y / close_1y.shift())
hv30    = float(log_ret.rolling(30).std().iloc[-1] * np.sqrt(252) * 100)

recent      = price_1y.iloc[-126:]
roll_high   = recent['High'].rolling(10, center=True).max()
roll_low    = recent['Low'].rolling(10, center=True).min()
supports    = sorted(list(set([round(v, -1) for v in roll_low.dropna().unique() if v < cur_price * 0.98])))[-5:]
resistances = sorted(list(set([round(v, -1) for v in roll_high.dropna().unique() if v > cur_price * 1.02])))[:5]

print(f"  ATR14: ${atr14:.2f}  HV30: {hv30:.1f}%")

# ════════════════════════════════════════════════════════════════════════
# 6. オプション・需給データ
# ════════════════════════════════════════════════════════════════════════
print("オプション / 需給データ取得中...")
iv_atm         = None
put_call_ratio = None
try:
    exps = ticker_obj.options
    if exps:
        target_date = datetime.now() + timedelta(days=35)
        best_exp    = min(exps, key=lambda d: abs(
            (datetime.strptime(d, '%Y-%m-%d') - target_date).days))
        chain      = ticker_obj.option_chain(best_exp)
        calls      = chain.calls
        puts       = chain.puts
        atm_calls  = calls[abs(calls['strike'] - cur_price) < 30]
        if not atm_calls.empty:
            iv_atm = float(atm_calls['impliedVolatility'].median() * 100)
        total_put_vol  = puts['volume'].sum()
        total_call_vol = calls['volume'].sum()
        if total_call_vol > 0:
            put_call_ratio = float(total_put_vol / total_call_vol)
        print(f"  IV(ATM): {iv_atm:.1f}%" if iv_atm else "  IV取得失敗")
except Exception as e:
    print(f"  オプションエラー: {e}")

short_ratio  = info.get('shortRatio', None)
short_pct    = info.get('shortPercentOfFloat', None)
float_shares = info.get('floatShares', None)
shares_out   = info.get('sharesOutstanding', None)

target_mean  = info.get('targetMeanPrice', 1214)
target_high  = info.get('targetHighPrice', 1500)
target_low   = info.get('targetLowPrice', 800)
target_med   = info.get('targetMedianPrice', 1200)
num_analysts = info.get('numberOfAnalystOpinions', 30)
rec_mean     = info.get('recommendationMean', 1.8)

# ─── EV/EBITDA データ取得 ────────────────────────────────
print("EV/EBITDA データ取得中...")

# 四半期 EBITDA 取得
ebitda_q = pd.Series(dtype=float)
try:
    qinc = ticker_obj.quarterly_income_stmt
    ebitda_keys = ['EBITDA', 'Normalized EBITDA']
    for k in ebitda_keys:
        if k in qinc.index:
            raw = qinc.loc[k].dropna()
            ebitda_q = pd.Series({pd.to_datetime(c).tz_localize(None): abs(float(v)) for c, v in raw.items()}).sort_index()
            break
    if ebitda_q.empty:
        qcf = ticker_obj.quarterly_cashflow
        da_keys = ['Depreciation And Amortization', 'Depreciation Amortization Depletion']
        oi_keys = ['Operating Income', 'EBIT']
        oi = da = None
        for k in oi_keys:
            if k in qinc.index: oi = qinc.loc[k]; break
        for k in da_keys:
            if k in qcf.index: da = qcf.loc[k]; break
        if oi is not None and da is not None:
            for col in oi.index:
                dt = pd.to_datetime(col).tz_localize(None)
                o_val = oi.get(col, np.nan)
                d_val = da.get(col, np.nan)
                if pd.notna(o_val) and pd.notna(d_val):
                    ebitda_q[dt] = abs(float(o_val)) + abs(float(d_val))
            ebitda_q = ebitda_q.sort_index()
    print(f"  四半期EBITDA: {len(ebitda_q)}件 {dict(list(ebitda_q.items())[:3])}")
except Exception as e:
    print(f"  EBITDA取得エラー: {e}")

# 年次 EBITDA（補間用）
ebitda_annual = {}
try:
    ainc = ticker_obj.income_stmt
    ebitda_keys = ['EBITDA', 'Normalized EBITDA']
    for k in ebitda_keys:
        if k in ainc.index:
            for col, val in ainc.loc[k].dropna().items():
                ebitda_annual[pd.to_datetime(col).tz_localize(None)] = abs(float(val))
            break
    if not ebitda_annual:
        acf = ticker_obj.cashflow
        da_keys = ['Depreciation And Amortization', 'Depreciation Amortization Depletion']
        oi_keys = ['Operating Income', 'EBIT']
        oi = da = None
        for k in oi_keys:
            if k in ainc.index: oi = ainc.loc[k]; break
        for k in da_keys:
            if k in acf.index: da = acf.loc[k]; break
        if oi is not None and da is not None:
            for col in oi.index:
                dt = pd.to_datetime(col).tz_localize(None)
                o_val = oi.get(col, np.nan)
                d_val = da.get(col, np.nan)
                if pd.notna(o_val) and pd.notna(d_val):
                    ebitda_annual[dt] = abs(float(o_val)) + abs(float(d_val))
    print(f"  年次EBITDA: {ebitda_annual}")
except Exception as e:
    print(f"  年次EBITDA取得エラー: {e}")

# 純負債（四半期バランスシートから）
net_debt_q = {}
try:
    qbs = ticker_obj.quarterly_balance_sheet
    debt_keys = ['Total Debt', 'Long Term Debt', 'Current Debt']
    cash_keys = ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments']
    for col in qbs.columns:
        dt = pd.to_datetime(col).tz_localize(None)
        debt = cash = 0
        for k in debt_keys:
            if k in qbs.index and pd.notna(qbs.loc[k, col]):
                debt = abs(float(qbs.loc[k, col])); break
        for k in cash_keys:
            if k in qbs.index and pd.notna(qbs.loc[k, col]):
                cash = abs(float(qbs.loc[k, col])); break
        net_debt_q[dt] = debt - cash
    net_debt_q = dict(sorted(net_debt_q.items()))
    print(f"  純負債 (直近): ${list(net_debt_q.values())[-1]/1e9:.1f}B")
except Exception as e:
    print(f"  純負債取得エラー: {e}")

# TTM EBITDA系列を構築
def build_ttm_ebitda(close_index, ebitda_q, ebitda_annual):
    result = {}
    ann_dates = sorted(ebitda_annual.keys()) if ebitda_annual else []
    for date in close_index:
        past_q = ebitda_q[ebitda_q.index <= date]
        if len(past_q) >= 4:
            last_4 = past_q.iloc[-4:]
            if last_4.notna().sum() >= 3:
                ttm = last_4.sum()
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

ttm_ebitda = build_ttm_ebitda(close_3y.index, ebitda_q, ebitda_annual)

# 日次EV = 時価総額 + 純負債（四半期更新値を前向き補完）
ev_shares_out = info.get('sharesOutstanding', info.get('impliedSharesOutstanding', None))
if ev_shares_out:
    mktcap_series = close_3y * ev_shares_out
else:
    mktcap_series = pd.Series(dtype=float)

if net_debt_q and not mktcap_series.empty:
    nd_series = pd.Series(net_debt_q).reindex(close_3y.index, method='ffill')
    ev_series = mktcap_series + nd_series
else:
    ev_series = pd.Series(dtype=float)

# EV/EBITDA 系列
if not ev_series.empty and not ttm_ebitda.empty:
    ev_ebitda_all = (ev_series / ttm_ebitda).dropna()
    ev_ebitda_all = ev_ebitda_all[(ev_ebitda_all > 0) & (ev_ebitda_all < 200)]
else:
    ev_ebitda_all = pd.Series(dtype=float)

# EV/EBITDA BB
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
    print("  EV/EBITDAデータ不足")

# 機関投資家
try:
    inst_holders = ticker_obj.institutional_holders
    top_inst = inst_holders.head(8) if inst_holders is not None else pd.DataFrame()
except:
    top_inst = pd.DataFrame()

# ════════════════════════════════════════════════════════════════════════
# 7. イベントスケジュール
# ════════════════════════════════════════════════════════════════════════
events = [
    {
        'date': '2026-03-15', 'label': 'JPモルガン医療フォーラム',
        'type': 'conf', 'impact': '+1〜3%', 'prob': '中', 'color': '#00d4ff',
        'detail': 'Lilly CEOによるパイプライン・ガイダンス再確認の場。新規カタリストは限定的だが、機関投資家へのIRとして需給に好影響。',
    },
    {
        'date': '2026-04-16', 'label': 'Q1 2026 決算発表（予定）',
        'type': 'earnings', 'impact': '±8〜12%', 'prob': '高', 'color': '#ffd700',
        'detail': 'Mounjaro・Zepboundの売上続伸が鍵。Q4 2025は両製品で+100%超のYoY成長。アナリストコンセンサスを上回るかどうかで大きく動く。過去4決算の平均株価反応は±9%。',
    },
    {
        'date': '2026-05-15', 'label': 'Orforglipron FDA PDUFA（Q2目標）',
        'type': 'fda', 'impact': '+10〜20% or −15%', 'prob': '最重要', 'color': '#ff6b35',
        'detail': '経口GLP-1薬（飲み薬）の初承認。注射不要で潜在患者層を大幅拡大。承認なら株価+15%以上の可能性。再延期・否決なら−10〜15%。2026年最大のバイナリーイベント。',
    },
    {
        'date': '2026-06-20', 'label': 'ADA 学術年次総会（糖尿病学会）',
        'type': 'conf', 'impact': '+2〜5%', 'prob': '中', 'color': '#00d4ff',
        'detail': '糖尿病・肥満症領域の最重要学会。Orforglipron承認後なら実臨床データの追加公表で買い継続。Zepboundの心血管データ（SURMOUNT-MMO）の追加解析も注目。',
    },
    {
        'date': '2026-07-22', 'label': 'Q2 2026 決算発表（予定）',
        'type': 'earnings', 'impact': '±6〜10%', 'prob': '高', 'color': '#ffd700',
        'detail': 'Orforglipron承認後初の決算。初期売上・2026年ガイダンス引き上げが焦点。製造能力の増強状況も確認。アナリスト予想 EPS $3.8〜4.2。',
    },
    {
        'date': '2026-09-10', 'label': 'ESMO 2026（腫瘍学会）',
        'type': 'conf', 'impact': '+1〜4%', 'prob': '低〜中', 'color': '#00d4ff',
        'detail': 'LLYの癌領域（Verzenio等）に関するデータ発表。GLP-1一本足からの多角化の観点で注目度は高まりつつある。',
    },
    {
        'date': '2026-10-20', 'label': 'Q3 2026 決算発表（予定）',
        'type': 'earnings', 'impact': '±5〜9%', 'prob': '高', 'color': '#ffd700',
        'detail': 'Orforglipron通年寄与が見え始める決算。2027年ガイダンスが株価のネクストレベルを決める。機関投資家の年末ポジション調整とも重なる。',
    },
]

# ════════════════════════════════════════════════════════════════════════
# 8. モンテカルロシミュレーション
# ════════════════════════════════════════════════════════════════════════
print("モンテカルロシミュレーション...")
np.random.seed(42)
N_SIM   = 2000
N_DAYS  = 252

daily_vol    = hv30 / 100 / np.sqrt(252)
annual_drift = np.log(target_mean / cur_price)
daily_drift  = annual_drift / N_DAYS

paths = np.zeros((N_SIM, N_DAYS + 1))
paths[:, 0] = cur_price
for t in range(1, N_DAYS + 1):
    z           = np.random.standard_normal(N_SIM)
    paths[:, t] = paths[:, t-1] * np.exp(
        (daily_drift - 0.5 * daily_vol**2) + daily_vol * z
    )

pct_5  = float(np.percentile(paths[:, -1], 5))
pct_25 = float(np.percentile(paths[:, -1], 25))
pct_50 = float(np.percentile(paths[:, -1], 50))
pct_75 = float(np.percentile(paths[:, -1], 75))
pct_95 = float(np.percentile(paths[:, -1], 95))

days_1m = 21
days_3m = 63
pct_1m  = np.percentile(paths[:, days_1m], [10, 25, 50, 75, 90])
pct_3m  = np.percentile(paths[:, days_3m], [10, 25, 50, 75, 90])

print(f"  1年後中央値: ${pct_50:.0f}  1ヶ月後中央値: ${pct_1m[2]:.0f}")

# ════════════════════════════════════════════════════════════════════════
# 9. チャート生成
# ════════════════════════════════════════════════════════════════════════
print("チャート生成中...")

# ─── fig1: 株価 + MACD（lly_report_v2.py fig1）
fig1_v2 = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.55, 0.18, 0.27],
    vertical_spacing=0.04,
    subplot_titles=('株価 + 移動平均', '出来高', 'MACD (12, 26, 9)')
)

fig1_v2.add_trace(go.Scatter(
    x=list(close_3y.index), y=list(close_3y),
    name='終値', line=dict(color='#00d4ff', width=2),
    hovertemplate='$%{y:.2f}<extra></extra>'), row=1, col=1)
fig1_v2.add_trace(go.Scatter(
    x=list(sma20_3y.index), y=list(sma20_3y),
    name='SMA20', line=dict(color='#ffd700', width=1.2, dash='dash')), row=1, col=1)
fig1_v2.add_trace(go.Scatter(
    x=list(sma50_3y.index), y=list(sma50_3y),
    name='SMA50', line=dict(color='#ff9944', width=1.5)), row=1, col=1)
fig1_v2.add_trace(go.Scatter(
    x=list(sma200_3y.index), y=list(sma200_3y),
    name='SMA200', line=dict(color='#ff4444', width=1.5, dash='dot')), row=1, col=1)

vol_colors_3y = list(np.where(close_3y >= close_3y.shift(1), '#00d4ff', '#ff4444'))
fig1_v2.add_trace(go.Bar(
    x=list(price_3y.index), y=list(price_3y['Volume']),
    name='出来高', marker_color=vol_colors_3y, showlegend=False,
    hovertemplate='%{y:,.0f}<extra></extra>'), row=2, col=1)

hist_colors = list(np.where(hist >= 0, '#00d4ff', '#ff4444'))
fig1_v2.add_trace(go.Bar(
    x=list(hist.index), y=list(hist),
    name='ヒストグラム', marker_color=hist_colors, showlegend=False), row=3, col=1)
fig1_v2.add_trace(go.Scatter(
    x=list(macd_line.index), y=list(macd_line),
    name='MACD', line=dict(color='#00d4ff', width=2)), row=3, col=1)
fig1_v2.add_trace(go.Scatter(
    x=list(sig_line.index), y=list(sig_line),
    name='シグナル', line=dict(color='#ffd700', width=1.5)), row=3, col=1)
fig1_v2.add_hline(y=0, line=dict(color='white', width=0.5, dash='dash'), row=3, col=1)

price_min = float(close_3y.min()) * 0.92
price_max = float(close_3y.max()) * 1.05
fig1_v2.update_yaxes(range=[price_min, price_max],
                     tickformat='$,.0f', title_text='USD', row=1, col=1)
fig1_v2.update_yaxes(title_text='出来高', row=2, col=1)
macd_abs_max = max(
    float(abs(macd_line.dropna()).max()),
    float(abs(sig_line.dropna()).max()),
    float(abs(hist.dropna()).max())
) * 1.3
fig1_v2.update_yaxes(range=[-macd_abs_max, macd_abs_max], title_text='MACD', row=3, col=1)

fig1_v2.update_layout(
    height=720,
    paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
    font=dict(color='white', family='Arial'),
    legend=dict(orientation='h', y=1.03, bgcolor='rgba(0,0,0,0)', font_size=12),
    margin=dict(l=70, r=20, t=50, b=20),
    xaxis3=dict(
        rangeselector=dict(
            buttons=[
                dict(count=3,  label='3M',  step='month'),
                dict(count=6,  label='6M',  step='month'),
                dict(count=1,  label='1Y',  step='year'),
                dict(count=2,  label='2Y',  step='year'),
                dict(step='all', label='全期間'),
            ],
            bgcolor='#1a1a2e', activecolor='#00d4ff',
            font=dict(color='white')
        ),
        rangeslider=dict(visible=False)
    )
)
for r in range(1, 4):
    fig1_v2.update_xaxes(gridcolor='#1e1e30', row=r, col=1)
    fig1_v2.update_yaxes(gridcolor='#1e1e30', row=r, col=1)

# ─── fig2: PER BB（lly_report_v2.py fig2）
fig2_v2 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.72, 0.28],
    vertical_spacing=0.06,
    subplot_titles=(f'PER ボリンジャーバンド（{bb_win}日移動平均, ±2σ）',
                    '%B（BBバンド内の位置）')
)

fig2_v2.add_trace(go.Scatter(
    x=list(per_plot.index), y=list(per_plot['upper']),
    mode='lines', line=dict(color='rgba(255,107,53,0.4)', width=1),
    name=f'BB上限(+2σ) {cur_upper:.1f}x', showlegend=True
), row=1, col=1)
fig2_v2.add_trace(go.Scatter(
    x=list(per_plot.index), y=list(per_plot['lower']),
    mode='lines', line=dict(color='rgba(255,107,53,0.4)', width=1),
    fill='tonexty', fillcolor='rgba(255,107,53,0.18)',
    name=f'BB下限(-2σ) {cur_lower:.1f}x', showlegend=True
), row=1, col=1)
fig2_v2.add_trace(go.Scatter(
    x=list(per_plot.index), y=list(per_plot['ma']),
    mode='lines', line=dict(color='#ffd700', width=1.5, dash='dash'),
    name=f'BB中央({bb_win}日MA) {cur_ma:.1f}x'
), row=1, col=1)
fig2_v2.add_trace(go.Scatter(
    x=list(per_plot.index), y=list(per_plot['per']),
    mode='lines', line=dict(color='#00d4ff', width=2.5),
    name=f'実績PER {cur_per:.1f}x',
    hovertemplate='PER: %{y:.1f}x<extra></extra>'
), row=1, col=1)

last_date = per_plot.index[-1]
fig2_v2.add_annotation(
    x=last_date, y=cur_per,
    text=f'現在 {cur_per:.1f}x',
    showarrow=True, arrowhead=2, arrowcolor='#00d4ff',
    ax=-60, ay=-30,
    font=dict(color='white', size=12),
    bgcolor='#1a1a2e', bordercolor='#00d4ff', borderwidth=1,
    row=1, col=1
)

fig2_v2.add_trace(go.Scatter(
    x=list(per_plot.index), y=list(per_plot['pct_b']),
    mode='lines', line=dict(color='#00d4ff', width=2),
    fill='tozeroy', fillcolor='rgba(0,212,255,0.12)',
    name=f'%B = {cur_pctb:.2f}',
    hovertemplate='%%B: %{y:.3f}<extra></extra>'
), row=2, col=1)

for lvl, col, lbl in [(1.0, '#ff4444', ''), (0.8, '#ff9944', ''),
                       (0.5, '#ffd700', '中央'), (0.2, '#44ff88', ''), (0.0, '#44ff88', '')]:
    fig2_v2.add_hline(y=lvl, line=dict(color=col, width=0.8, dash='dash'), row=2, col=1)
fig2_v2.add_hrect(y0=0.8, y1=1.2, fillcolor='rgba(255,68,68,0.08)',
                  line_width=0, row=2, col=1,
                  annotation_text='割高', annotation_font_color='#ff4444',
                  annotation_position='right')
fig2_v2.add_hrect(y0=-0.2, y1=0.2, fillcolor='rgba(68,255,136,0.08)',
                  line_width=0, row=2, col=1,
                  annotation_text='割安', annotation_font_color='#44ff88',
                  annotation_position='right')

per_y_min = float(max(0, per_plot[['per', 'lower']].min().min() * 0.90))
per_y_max = float(per_plot[['per', 'upper']].max().max() * 1.08)
fig2_v2.update_yaxes(range=[per_y_min, per_y_max],
                     tickformat='.0f', ticksuffix='x', title_text='PER',
                     row=1, col=1)
fig2_v2.update_yaxes(range=[-0.15, 1.3],
                     tickformat='.1f', title_text='%B',
                     row=2, col=1)

fig2_v2.update_layout(
    height=640,
    paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
    font=dict(color='white', family='Arial'),
    legend=dict(orientation='h', y=1.03, bgcolor='rgba(0,0,0,0)', font_size=12),
    margin=dict(l=70, r=20, t=50, b=20),
)
for r in range(1, 3):
    fig2_v2.update_xaxes(gridcolor='#1e1e30', row=r, col=1)
    fig2_v2.update_yaxes(gridcolor='#1e1e30', row=r, col=1)

# ─── fig1_fc: 株価予測チャート（lly_forecast_report.py fig1）
dates_hist  = list(close_1y.index)
future_dates = pd.bdate_range(
    start=close_1y.index[-1] + timedelta(days=1),
    periods=N_DAYS
)

band_5  = np.percentile(paths[:, 1:], 5,  axis=0)
band_25 = np.percentile(paths[:, 1:], 25, axis=0)
band_50 = np.percentile(paths[:, 1:], 50, axis=0)
band_75 = np.percentile(paths[:, 1:], 75, axis=0)
band_95 = np.percentile(paths[:, 1:], 95, axis=0)

fig1_fc = go.Figure()

fig1_fc.add_trace(go.Scatter(
    x=list(sma50_1y.index), y=list(sma50_1y),
    name='SMA50', line=dict(color='#ff9944', width=1.2, dash='dash'), opacity=0.7))
fig1_fc.add_trace(go.Scatter(
    x=list(sma200_1y.index), y=list(sma200_1y),
    name='SMA200', line=dict(color='#ff4444', width=1.2, dash='dot'), opacity=0.7))

fig1_fc.add_trace(go.Scatter(
    x=list(future_dates) + list(future_dates[::-1]),
    y=list(band_95) + list(band_5[::-1]),
    fill='toself', fillcolor='rgba(0,212,255,0.06)',
    line=dict(width=0), name='90%信頼区間', showlegend=True
))
fig1_fc.add_trace(go.Scatter(
    x=list(future_dates) + list(future_dates[::-1]),
    y=list(band_75) + list(band_25[::-1]),
    fill='toself', fillcolor='rgba(0,212,255,0.12)',
    line=dict(width=0), name='50%信頼区間', showlegend=True
))

fig1_fc.add_trace(go.Scatter(
    x=list(future_dates), y=list(band_50),
    name=f'予測中央値 (${pct_50:.0f})', line=dict(color='#ff6b35', width=2.5, dash='dot')
))

fig1_fc.add_hline(y=target_mean, line=dict(color='#ffd700', width=1, dash='dash'),
                  annotation_text=f'アナリスト平均 ${target_mean:.0f}',
                  annotation_font_color='#ffd700', annotation_position='bottom right')
fig1_fc.add_hline(y=target_high, line=dict(color='#00ff88', width=0.8, dash='dot'),
                  annotation_text=f'強気目標 ${target_high:.0f}',
                  annotation_font_color='#00ff88', annotation_position='top right')

for ev in events:
    ev_date = datetime.strptime(ev['date'], '%Y-%m-%d')
    if ev_date > close_1y.index[-1]:
        fig1_fc.add_vline(x=ev_date, line=dict(color=ev['color'], width=1, dash='dot'),
                          opacity=0.6)
        fig1_fc.add_annotation(
            x=ev_date, y=cur_price * 1.38,
            text=ev['label'][:15] + '...' if len(ev['label']) > 15 else ev['label'],
            textangle=-70, font=dict(size=9, color=ev['color']),
            showarrow=False, xanchor='left'
        )

for sp in supports[-3:]:
    fig1_fc.add_hline(y=sp, line=dict(color='rgba(0,255,136,0.3)', width=0.8, dash='dash'))
for rs in resistances[:3]:
    fig1_fc.add_hline(y=rs, line=dict(color='rgba(255,100,100,0.3)', width=0.8, dash='dash'))

fig1_fc.add_trace(go.Scatter(
    x=dates_hist, y=list(close_1y),
    name='実績株価', line=dict(color='#00ffff', width=3),
    hovertemplate='$%{y:.2f}<extra></extra>'
))

fig1_fc.update_layout(
    title=dict(text=f'{TICKER_SYM} 株価チャート + 1年予測（モンテカルロ）',
               font=dict(color='white', size=16)),
    height=540,
    paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
    font=dict(color='white', family='Arial'),
    legend=dict(orientation='h', y=-0.12, bgcolor='rgba(0,0,0,0)', font_size=11),
    margin=dict(l=70, r=20, t=60, b=60),
    xaxis=dict(gridcolor='#1e1e30', range=[close_1y.index[0], future_dates[-1]]),
    yaxis=dict(gridcolor='#1e1e30', tickformat='$,.0f', range=[500, 1800]),
    hovermode='x unified'
)

# ─── fig2_fc: アナリスト目標 & MC 分布（lly_forecast_report.py fig2）
x_range = np.linspace(target_low * 0.9, target_high * 1.05, 200)
sigma   = (target_high - target_low) / 4
mu      = target_mean
dist    = np.exp(-0.5 * ((x_range - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

fig2_fc = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.45, 0.55], vertical_spacing=0.06,
    subplot_titles=('アナリスト目標分布（近似）', 'モンテカルロ 1年後分布（2,000本）')
)

fig2_fc.add_trace(go.Scatter(
    x=list(x_range), y=list(dist),
    fill='tozeroy', fillcolor='rgba(0,212,255,0.15)',
    line=dict(color='#00d4ff', width=2),
    name='アナリスト目標分布（近似）'
), row=1, col=1)

fig2_fc.add_trace(go.Histogram(
    x=list(paths[:, -1]),
    nbinsx=60, name='モンテカルロ1年後分布',
    marker_color='rgba(255,107,53,0.5)',
    opacity=0.8
), row=2, col=1)

for row in [1, 2]:
    fig2_fc.add_vline(x=cur_price, line=dict(color='white', width=2),
                      annotation_text=f'現在値 ${cur_price:.0f}' if row == 1 else '',
                      annotation_font_color='white', annotation_position='top right')
    fig2_fc.add_vline(x=target_mean, line=dict(color='#ffd700', width=1.5, dash='dash'),
                      annotation_text=f'平均目標 ${target_mean:.0f}' if row == 1 else '',
                      annotation_font_color='#ffd700', annotation_position='top left')
    fig2_fc.add_vline(x=target_med, line=dict(color='#00ff88', width=1, dash='dot'),
                      annotation_text=f'中央値 ${target_med:.0f}' if row == 1 else '',
                      annotation_font_color='#00ff88', annotation_position='bottom right')

fig2_fc.update_layout(
    title=dict(text='アナリスト目標株価 & モンテカルロ1年後分布',
               font=dict(color='white', size=16)),
    height=480,
    paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
    font=dict(color='white', family='Arial'),
    legend=dict(orientation='h', y=-0.1, bgcolor='rgba(0,0,0,0)'),
    margin=dict(l=70, r=30, t=60, b=60),
    showlegend=True
)
fig2_fc.update_xaxes(gridcolor='#1e1e30', tickformat='$,.0f', title_text='株価 (USD)', row=2, col=1)
fig2_fc.update_yaxes(gridcolor='#1e1e30', title_text='確率密度', row=1, col=1)
fig2_fc.update_yaxes(gridcolor='#1e1e30', title_text='本数', row=2, col=1)

# ─── fig3_fc: 出来高（lly_forecast_report.py fig3）
obv      = (np.sign(close_1y.diff()) * volume_1y).fillna(0).cumsum()
obv_ma20 = obv.rolling(20).mean()

vol_colors_1y = [str(c) for c in np.where(close_1y >= close_1y.shift(1), '#00d4ff', '#ff4444')]
fig3_fc = go.Figure()
fig3_fc.add_trace(go.Bar(
    x=list(price_1y.index),
    y=[float(v) for v in volume_1y],
    marker_color=vol_colors_1y,
    name='出来高',
    hovertemplate='%{y:,.0f}<extra></extra>'
))
fig3_fc.update_layout(
    title=dict(text='出来高', font=dict(color='white', size=15)),
    height=260,
    paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
    font=dict(color='white', family='Arial'),
    showlegend=False,
    margin=dict(l=70, r=20, t=40, b=30),
    xaxis=dict(gridcolor='#1e1e30'),
    yaxis=dict(gridcolor='#1e1e30', title='出来高'),
)

# ─── fig4_fc: OBV（lly_forecast_report.py fig4）
fig4_fc = go.Figure()
fig4_fc.add_trace(go.Scatter(
    x=list(obv.index), y=[float(v) for v in obv],
    name='OBV', line=dict(color='#00ff88', width=2),
    hovertemplate='OBV: %{y:,.0f}<extra></extra>'
))
fig4_fc.add_trace(go.Scatter(
    x=list(obv_ma20.index), y=[float(v) for v in obv_ma20],
    name='OBV MA20', line=dict(color='#ffd700', width=1.5, dash='dash')
))
fig4_fc.update_layout(
    title=dict(text='OBV（On-Balance Volume）— 需給トレンド',
               font=dict(color='white', size=15)),
    height=280,
    paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
    font=dict(color='white', family='Arial'),
    legend=dict(orientation='h', y=1.1, bgcolor='rgba(0,0,0,0)'),
    margin=dict(l=70, r=20, t=50, b=30),
    xaxis=dict(gridcolor='#1e1e30'),
    yaxis=dict(gridcolor='#1e1e30', title='OBV'),
)

# ─── fig_ee: EV/EBITDA ボリンジャーバンド（チャート⑤）
fig_ee = None
if has_ev_ebitda:
    fig_ee = make_subplots(rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.06,
        subplot_titles=('EV/EBITDA ボリンジャーバンド（52日±2σ）', '%B（BBバンド内の位置）'))

    fig_ee.add_trace(go.Scatter(x=list(ee_plot.index), y=list(ee_plot['upper']),
        mode='lines', line=dict(color='rgba(255,107,53,0.4)', width=1),
        name=f'BB上限 {cur_ee_up:.1f}x'), row=1, col=1)
    fig_ee.add_trace(go.Scatter(x=list(ee_plot.index), y=list(ee_plot['lower']),
        mode='lines', line=dict(color='rgba(255,107,53,0.4)', width=1),
        fill='tonexty', fillcolor='rgba(255,107,53,0.15)',
        name=f'BB下限 {cur_ee_lo:.1f}x'), row=1, col=1)
    fig_ee.add_trace(go.Scatter(x=list(ee_plot.index), y=list(ee_plot['ma']),
        mode='lines', line=dict(color='#ffd700', width=1.5, dash='dash'),
        name=f'BB中央 {cur_ee_ma:.1f}x'), row=1, col=1)
    fig_ee.add_trace(go.Scatter(x=list(ee_plot.index), y=list(ee_plot['ee']),
        mode='lines', line=dict(color='#ff6b35', width=2.5),
        name=f'EV/EBITDA {cur_ee:.1f}x',
        hovertemplate='EV/EBITDA: %{y:.1f}x<extra></extra>'), row=1, col=1)
    fig_ee.add_annotation(x=ee_plot.index[-1], y=cur_ee,
        text=f'現在 {cur_ee:.1f}x', showarrow=True, arrowhead=2,
        arrowcolor='#ff6b35', ax=-60, ay=-30,
        font=dict(color='white', size=12), bgcolor='#1a1a2e',
        bordercolor='#ff6b35', borderwidth=1, row=1, col=1)

    fig_ee.add_trace(go.Scatter(x=list(ee_plot.index), y=list(ee_plot['pct_b']),
        mode='lines', line=dict(color='#ff6b35', width=2),
        fill='tozeroy', fillcolor='rgba(255,107,53,0.12)',
        name=f'%B = {cur_ee_pctb:.2f}'), row=2, col=1)
    for lvl, col in [(1.0,'#ff4444'),(0.8,'#ff9944'),(0.5,'#ffd700'),(0.2,'#44ff88'),(0.0,'#44ff88')]:
        fig_ee.add_hline(y=lvl, line=dict(color=col, width=0.8, dash='dash'), row=2, col=1)
    fig_ee.add_hrect(y0=0.8, y1=1.2, fillcolor='rgba(255,68,68,0.08)', line_width=0,
        row=2, col=1, annotation_text='割高', annotation_font_color='#ff4444', annotation_position='right')
    fig_ee.add_hrect(y0=-0.2, y1=0.2, fillcolor='rgba(68,255,136,0.08)', line_width=0,
        row=2, col=1, annotation_text='割安', annotation_font_color='#44ff88', annotation_position='right')

    ee_y_min = max(0, float(ee_plot[['ee','lower']].min().min()) * 0.90)
    ee_y_max = float(ee_plot[['ee','upper']].max().max()) * 1.08
    fig_ee.update_yaxes(range=[ee_y_min, ee_y_max], tickformat='.0f', ticksuffix='x', row=1, col=1)
    fig_ee.update_yaxes(range=[-0.15, 1.3], tickformat='.1f', row=2, col=1)
    fig_ee.update_layout(height=600, paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
        font=dict(color='white', family='Arial'),
        legend=dict(orientation='h', y=1.03, bgcolor='rgba(0,0,0,0)', font_size=12),
        margin=dict(l=70, r=20, t=50, b=20))
    for r in range(1, 3):
        fig_ee.update_xaxes(gridcolor='#1e1e30', row=r, col=1)
        fig_ee.update_yaxes(gridcolor='#1e1e30', row=r, col=1)

# ─── fig_matrix: PER × EV/EBITDA 二軸マトリクス（チャート⑥）
fig_matrix = None
if has_ev_ebitda and len(per_plot) > 0:
    common_idx = per_plot.index.intersection(ee_plot.index)
    if len(common_idx) > 0:
        common_idx = common_idx[common_idx >= common_idx[-1] - pd.Timedelta(days=365)]
        per_aligned = per_plot.loc[common_idx, 'pct_b']
        ee_aligned  = ee_plot.loc[common_idx, 'pct_b']

        n_traj = len(common_idx)
        colors_traj = [f'rgba(0,212,255,{0.15 + 0.7*i/max(n_traj-1,1):.2f})' for i in range(n_traj)]

        fig_matrix = go.Figure()
        fig_matrix.add_hrect(y0=0.5, y1=1.5, fillcolor='rgba(255,68,68,0.05)', line_width=0)
        fig_matrix.add_vrect(x0=0.5, x1=1.5, fillcolor='rgba(255,68,68,0.05)', line_width=0)
        fig_matrix.add_hrect(y0=-0.5, y1=0.5, fillcolor='rgba(68,255,136,0.03)', line_width=0)
        fig_matrix.add_vrect(x0=-0.5, x1=0.5, fillcolor='rgba(68,255,136,0.03)', line_width=0)
        for txt, x, y, col_ann in [
            ('◎ 強い買い', 0.1, 0.1, '#00ff88'),
            ('✕ 見送り',   0.9, 0.9, '#ff4444'),
            ('△ 負債過多?', 0.1, 0.9, '#ff9944'),
            ('△ 利益歪み?', 0.9, 0.1, '#ff9944'),
        ]:
            fig_matrix.add_annotation(x=x, y=y, text=txt, showarrow=False,
                font=dict(color=col_ann, size=12), xref='paper', yref='paper', opacity=0.5)
        fig_matrix.add_hline(y=0.5, line=dict(color='rgba(255,255,255,0.15)', width=1))
        fig_matrix.add_vline(x=0.5, line=dict(color='rgba(255,255,255,0.15)', width=1))
        fig_matrix.add_trace(go.Scatter(
            x=list(per_aligned), y=list(ee_aligned),
            mode='lines+markers',
            line=dict(color='rgba(0,212,255,0.3)', width=1),
            marker=dict(size=3, color=colors_traj),
            name='過去1年の軌跡', showlegend=True,
            hovertemplate='PER %%B: %{x:.2f}<br>EV/EBITDA %%B: %{y:.2f}<extra></extra>'
        ))
        fig_matrix.add_trace(go.Scatter(
            x=[float(per_aligned.iloc[-1])], y=[float(ee_aligned.iloc[-1])],
            mode='markers+text',
            marker=dict(size=16, color='#ff6b35', symbol='star',
                        line=dict(color='white', width=2)),
            text=['現在'], textposition='top center',
            textfont=dict(color='white', size=12),
            name=f'現在 (PER%B={per_aligned.iloc[-1]:.2f}, EE%B={ee_aligned.iloc[-1]:.2f})',
            showlegend=True
        ))
        fig_matrix.update_layout(
            title=dict(text='PER × EV/EBITDA マトリクス（%B 二軸）', font=dict(color='white', size=16)),
            height=500,
            paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
            font=dict(color='white', family='Arial'),
            xaxis=dict(title='PER %B（← 割安 | 割高 →）', gridcolor='#1e1e30',
                       range=[-0.1, 1.1], tickformat='.1f'),
            yaxis=dict(title='EV/EBITDA %B（← 割安 | 割高 →）', gridcolor='#1e1e30',
                       range=[-0.1, 1.1], tickformat='.1f'),
            margin=dict(l=70, r=20, t=60, b=60),
            legend=dict(orientation='h', y=-0.15, bgcolor='rgba(0,0,0,0)')
        )

# ─── fig_peer: ピアー比較（チャート⑦）
print("ピアー比較データ取得中...")
ticker_symbol = TICKER_SYM
peers = ['LLY', 'NVO', 'ABBV', 'PFE', 'MRK', 'JNJ']
peer_data = []
for sym in peers:
    try:
        t = yf.Ticker(sym)
        i = t.info
        ev = i.get('enterpriseValue')
        eb = i.get('ebitda')
        mc = i.get('marketCap')
        name = i.get('shortName', sym)[:20]
        fpe = i.get('forwardPE')
        eg = i.get('earningsGrowth')
        if ev and eb and ev > 0 and eb > 0:
            peer_data.append({
                'ticker': sym, 'name': name,
                'ev_ebitda': ev / eb,
                'fpe': fpe,
                'eg': eg * 100 if eg else None,
                'mktcap': mc
            })
            print(f"  {sym}: EV/EBITDA={ev/eb:.1f}x fPE={fpe}")
    except Exception as e:
        print(f"  {sym} エラー: {e}")

fig_peer = None
avg_ev = 0.0
if len(peer_data) >= 2:
    peer_data.sort(key=lambda x: x['ev_ebitda'], reverse=True)
    peer_names  = [f"{d['ticker']}" for d in peer_data]
    peer_values = [d['ev_ebitda'] for d in peer_data]
    peer_colors = ['#ff6b35' if d['ticker'] == ticker_symbol else '#00d4ff' for d in peer_data]

    fig_peer = go.Figure()
    fig_peer.add_trace(go.Bar(
        y=peer_names, x=peer_values,
        orientation='h',
        marker_color=peer_colors,
        text=[f'{v:.1f}x' for v in peer_values],
        textposition='outside',
        textfont=dict(color='white', size=12),
        name='EV/EBITDA',
        hovertemplate='%{y}: %{x:.1f}x<extra></extra>'
    ))
    avg_ev = sum(peer_values) / len(peer_values)
    fig_peer.add_vline(x=avg_ev, line=dict(color='#ffd700', width=1.5, dash='dash'),
        annotation_text=f'平均 {avg_ev:.1f}x',
        annotation_font_color='#ffd700', annotation_position='top')

    fig_peer.update_layout(
        title=dict(text='EV/EBITDA ピアー比較（製薬大手）', font=dict(color='white', size=16)),
        height=360,
        paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
        font=dict(color='white', family='Arial'),
        xaxis=dict(title='EV/EBITDA (倍)', gridcolor='#1e1e30', ticksuffix='x'),
        yaxis=dict(gridcolor='#1e1e30'),
        margin=dict(l=80, r=60, t=60, b=40),
        showlegend=False
    )

# ════════════════════════════════════════════════════════════════════════
# 10. HTML パーツ生成
# ════════════════════════════════════════════════════════════════════════
chart_price_html = fig1_v2.to_html(full_html=False, include_plotlyjs=False,
                                   div_id='chart-price-macd')
chart_per_html   = fig2_v2.to_html(full_html=False, include_plotlyjs=False,
                                   div_id='chart-per')
chart_mc_html    = fig1_fc.to_html(full_html=False, include_plotlyjs=False,
                                   div_id='chart-mc')
chart_dist_html  = fig2_fc.to_html(full_html=False, include_plotlyjs=False,
                                   div_id='chart-dist')
chart_vol_html   = fig3_fc.to_html(full_html=False, include_plotlyjs=False,
                                   div_id='chart-vol')
chart_obv_html   = fig4_fc.to_html(full_html=False, include_plotlyjs=False,
                                   div_id='chart-obv')

chart_ee_html     = fig_ee.to_html(full_html=False, include_plotlyjs=False, div_id='chart-ev-ebitda') if fig_ee else ''
chart_matrix_html = fig_matrix.to_html(full_html=False, include_plotlyjs=False, div_id='chart-matrix') if fig_matrix else ''
chart_peer_html   = fig_peer.to_html(full_html=False, include_plotlyjs=False, div_id='chart-peer') if fig_peer else ''

# ─── PER 判定
if cur_pctb < 0.2:
    per_judge, per_color = '割安ゾーン',       '#00ff88'
    per_desc = f'PERが歴史的BBの下限付近（%B={cur_pctb:.2f}）。統計的に割安。'
elif cur_pctb < 0.4:
    per_judge, per_color = 'やや割安',         '#7fff00'
    per_desc = f'PERが中央より下（%B={cur_pctb:.2f}）。バリュエーション的に有利。'
elif cur_pctb < 0.6:
    per_judge, per_color = 'フェアバリュー',   '#ffd700'
    per_desc = f'PERが歴史的中央値付近（%B={cur_pctb:.2f}）。適正水準。'
elif cur_pctb < 0.8:
    per_judge, per_color = 'やや割高',         '#ff9944'
    per_desc = f'PERが中央より上（%B={cur_pctb:.2f}）。成長期待が織り込まれている。'
else:
    per_judge, per_color = '割高ゾーン',       '#ff4444'
    per_desc = f'PERがBBの上限付近（%B={cur_pctb:.2f}）。歴史的に高バリュエーション。'

# ─── MACD 判定
last_hist_val = float(hist.iloc[-1])
prev_hist_val = float(hist.iloc[-2])
if last_hist_val > 0 and prev_hist_val <= 0:
    macd_judge, macd_color = 'ゴールデンクロス', '#00ff88'
elif last_hist_val < 0 and prev_hist_val >= 0:
    macd_judge, macd_color = 'デッドクロス',     '#ff4444'
elif last_hist_val > 0 and last_hist_val > prev_hist_val:
    macd_judge, macd_color = '上昇モメンタム継続', '#00d4ff'
elif last_hist_val > 0:
    macd_judge, macd_color = '上昇鈍化',          '#ffd700'
else:
    macd_judge, macd_color = '下降トレンド',       '#ff9944'

# ─── 機関投資家テーブル
inst_rows = ''
if not top_inst.empty:
    for _, row in top_inst.iterrows():
        holder = row.get('Holder', row.get('Name', '—'))
        shares = row.get('Shares', row.get('Value', '—'))
        pct    = row.get('% Out', row.get('pctHeld', '—'))
        val    = row.get('Value', '—')
        inst_rows += f"""<tr>
          <td style="color:#e0e0f0">{holder}</td>
          <td style="text-align:right">{f'{shares:,.0f}' if isinstance(shares, (int, float)) else shares}</td>
          <td style="text-align:right;color:#00d4ff">{f'{float(pct)*100:.2f}%' if isinstance(pct, (int, float)) else pct}</td>
          <td style="text-align:right;color:#ffd700">{f'${val/1e9:.2f}B' if isinstance(val, (int, float)) and val > 1e8 else val}</td>
        </tr>"""

# ─── イベントカード
event_cards = ''
ev_type_label = {'earnings': '決算', 'fda': 'FDA', 'conf': '学会'}
for ev in events:
    ev_dt   = datetime.strptime(ev['date'], '%Y-%m-%d')
    days_to = (ev_dt - datetime.now()).days
    if days_to < 0:
        days_badge = '<span style="background:rgba(255,255,255,.08);padding:2px 7px;border-radius:4px;font-size:10px;color:#666">過去</span>'
    elif days_to < 30:
        days_badge = f'<span style="background:rgba(255,68,68,.2);color:#ff4444;padding:2px 7px;border-radius:4px;font-size:10px">あと {days_to}日</span>'
    elif days_to < 60:
        days_badge = f'<span style="background:rgba(255,153,68,.15);color:#ff9944;padding:2px 7px;border-radius:4px;font-size:10px">あと {days_to}日</span>'
    else:
        days_badge = f'<span style="background:rgba(0,212,255,.1);color:#00d4ff;padding:2px 7px;border-radius:4px;font-size:10px">あと {days_to}日</span>'

    event_cards += f"""
    <div style="background:var(--card);border:1px solid {ev['color']}33;border-left:3px solid {ev['color']};
      border-radius:0 12px 12px 0;padding:16px 18px;margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
        <div>
          <span style="font-size:10px;background:rgba(255,255,255,.07);padding:2px 7px;
            border-radius:4px;color:var(--dim);margin-right:7px">{ev_type_label[ev['type']]}</span>
          <strong style="color:{ev['color']};font-size:15px">{ev['label']}</strong>
        </div>
        <div style="text-align:right;flex-shrink:0;margin-left:12px">
          <div style="font-size:12px;color:var(--dim)">{ev['date']}</div>
          <div style="margin-top:3px">{days_badge}</div>
        </div>
      </div>
      <div style="display:flex;gap:16px;margin-bottom:8px">
        <div style="font-size:12px">
          <span style="color:var(--dim)">予想インパクト：</span>
          <strong style="color:{ev['color']}">{ev['impact']}</strong>
        </div>
        <div style="font-size:12px">
          <span style="color:var(--dim)">重要度：</span>
          <strong style="color:{ev['color']}">{ev['prob']}</strong>
        </div>
      </div>
      <div style="font-size:12px;color:var(--dim);line-height:1.7">{ev['detail']}</div>
    </div>"""

# ─── 価格予測フォーマッタ
def fmt_ret(price):
    r    = (price / cur_price - 1) * 100
    col  = '#00ff88' if r > 0 else '#ff4444' if r < 0 else '#ffd700'
    sign = '+' if r > 0 else ''
    return f'<span style="color:{col}">{sign}{r:.1f}%</span>'

# ─── EV/EBITDA セクション HTML 生成
if has_ev_ebitda:
    _ee_judge = ('割安ゾーン — EV/EBITDAベースでも買いシグナル点灯' if cur_ee_pctb < 0.2
                 else '割安寄り — 統計的に有利な水準' if cur_ee_pctb < 0.4
                 else 'フェアバリュー付近' if cur_ee_pctb < 0.6
                 else 'やや割高')
    ev_ebitda_bb_html = (
        f'<div class="cc"><div class="ct">EV/EBITDA ボリンジャーバンド</div>'
        f'<div class="cd">日次EV（時価総額+純負債）÷ TTM EBITDA に 52日BB(±2σ)を適用。%B &lt; 0.2 = 統計的割安</div>'
        + chart_ee_html
        + f'<div class="ib"><strong>現在 EV/EBITDA {cur_ee:.1f}x</strong>'
        f'（BB中央 {cur_ee_ma:.1f}x / %B={cur_ee_pctb:.2f}）。{_ee_judge}</div></div>'
    )
else:
    ev_ebitda_bb_html = '<div class="cc"><div class="cd" style="color:var(--dim)">EV/EBITDAデータ取得不可</div></div>'

ev_matrix_html = ''
if fig_matrix:
    ev_matrix_html = (
        '<div class="cc"><div class="ct">PER × EV/EBITDA マトリクス</div>'
        '<div class="cd">両指標の%Bを二軸にプロット。左下=◎強い買い、右上=✕見送り。星印が現在位置。</div>'
        + chart_matrix_html + '</div>'
    )

ev_peer_html = ''
if fig_peer:
    _peer_list = '、'.join([f'<strong>{d["ticker"]}</strong>: {d["ev_ebitda"]:.1f}x' for d in peer_data])
    _lly_idx = next((i for i, d in enumerate(peer_data) if d['ticker'] == ticker_symbol), None)
    _lly_vs_avg = '割高' if (_lly_idx is not None and peer_data[_lly_idx]['ev_ebitda'] > avg_ev) else '割安'
    ev_peer_html = (
        '<div class="cc"><div class="ct">EV/EBITDA ピアー比較</div>'
        '<div class="cd">製薬大手6社の現在EV/EBITDA。オレンジ=LLY。業界平均との乖離を確認。</div>'
        + chart_peer_html
        + f'<div class="ib">{_peer_list}。'
        f'LLYは業界平均{avg_ev:.1f}xに対して{_lly_vs_avg}。GLP-1成長プレミアムを反映。</div></div>'
    )

# ════════════════════════════════════════════════════════════════════════
# 11. HTML 出力
# ════════════════════════════════════════════════════════════════════════
print("HTML 生成中...")

# 為替ヘッジ分析の固定HTML
fx_hedge_html = """<div style="background:var(--card);border:1px solid rgba(255,255,255,.07);border-radius:16px;padding:22px;margin-bottom:28px">
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:20px">
    <div style="background:rgba(255,68,68,.08);border:1px solid rgba(255,68,68,.3);border-radius:10px;padding:14px;text-align:center">
      <div style="font-size:11px;color:var(--dim);margin-bottom:4px">現在レート</div>
      <div style="font-size:22px;font-weight:700;color:#ff4444">¥158.94</div>
      <div style="font-size:11px;color:#ff4444;margin-top:4px">歴史的高水準</div>
    </div>
    <div style="background:rgba(255,215,0,.08);border:1px solid rgba(255,215,0,.3);border-radius:10px;padding:14px;text-align:center">
      <div style="font-size:11px;color:var(--dim);margin-bottom:4px">PPP フェアバリュー</div>
      <div style="font-size:22px;font-weight:700;color:#ffd700">¥120〜130</div>
      <div style="font-size:11px;color:#ffd700;margin-top:4px">18〜25% 円安乖離</div>
    </div>
    <div style="background:rgba(0,255,136,.08);border:1px solid rgba(0,255,136,.3);border-radius:10px;padding:14px;text-align:center">
      <div style="font-size:11px;color:var(--dim);margin-bottom:4px">10年平均レート</div>
      <div style="font-size:22px;font-weight:700;color:#00ff88">¥115〜125</div>
      <div style="font-size:11px;color:#00ff88;margin-top:4px">中長期の戻り余地大</div>
    </div>
  </div>
  <div style="font-size:12px;color:var(--dim);margin-bottom:10px">
    ヘッジコスト試算（日米金利差 ≈ 4.5%/年）：1ヶ月 ≈ <strong style="color:#ffd700">−0.38%</strong>　／　3ヶ月 ≈ <strong style="color:#ffd700">−1.13%</strong>
  </div>
  <div style="margin-top:16px;padding:14px;background:rgba(255,68,68,.06);border-left:3px solid #ff4444;border-radius:0 8px 8px 0;font-size:13px;line-height:1.7">
    <strong style="color:#ff4444">⚠ 円安リスクの評価</strong><br>
    現在の¥158台はPPPフェアバリュー¥120〜130への回帰が1〜2年内に起きれば、<strong>株価が+10%上昇しても円建てでは損失になりうる。</strong>
    1ヶ月の短期保有ならヘッジコスト（≈0.38%）は軽微。
    <strong>ポジションサイズを最大投資額の10〜15%以内</strong>に抑えることでリスク管理も有効。
  </div>
</div>"""

HTML = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{TICKER_SYM} 統合クオンツ分析レポート</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {{
  --bg:#0a0a1a; --card:#0f0f23; --card2:#13132a;
  --ac:#00d4ff; --ac2:#ffd700; --ac3:#ff6b35;
  --gr:#00ff88; --rd:#ff4444; --tx:#e0e0f0; --dim:#8888aa;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--tx);font-family:Arial,sans-serif}}
.hdr{{
  background:linear-gradient(135deg,#0a0a1a,#180828,#0a1a2a);
  border-bottom:1px solid rgba(255,107,53,.25);
  padding:36px 56px 28px;position:relative;overflow:hidden
}}
.hdr::before{{content:'';position:absolute;top:-60px;right:200px;
  width:400px;height:400px;
  background:radial-gradient(circle,rgba(255,107,53,.05) 0%,transparent 70%);
  pointer-events:none}}
.badge{{display:inline-block;background:rgba(255,107,53,.1);border:1px solid var(--ac3);
  border-radius:5px;padding:3px 11px;font-size:11px;letter-spacing:2px;
  color:var(--ac3);margin-bottom:10px}}
.badge2{{display:inline-block;background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.3);
  border-radius:5px;padding:3px 10px;font-size:10px;letter-spacing:1px;
  color:var(--ac);margin-left:8px}}
.co{{font-size:30px;font-weight:700;
  background:linear-gradient(135deg,#fff,#ff6b35);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;margin-bottom:5px}}
.meta{{color:var(--dim);font-size:12px}} .meta span{{color:var(--ac2)}}
.price-hero{{position:absolute;right:56px;top:36px;text-align:right}}
.p-main{{font-size:46px;font-weight:700;color:#fff;line-height:1}}
.p-sub{{font-size:13px;color:var(--dim);margin-top:3px}}
.p-jpy{{font-size:19px;color:var(--ac2);margin-top:3px}}
.wrap{{max-width:1380px;margin:0 auto;padding:28px 36px}}
.sec{{font-size:17px;font-weight:700;color:var(--ac3);
  margin:32px 0 18px;padding-bottom:8px;
  border-bottom:1px solid rgba(255,107,53,.2);
  display:flex;align-items:center;gap:9px}}
.sec::before{{content:'';display:inline-block;width:4px;height:17px;
  background:var(--ac3);border-radius:2px}}
.kgrid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(195px,1fr));gap:14px;margin-bottom:36px}}
.kc{{background:var(--card);border:1px solid rgba(255,255,255,.07);
  border-radius:12px;padding:18px;transition:border-color .2s}}
.kc:hover{{border-color:rgba(255,107,53,.3)}}
.kl{{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1px;margin-bottom:7px}}
.kv{{font-size:26px;font-weight:700;color:#fff;line-height:1}}
.ks{{font-size:11px;color:var(--dim);margin-top:5px}}
.cc{{background:var(--card);border:1px solid rgba(255,255,255,.07);
  border-radius:16px;padding:22px;margin-bottom:28px}}
.ct{{font-size:15px;font-weight:600;color:#fff;margin-bottom:5px}}
.cd{{font-size:12px;color:var(--dim);margin-bottom:18px;line-height:1.6}}
.ib{{background:rgba(255,107,53,.05);border:1px solid rgba(255,107,53,.2);
  border-radius:10px;padding:14px 18px;margin-top:14px;
  font-size:13px;line-height:1.7;color:var(--tx)}}
.ib strong{{color:var(--ac2)}}
.ag{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:28px}}
@media(max-width:860px){{.ag{{grid-template-columns:1fr}}.price-hero{{display:none}}}}
.ac{{background:var(--card);border:1px solid rgba(255,255,255,.07);
  border-radius:14px;padding:22px}}
.ac h3{{font-size:11px;color:var(--dim);margin-bottom:14px;
  text-transform:uppercase;letter-spacing:1px}}
.ri{{display:flex;align-items:flex-start;gap:11px;padding:9px 0;
  border-bottom:1px solid rgba(255,255,255,.05);font-size:12px;line-height:1.5}}
.ri:last-child{{border-bottom:none}}
.rd{{width:7px;height:7px;border-radius:50%;margin-top:4px;flex-shrink:0}}
.stbl{{width:100%;border-collapse:collapse;font-size:13px}}
.stbl th{{text-align:left;font-size:10px;color:var(--dim);text-transform:uppercase;
  letter-spacing:1px;padding:7px 11px;border-bottom:1px solid rgba(255,255,255,.1)}}
.stbl td{{padding:10px 11px;border-bottom:1px solid rgba(255,255,255,.04)}}
.stbl tr:hover td{{background:rgba(255,255,255,.02)}}
.pgrid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:28px}}
@media(max-width:860px){{.pgrid{{grid-template-columns:1fr}}}}
.pcard{{background:var(--card);border:1px solid rgba(255,255,255,.07);border-radius:14px;padding:20px}}
.vcard{{background:linear-gradient(135deg,#0f0f23,#1e0f28);
  border:1px solid rgba(255,107,53,.3);border-radius:16px;
  padding:30px;margin-bottom:28px;position:relative;overflow:hidden}}
.vcard::after{{content:'INTEGRATED REPORT';position:absolute;right:28px;top:18px;
  font-size:10px;letter-spacing:3px;color:rgba(255,107,53,.1);font-weight:700}}
.vlbl{{font-size:10px;color:var(--dim);letter-spacing:2px;text-transform:uppercase;margin-bottom:10px}}
.vmain{{font-size:34px;font-weight:700;color:var(--ac3);margin-bottom:14px}}
.vbody{{font-size:13px;line-height:1.8;color:var(--tx);max-width:800px}}
.esg{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}}
@media(max-width:900px){{.esg{{grid-template-columns:repeat(2,1fr)}}}}
.ei{{background:rgba(0,0,0,.3);border-radius:9px;padding:14px;text-align:center}}
.el{{font-size:10px;color:var(--dim);margin-bottom:7px;text-transform:uppercase;letter-spacing:1px}}
.ev{{font-size:17px;font-weight:700;color:var(--ac2)}}
.foot{{text-align:center;padding:28px;color:var(--dim);font-size:10px;
  border-top:1px solid rgba(255,255,255,.05);line-height:2.2}}
.tag{{display:inline-block;background:rgba(255,255,255,.05);border-radius:4px;
  padding:2px 7px;font-size:10px;color:var(--dim);margin:2px}}
</style>
</head>
<body>

<!-- ① ヘッダー -->
<div class="hdr">
  <div class="badge">NYSE: {TICKER_SYM}</div>
  <span class="badge2">統合クオンツレポート</span>
  <div class="co">Eli Lilly and Company — 統合クオンツ分析</div>
  <div class="meta">
    <span>{datetime.now().strftime('%Y年%m月%d日')}</span> &thinsp;·&thinsp;
    USD/JPY: <span>¥{USD_JPY}</span> &thinsp;·&thinsp;
    HV30: <span>{hv30:.1f}%</span>
    {f'&thinsp;·&thinsp; IV(ATM): <span>{iv_atm:.1f}%</span>' if iv_atm else ''}
    &thinsp;·&thinsp; yfinance + Plotly
  </div>
  <div class="price-hero">
    <div class="p-main">${cur_price:.2f}</div>
    <div class="p-sub">現在株価</div>
    <div class="p-jpy">¥{cur_price * USD_JPY:,.0f}</div>
  </div>
</div>

<div class="wrap">

  <!-- ② 主要指標 KPIグリッド -->
  <div class="sec">主要指標</div>
  <div class="kgrid">
    <div class="kc">
      <div class="kl">現在株価</div>
      <div class="kv">${cur_price:.2f}</div>
      <div class="ks">¥{cur_price * USD_JPY:,.0f} (@¥{USD_JPY:.0f})</div>
    </div>
    <div class="kc">
      <div class="kl">52週 高値 / 安値</div>
      <div class="kv" style="font-size:19px">${high_52w:.0f} / ${low_52w:.0f}</div>
      <div class="ks">高値から {(1-cur_price/high_52w)*100:.1f}% 下</div>
    </div>
    <div class="kc">
      <div class="kl">RSI (14日)</div>
      <div class="kv" style="color:{'#ffd700' if 40<cur_rsi<60 else '#00ff88' if cur_rsi<30 else '#ff4444'}">{cur_rsi:.1f}</div>
      <div class="ks">{'売られすぎ' if cur_rsi<30 else '買われすぎ' if cur_rsi>70 else 'ニュートラル'}</div>
    </div>
    <div class="kc">
      <div class="kl">MACD シグナル</div>
      <div class="kv" style="font-size:18px;color:{macd_color}">{macd_judge}</div>
      <div class="ks">MACD {cur_macd:.2f} / Sig {cur_sig:.2f}</div>
    </div>
    <div class="kc">
      <div class="kl">実績 PER (TTM)</div>
      <div class="kv">{cur_per:.1f}x</div>
      <div class="ks">BB %B = {cur_pctb:.2f}</div>
    </div>
    <div class="kc">
      <div class="kl">PER バリュエーション</div>
      <div class="kv" style="font-size:17px;color:{per_color}">{per_judge}</div>
      <div class="ks">BB中央 {cur_ma:.1f}x / 範囲 {cur_lower:.0f}〜{cur_upper:.0f}x</div>
    </div>
    <div class="kc">
      <div class="kl">アナリスト平均目標株価</div>
      <div class="kv" style="font-size:21px">${target_mean:.0f}</div>
      <div class="ks">上昇余地 +{(target_mean/cur_price-1)*100:.1f}% / {num_analysts}人</div>
    </div>
    <div class="kc">
      <div class="kl">1ヶ月後 中央値（MC）</div>
      <div class="kv">${pct_1m[2]:.0f}</div>
      <div class="ks">25%ile ${pct_1m[1]:.0f} — 75%ile ${pct_1m[3]:.0f}</div>
    </div>
    <div class="kc">
      <div class="kl">ヒストリカルVol (30日)</div>
      <div class="kv">{hv30:.1f}%</div>
      <div class="ks">年率換算</div>
    </div>
    <div class="kc">
      <div class="kl">2026年度 売上高ガイダンス</div>
      <div class="kv" style="font-size:19px">$81.5B</div>
      <div class="ks">前年比 +25%（会社見通し中央値）</div>
    </div>
  </div>

  <!-- ③ 株価チャート + MACD -->
  <div class="sec">株価チャート + MACD</div>
  <div class="cc">
    <div class="ct">株価チャート + MACD（過去3年）</div>
    <div class="cd">終値 + SMA20/50/200 ／ 出来高 ／ MACD(12,26,9)</div>
    {chart_price_html}
    <div class="ib">
      <strong>MACDリーディング</strong>：MACD <strong>{cur_macd:.2f}</strong>
      / シグナル <strong>{cur_sig:.2f}</strong>
      / ヒストグラム <strong>{cur_hist:.2f}</strong> → <span style="color:{macd_color}">{macd_judge}</span>。
      50日MAとの位置関係に注意。
      <strong>$1,034レジスタンス突破</strong>が上昇モメンタム回復の鍵。
    </div>
  </div>

  <!-- ④ PER ボリンジャーバンド -->
  <div class="sec">PER ボリンジャーバンド</div>
  <div class="cc">
    <div class="ct">PER ボリンジャーバンド — クオンツ手法</div>
    <div class="cd">実績PER（TTM）に {bb_win}日ボリンジャーバンド(±2σ)を適用。
      %B &lt; 0.2 → 統計的割安（買いゾーン）、%B &gt; 0.8 → 統計的割高（警戒ゾーン）。
      現在 %B = <strong style="color:{per_color}">{cur_pctb:.2f}</strong>（{per_judge}）
    </div>
    {chart_per_html}
    <div class="ib">
      <strong>PER BBリーディング</strong>：現在PER <strong>{cur_per:.1f}x</strong>
      は {bb_win}日BB中央 <strong>{cur_ma:.1f}x</strong>
      （上限 {cur_upper:.1f}x / 下限 {cur_lower:.1f}x）に対して
      <span style="color:{per_color}"><strong>%B = {cur_pctb:.2f}（{per_judge}）</strong></span>。
      {per_desc}
      フォワードEPS $42（会社ガイダンスEPS $33.5-35 + 成長）ベースでは
      <strong>フォワードPER ≈ 24x</strong> と、PEG ~1.1 で成長プレミアムは正当化できる水準。
    </div>
  </div>

  <!-- ⑤ 株価予測チャート（モンテカルロ） -->
  <div class="sec">株価予測チャート（モンテカルロ {N_SIM:,}本）</div>
  <div class="cc">
    <div class="ct">過去1年 + 将来1年予測</div>
    <div class="cd">ドリフト: アナリスト平均目標株価ベース（年率 {annual_drift*100:.1f}%） / HV30={hv30:.1f}% / 縦点線=主要イベント</div>
    {chart_mc_html}
    <div class="ib">
      <strong>読み方</strong>：濃い青帯=50%信頼区間（25〜75%ile）、薄い青帯=90%信頼区間（5〜95%ile）。
      点線縦線は主要イベント（決算・FDA・学会）を示す。
      現時点のドリフト前提（アナリストコンセンサスベース）では
      <strong>1年後中央値 ${pct_50:.0f}</strong>（現値比 {(pct_50/cur_price-1)*100:+.1f}%）。
    </div>
  </div>

  <!-- ⑥ アナリスト目標 & モンテカルロ分布 -->
  <div class="sec">アナリスト目標 &amp; モンテカルロ1年後分布</div>
  <div class="cc">
    <div class="ct">コンセンサス分布 vs 確率的予測分布</div>
    <div class="cd">上段=アナリスト目標の正規近似、下段=モンテカルロ1年後の終値分布（{N_SIM:,}本）</div>
    {chart_dist_html}
    <div class="ib">
      アナリスト{num_analysts}人のレンジ: <strong style="color:#ff4444">${target_low:.0f}</strong>（弱気）〜
      <strong style="color:#00ff88">${target_high:.0f}</strong>（強気）、
      平均 <strong style="color:#ffd700">${target_mean:.0f}</strong>、中央値 <strong>${target_med:.0f}</strong>。
      現在値${cur_price:.0f}はコンセンサスの下位寄りに位置しており、
      <strong>上昇余地の方がダウンサイドより統計的に大きい</strong>局面。
    </div>
  </div>

  <!-- ⑦ 需給分析（出来高 + OBV） -->
  <div class="sec">需給分析 — 出来高 &amp; OBV</div>
  <div class="cc">
    <div class="ct">出来高トレンド &amp; OBV（On-Balance Volume）</div>
    <div class="cd">OBVは上昇日の出来高を累積加算・下落日を減算。上昇トレンド＝機関投資家の買い集め示唆。</div>
    {chart_vol_html}
    {chart_obv_html}
    <div class="ib">
      <strong>OBVリーディング</strong>：OBVが{'上昇' if float(obv.iloc[-1]) > float(obv.iloc[-20]) else '下落'}トレンド。
      {'OBV > MA20 → 買い需要が優勢。株価の上昇継続を示唆。' if float(obv.iloc[-1]) > float(obv_ma20.iloc[-1]) else 'OBV < MA20 → 売り圧力継続。短期的に慎重が必要。'}
      機関投資家の持分比率は約 {info.get("heldPercentInstitutions", 0.85)*100:.0f}% と高く、
      大口の方向転換が株価に直接影響する。
    </div>
  </div>

  <!-- ⑧ イベントカレンダー -->
  <div class="sec">イベントカレンダー（需給インパクト）</div>
  <div style="margin-bottom:28px">
    {event_cards}
  </div>

  <!-- EV/EBITDA 分析 -->
  <div class="sec">EV/EBITDA 分析</div>

  <!-- ① BB チャート -->
  {ev_ebitda_bb_html}

  <!-- ② マトリクス -->
  {ev_matrix_html}

  <!-- ③ ピアー比較 -->
  {ev_peer_html}

  <!-- ⑨ ファンダメンタルズ（カタリスト + リスク） -->
  <div class="sec">ファンダメンタルズ分析</div>
  <div class="ag">
    <div class="ac">
      <h3>カタリスト</h3>
      <div class="ri">
        <span style="color:#00ff88;font-size:16px;line-height:1.3">▲</span>
        <div><strong style="color:#00ff88">Mounjaro +110% YoY</strong><br>Q4 2025 で $7.4B。米国外承認拡大継続。</div>
      </div>
      <div class="ri">
        <span style="color:#00ff88;font-size:16px;line-height:1.3">▲</span>
        <div><strong style="color:#00ff88">Zepbound +123% YoY</strong><br>Q4 2025 で $4.2B。メディケア適用拡大が追い風。</div>
      </div>
      <div class="ri">
        <span style="color:#00d4ff;font-size:16px;line-height:1.3">★</span>
        <div><strong style="color:#00d4ff">Orforglipron 経口GLP-1（Q2/2026 FDA承認予定）</strong><br>注射不要の飲み薬。市場拡大の第2波カタリスト。</div>
      </div>
      <div class="ri">
        <span style="color:#ffd700;font-size:16px;line-height:1.3">◆</span>
        <div><strong style="color:#ffd700">2026ガイダンス $80-83B（+25% YoY）</strong><br>コンセンサス $77.6B を上回る強気見通し。</div>
      </div>
      <div class="ri">
        <span style="color:#ffd700;font-size:16px;line-height:1.3">◆</span>
        <div><strong style="color:#ffd700">製造能力増強</strong><br>インディアナ・ドイツ新工場で供給制約を緩和。</div>
      </div>
    </div>
    <div class="ac">
      <h3>リスクファクター</h3>
      <div class="ri"><div class="rd" style="background:#ff4444"></div>
        <div><strong style="color:#ff4444">Orforglipron FDA再遅延リスク</strong><br>既にQ1→Q2に延期済み。再延期なら $80-100 下振れ。</div>
      </div>
      <div class="ri"><div class="rd" style="background:#ff9944"></div>
        <div><strong style="color:#ff9944">競合GLP-1の台頭</strong><br>ノボノルディスク・AZ・Pfizer が追走。コンパウンド薬による価格圧迫。</div>
      </div>
      <div class="ri"><div class="rd" style="background:#ff9944"></div>
        <div><strong style="color:#ff9944">IRA 薬価交渉（2027年〜）</strong><br>メディケア薬価交渉で長期収益を圧迫。</div>
      </div>
      <div class="ri"><div class="rd" style="background:#ffd700"></div>
        <div><strong style="color:#ffd700">関税・政治リスク</strong><br>製薬業界への関税/価格政策の不確実性。</div>
      </div>
      <div class="ri"><div class="rd" style="background:#00d4ff"></div>
        <div><strong style="color:#00d4ff">円高リスク（日本人投資家固有）</strong><br>BOJ 利上げ継続 → USD/JPY 145-150 で円建てリターンを圧縮。</div>
      </div>
    </div>
  </div>

  <!-- ⑩ 為替ヘッジ分析 -->
  <div class="sec">為替ヘッジ分析（円建て投資家）</div>
  {fx_hedge_html}

  <!-- ⑪ シナリオ分析（円建て） -->
  <div class="sec">シナリオ分析 — 時間軸別 価格予測（円建て）</div>
  <div class="pgrid">
    <div class="pcard">
      <div class="ct" style="margin-bottom:14px">1ヶ月後（〜4月上旬）</div>
      <table class="stbl">
        <tr><th>シナリオ</th><th>価格</th><th>リターン</th><th>背景</th></tr>
        <tr>
          <td style="color:#00ff88">強気</td>
          <td>${pct_1m[4]:.0f}</td>
          <td>{fmt_ret(pct_1m[4])}</td>
          <td style="color:var(--dim);font-size:11px">学会でポジティブ発言 + 機関買い</td>
        </tr>
        <tr>
          <td style="color:#00d4ff">中立強</td>
          <td>${pct_1m[3]:.0f}</td>
          <td>{fmt_ret(pct_1m[3])}</td>
          <td style="color:var(--dim);font-size:11px">現状維持 + 緩やかな値戻し</td>
        </tr>
        <tr>
          <td style="color:#ffd700">基本</td>
          <td>${pct_1m[2]:.0f}</td>
          <td>{fmt_ret(pct_1m[2])}</td>
          <td style="color:var(--dim);font-size:11px">MC中央値（ドリフト継続）</td>
        </tr>
        <tr>
          <td style="color:#ff9944">弱気</td>
          <td>${pct_1m[1]:.0f}</td>
          <td>{fmt_ret(pct_1m[1])}</td>
          <td style="color:var(--dim);font-size:11px">テクニカル悪化 + 売り継続</td>
        </tr>
        <tr>
          <td style="color:#ff4444">ストレス</td>
          <td>${pct_1m[0]:.0f}</td>
          <td>{fmt_ret(pct_1m[0])}</td>
          <td style="color:var(--dim);font-size:11px">市場全体のリスクオフ</td>
        </tr>
      </table>
    </div>
    <div class="pcard">
      <div class="ct" style="margin-bottom:14px">3ヶ月後（〜6月上旬）</div>
      <table class="stbl">
        <tr><th>シナリオ</th><th>価格</th><th>リターン</th><th>背景</th></tr>
        <tr>
          <td style="color:#00ff88">強気</td>
          <td>${pct_3m[4]:.0f}</td>
          <td>{fmt_ret(pct_3m[4])}</td>
          <td style="color:var(--dim);font-size:11px">Q1 beat + Orforglipron承認</td>
        </tr>
        <tr>
          <td style="color:#00d4ff">中立強</td>
          <td>${pct_3m[3]:.0f}</td>
          <td>{fmt_ret(pct_3m[3])}</td>
          <td style="color:var(--dim);font-size:11px">Q1 beat + FDA pending</td>
        </tr>
        <tr>
          <td style="color:#ffd700">基本</td>
          <td>${pct_3m[2]:.0f}</td>
          <td>{fmt_ret(pct_3m[2])}</td>
          <td style="color:var(--dim);font-size:11px">決算インライン + FDA Q2承認</td>
        </tr>
        <tr>
          <td style="color:#ff9944">弱気</td>
          <td>${pct_3m[1]:.0f}</td>
          <td>{fmt_ret(pct_3m[1])}</td>
          <td style="color:var(--dim);font-size:11px">Q1 miss or FDA再延期</td>
        </tr>
        <tr>
          <td style="color:#ff4444">ストレス</td>
          <td>${pct_3m[0]:.0f}</td>
          <td>{fmt_ret(pct_3m[0])}</td>
          <td style="color:var(--dim);font-size:11px">Q1 miss + FDA否決</td>
        </tr>
      </table>
    </div>
  </div>

  <!-- ⑫ 機関投資家ホルダー -->
  <div class="sec">機関投資家 主要ホルダー（需給の主役）</div>
  <div class="cc">
    <div class="cd">機関投資家保有比率 ≈ {info.get("heldPercentInstitutions", 0.85)*100:.0f}%。大口の売買動向が需給を支配する。</div>
    {'<table class="stbl"><tr><th>機関名</th><th style="text-align:right">保有株数</th><th style="text-align:right">保有比率</th><th style="text-align:right">評価額</th></tr>' + inst_rows + '</table>' if inst_rows else '<div style="color:var(--dim);font-size:13px">機関投資家データ取得失敗（yfinance制限）</div>'}
    <div class="ib" style="margin-top:16px">
      <strong>需給インプリケーション</strong>：機関保有比率{info.get("heldPercentInstitutions", 0.85)*100:.0f}%超は
      流動性が高く、指数リバランス・決算後のポジション調整で瞬間的に大きく動きやすい。
      ショート比率 {'%.1f%%' % (short_pct*100) if short_pct else 'N/A'} は
      {'低水準で踏み上げリスクが低い' if short_pct and short_pct < 0.02 else
       '中程度。大型カタリストでショートカバー（踏み上げ）が上昇を加速しうる' if short_pct and short_pct < 0.05 else
       '高水準。カタリスト時のショートスクイーズに注意'}.
    </div>
  </div>

  <!-- ⑬ 最終判定 -->
  <div class="sec">総合判定</div>
  <div class="vcard">
    <div class="vlbl">Integrated Verdict · イベント × クオンツ × 需給</div>
    <div class="vmain">Q2イベント待ち → 段階的強気</div>
    <div class="vbody">
      <p><strong>直近1ヶ月（〜4月）</strong>：大きなカタリストなし。
      テクニカル（50日MA下抜け）が重しで、$950〜1,020のボックス圏を想定。
      MC中央値 <strong>${pct_1m[2]:.0f}</strong>。</p>
      <br>
      <p><strong>3ヶ月（Q1決算 + Orforglipron FDA）</strong>：
      4月決算 × 5月FDA承認のダブルカタリスト期。
      両方ポジティブなら<strong style="color:#00ff88">${pct_3m[3]:.0f}〜{pct_3m[4]:.0f}（+{(pct_3m[3]/cur_price-1)*100:.0f}〜{(pct_3m[4]/cur_price-1)*100:.0f}%）</strong>も視野。
      FDA再延期なら$900割れリスク。<strong>バイナリーイベントに注意。</strong></p>
      <br>
      <p><strong>バリュエーション面</strong>：PER BB %B={cur_pctb:.2f}（{per_judge}）・フォワードPER ≈ 24x・PEG ≈ 1.1
      と歴史的に正当化できる水準。<strong>分割投資 + 為替ヘッジ</strong>が合理的。</p>
      <br>
      <p><strong>需給面</strong>：機関保有比率高く、指数ETFのリバランス需要が継続的な下値支持。
      円建て投資家は¥158台の為替リスクも考慮（ヘッジ or ポジション抑制推奨）。</p>
    </div>
    <div class="esg">
      <div class="ei">
        <div class="el">1ヶ月後中央値</div>
        <div class="ev">${pct_1m[2]:.0f}</div>
        <div style="font-size:10px;color:var(--dim);margin-top:5px">{fmt_ret(pct_1m[2])}</div>
      </div>
      <div class="ei">
        <div class="el">3ヶ月後中央値</div>
        <div class="ev">${pct_3m[2]:.0f}</div>
        <div style="font-size:10px;color:var(--dim);margin-top:5px">{fmt_ret(pct_3m[2])}</div>
      </div>
      <div class="ei">
        <div class="el">アナリスト平均目標</div>
        <div class="ev">${target_mean:.0f}</div>
        <div style="font-size:10px;color:var(--dim);margin-top:5px">{fmt_ret(target_mean)}</div>
      </div>
      <div class="ei">
        <div class="el">強気目標（FDA承認時）</div>
        <div class="ev">${target_high:.0f}</div>
        <div style="font-size:10px;color:var(--dim);margin-top:5px">{fmt_ret(target_high)}</div>
      </div>
    </div>
  </div>

  <div style="margin-bottom:16px;color:var(--dim);font-size:11px">
    <span class="tag">yfinance</span>
    <span class="tag">MACD(12/26/9)</span>
    <span class="tag">PER BB({bb_win}日±2σ)</span>
    <span class="tag">RSI(14日)</span>
    <span class="tag">SMA 20/50/200</span>
    <span class="tag">モンテカルロ {N_SIM:,}本</span>
    <span class="tag">HV30={hv30:.1f}%</span>
    <span class="tag">OBV需給分析</span>
    <span class="tag">アナリストコンセンサス</span>
  </div>
</div>

<div class="foot">
  本レポートは情報提供目的のみ。投資判断はご自身の責任で行ってください。<br>
  Generated by Claude Code + yfinance + Plotly &thinsp;·&thinsp; {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
</body>
</html>"""

with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write(HTML)

print(f"\n完了: {OUT_FILE}  ({len(HTML)/1024:.0f} KB)")
