import os, time, math, json
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Query
from pydantic import BaseModel

# ---------------------- Integrations ----------------------

# Telegram integration
USE_TELEGRAM = bool(os.getenv("TELEGRAM_BOT_TOKEN")) and bool(os.getenv("TELEGRAM_CHAT_ID"))
tg_bot = None
TG_CHAT_ID = None
if USE_TELEGRAM:
    try:
        from telegram import Bot
        tg_bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
        TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
        print("✅ Telegram initialized")
    except Exception as e:
        print("❌ Telegram init failed:", e)
        USE_TELEGRAM = False

# Google Sheets integration
USE_SHEETS = bool(os.getenv("SHEET_NAME")) and (
    bool(os.getenv("GCP_SA_JSON")) or bool(os.getenv("GCP_SA_JSON_PATH"))
)
if USE_SHEETS:
    import gspread
    from google.oauth2.service_account import Credentials

# ---------------------- FastAPI ----------------------
app = FastAPI(title="AI Playbook — Server-Side Market Scanner", version="1.0.0")

@app.get("/")
def root():
    return {"status": "ok", "message": "Server-Side Scanner ready"}

# ---------------------- Utils ----------------------

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    roll_down = pd.Series(loss, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(0)

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def macd_hist(close: pd.Series, fast: int=12, slow: int=26, signal: int=9) -> pd.Series:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return hist

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df['High']; low = df['Low']; close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

# ---------------------- Scoring ----------------------

def score_row(row: pd.Series) -> Dict[str, Any]:
    score = 0.0
    # Weights similar to Pine v5.2
    score += 15 if row['RSI'] > 60 else 10 if row['RSI'] > 55 else 0
    score += 15 if row['MACD_Hist'] > 0 else 0
    score += 20 if row['VolSpike'] else 0
    score += 20 if row['Breakout'] else 0
    score += 10 if row['StrongCandle'] else 0
    score += 10 if row['RR'] > 2.0 else 0
    score += 10 if row['TrendUp'] else 0
    rating = 'A' if score >= 80 else 'B' if score >= 65 else 'C' if score >= 50 else 'X'
    return {"score": round(float(score), 2), "rating": rating}

# ---------------------- Scanner Core ----------------------

DEFAULT_UNIVERSE = os.getenv(
    "UNIVERSE",
    "AAPL,MSFT,NVDA,AMD,AMZN,META,GOOGL,IONQ,SOUN,RBLX,MSTR,GWH,LAC,HBM,SOFI,AFRM,NET,DDOG,CRDO,ABNB,SHOP,TSLA,PLTR,ROKU,SNAP,INTC,AVGO,LLY,NKE,NRG,DKNG,AES"
).split(',')

class ScanParams(BaseModel):
    universe: List[str] = DEFAULT_UNIVERSE
    period: str = "6mo"
    interval: str = "1d"
    price_max: float = 50.0
    price_min: float = 1.0
    vol_min: float = 300_000
    lookback_high: int = 20
    breakout_mult: float = 1.02

def enrich_and_score(tickers: List[str], period: str, interval: str, lookback_high: int, breakout_mult: float) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    data = yf.download(tickers=tickers, period=period, interval=interval, auto_adjust=False, progress=False, group_by='ticker', threads=True)
    rows: List[Dict[str, Any]] = []

    single = isinstance(data.columns, pd.MultiIndex) is False

    for t in tickers:
        try:
            df = data.copy() if single else data[t].copy()
            df = df.dropna()
            if df.empty or len(df) < 60:
                continue

            df['EMA50'] = ema(df['Close'], 50)
            df['EMA200'] = ema(df['Close'], 200)
            df['RSI'] = rsi(df['Close'], 14)
            df['MACD_Hist'] = macd_hist(df['Close'])
            df['VolSMA20'] = df['Volume'].rolling(20).mean()
            df['VolSpike'] = df['Volume'] > df['VolSMA20'] * 1.8
            df['High20'] = df['High'].rolling(lookback_high).max().shift(1)
            df['Breakout'] = df['Close'] > (df['High20'] * breakout_mult)
            df['StrongCandle'] = (df['Close'] > df['Open'] * 1.02) & (df['Close'] >= df['High'] * 0.999)
            df['ATR14'] = atr(df, 14)
            df['RR'] = (df['High'] - df['Low']) / df['ATR14']
            df['TrendUp'] = df['EMA50'] > df['EMA200']

            last = df.iloc[-1]
            s = score_row(last)

            rows.append({
                "symbol": t,
                "close": round(float(last['Close']), 4),
                "change_%": round(float((last['Close']/df['Close'].iloc[-2]-1)*100), 2) if len(df) > 1 else 0.0,
                "volume": int(last['Volume']),
                "vol_x": round(float(last['Volume']/max(last['VolSMA20'], 1)), 2),
                "rsi": round(float(last['RSI']), 2),
                "macd_hist": round(float(last['MACD_Hist']), 4),
                "atr": round(float(last['ATR14']), 4),
                "trend_up": bool(last['TrendUp']),
                "breakout": bool(last['Breakout']),
                "score": s['score'],
                "rating": s['rating']
            })
        except Exception:
            continue

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out
    df_out = df_out.sort_values(["rating", "score", "vol_x", "change_%"], ascending=[True, False, False, False])
    return df_out

# ---------------------- Sheets & Telegram ----------------------

def append_to_sheets(df: pd.DataFrame, worksheet_name: str = "Today_Watchlist"):
    if not USE_SHEETS or df.empty:
        return
    creds = None
    if os.getenv("GCP_SA_JSON"):
        info = json.loads(os.getenv("GCP_SA_JSON"))
        creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets"]) 
    else:
        path = os.getenv("GCP_SA_JSON_PATH", "/app/sa.json")
        creds = Credentials.from_service_account_file(path, scopes=["https://www.googleapis.com/auth/spreadsheets"]) 

    gc = gspread.authorize(creds)
    sh = None
    sheet_name = os.getenv("SHEET_NAME", "AI_Playbook")
    try:
        sh = gc.open(sheet_name)
    except Exception:
        sh = gc.create(sheet_name)

    try:
        ws = sh.worksheet(worksheet_name)
    except Exception:
        ws = sh.add_worksheet(title=worksheet_name, rows="1000", cols="20")
        ws.append_row(list(df.columns))

    rows = df.values.tolist()
    for r in rows:
        ws.append_row(r)

def notify_telegram(df: pd.DataFrame):
    if not USE_TELEGRAM or df.empty:
        return
    top = df[df['rating'] == 'A'].head(10)
    if top.empty:
        return
    lines = ["🔥 A-Ready Breakouts (Server Scanner) 🔥"]
    for _, row in top.iterrows():
        lines.append(f"{row['symbol']}: {row['close']} | RSI {row['rsi']} | Vol× {row['vol_x']} | Score {row['score']}")
    msg = "\n".join(lines)
    try:
        tg_bot.send_message(chat_id=TG_CHAT_ID, text=msg)
    except Exception as e:
        print("❌ Telegram send failed:", e)

# ---------------------- API Endpoints ----------------------

@app.post("/scan")
def scan(params: ScanParams):
    universe = [t.strip().upper() for t in params.universe if t.strip()]
    df = enrich_and_score(universe, params.period, params.interval, params.lookback_high, params.breakout_mult)
    if df.empty:
        return {"count": 0, "results": []}

    df = df[(df['close'] >= params.price_min) & (df['close'] <= params.price_max) & (df['volume'] >= params.vol_min)]
    df = df.reset_index(drop=True)

    append_to_sheets(df)
    notify_telegram(df)

    return {"count": int(len(df)), "results": df.to_dict(orient='records')}

@app.get("/scan/simple")
def scan_simple(universe: str = Query(",".join(DEFAULT_UNIVERSE), description="Comma-separated tickers")):
    tickers = [t.strip().upper() for t in universe.split(',') if t.strip()]
    df = enrich_and_score(tickers, "6mo", "1d", 20, 1.02)
    if df.empty:
        return {"count": 0, "results": []}
    return {"count": int(len(df)), "results": df.to_dict(orient='records')}

# ---------------------- CLI helper ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
