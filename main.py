import os, time, math, json, tempfile
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gspread
from google.oauth2.service_account import Credentials

# ==== Google Sheets Auth ====

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# ✅ יוצרים קובץ Service Account מתוך המשתנה הסביבתי
if os.getenv("GCP_SA_JSON"):
    with open("/app/sa.json", "w") as f:
        f.write(os.getenv("GCP_SA_JSON"))

GCP_JSON = os.getenv("GCP_SA_JSON_PATH", "/app/sa.json")
SPREADSHEET_NAME = os.getenv("SHEET_NAME", "AI_Playbook")

creds = Credentials.from_service_account_file(GCP_JSON, scopes=SCOPES)
gc = gspread.authorize(creds)

try:
    sh = gc.open(SPREADSHEET_NAME)
except Exception:
    sh = gc.create(SPREADSHEET_NAME)
    sh.share(os.getenv("SHARE_EMAIL", "your@email.com"), perm_type='user', role='writer')

# Ensure worksheets
for ws_name, header in {
    "Today_Watchlist": [
        "Timestamp","Symbol","TF","Price","RSI","MACD","VolSpike","Breakout%","Gap%","ATR%",
        "Float(M)","Sector","MktCond","A_Score","Rank","Reason","Link"
    ],
    "Alerts_Log": [
        "Timestamp","Symbol","TF","Price","RSI","MACD","VolSpike","Breakout%","Gap%","A_Score",
        "Rank","Outcome","R","Notes"
    ],
}.items():
    try:
        ws = sh.worksheet(ws_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=ws_name, rows=2000, cols=len(header))
        ws.append_row(header)

watch = sh.worksheet("Today_Watchlist")
log = sh.worksheet("Alerts_Log")

# ==== Models ====

class Alert(BaseModel):
    symbol: str
    tf: str
    price: float
    rsi: float
    macd: float
    volSpike: float
    breakoutPct: float
    gapPct: float
    ts: Optional[float] = None

app = FastAPI(title="AI Playbook API", version="1.0.0")

# ==== Helpers ====

DEFAULT_FLOAT = 50.0

def fetch_enrichment(sym: str):
    try:
        info = yf.Ticker(sym).info
        atr = None
        try:
            hist = yf.Ticker(sym).history(period="1mo", interval="1d")
            tr = (hist["High"] - hist["Low"]).rolling(14).mean()
            atr = float(tr.iloc[-1])
        except Exception:
            pass
        sector = info.get("sector") or "?"
        float_shares = info.get("floatShares")
        flt_m = float(float_shares)/1e6 if float_shares else DEFAULT_FLOAT
        return atr, sector, flt_m
    except Exception:
        return None, "?", DEFAULT_FLOAT

def market_condition():
    try:
        qqq = yf.Ticker("QQQ").history(period="5d", interval="1h")["Close"]
        slope = float(qqq.pct_change().tail(6).mean())
        return "Bull" if slope > 0 else "Bear" if slope < 0 else "Flat"
    except Exception:
        return "?"

def compute_score(rsi, macd, volSpike, breakoutPct, gapPct, atrp, flt_m, mkt):
    # Normalize components
    rsi_center = 60.0
    rsi_score = max(0.0, 1.0 - abs(rsi - rsi_center)/10.0)  # within ±10 gets >0
    macd_score = np.tanh(max(0.0, macd) * 3.0)
    vol_score = np.tanh((volSpike - 1.0))
    bo_score = np.tanh(max(0.0, breakoutPct) * 8.0)
    gap_pen = np.tanh(max(0.0, abs(gapPct) - 0.03) * 4.0)  # >3% gap penalized
    atr_score = np.tanh(min(0.08, atrp) * 10.0) if atrp is not None else 0.3
    float_bonus = np.tanh(max(0.0, (60.0 - flt_m)) / 20.0)  # smaller float → bonus
    mkt_mult = 1.10 if mkt == "Bull" else 0.95 if mkt == "Bear" else 1.0
    base = (0.20*rsi_score + 0.18*macd_score + 0.22*vol_score + 0.22*bo_score +
            0.08*atr_score + 0.10*float_bonus)
    score = (base - 0.10*gap_pen) * mkt_mult
    return int(max(0, min(100, round(score * 100))))

def rank_from_score(s):
    return "A" if s >= 78 else "B" if s >= 65 else "C"

# ==== Routes ====

@app.post("/webhook")
async def webhook(a: Alert):
    atr, sector, flt_m = fetch_enrichment(a.symbol)
    atrp = (atr / a.price) if atr else None
    mkt = market_condition()
    score = compute_score(a.rsi, a.macd, a.volSpike, a.breakoutPct, a.gapPct,
                          atrp, flt_m, mkt)
    rank = rank_from_score(score)
    reason = (f"RSI≈{a.rsi:.1f}, MACDΔ≈{a.macd:.2f}, Vol×{a.volSpike:.1f}, "
              f"BO {a.breakoutPct:.1%}, Gap {a.gapPct:.1%}, Float {flt_m:.0f}M, {mkt}")
    link = f"https://www.tradingview.com/chart/?symbol={a.symbol}"
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())

    # Log immutable
    log.append_row([
        ts, a.symbol, a.tf, a.price, a.rsi, a.macd, a.volSpike, a.breakoutPct,
        a.gapPct, score, rank, "", "", ""
    ])

    # Upsert Today_Watchlist by Symbol+TF (simple append)
    watch.append_row([
        ts, a.symbol, a.tf, a.price, a.rsi, a.macd, a.volSpike, a.breakoutPct,
        a.gapPct, (atrp or 0.0), flt_m, sector, mkt, score, rank, reason, link
    ])

    return {"ok": True, "score": score, "rank": rank}
