import os
import time
import math
import logging
from typing import Optional, Tuple
import json

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import gspread
from google.oauth2.service_account import Credentials

# =========================
# Logging
# =========================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ai-playbook")

# =========================
# Environment Variables
# =========================
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SHEET_NAME = os.getenv("SHEET_NAME", "AI_Playbook")
GCP_SA_JSON_PATH = os.getenv("GCP_SA_JSON_PATH", "/app/sa.json")
GCP_SA_JSON = os.getenv("GCP_SA_JSON")
SHARE_EMAIL = os.getenv("SHARE_EMAIL", "")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# =========================
# Initialize App
# =========================
app = FastAPI(title="AI Playbook — Server-Side Market Scanner", version="1.3.0")

# =========================
# Google Sheets Auth
# =========================
def build_credentials():
    """Builds Google credentials from file or env JSON."""
    try:
        if os.path.exists(GCP_SA_JSON_PATH):
            return Credentials.from_service_account_file(GCP_SA_JSON_PATH, scopes=SCOPES)
        elif GCP_SA_JSON:
            info = json.loads(GCP_SA_JSON)
            return Credentials.from_service_account_info(info, scopes=SCOPES)
        else:
            raise FileNotFoundError("No Google credentials found.")
    except Exception as e:
        log.error(f"Google auth failed: {e}")
        send_telegram(f"⚠️ Google Sheets authentication failed:\n{e}")
        raise

def ensure_sheets():
    """Ensures main spreadsheet and worksheets exist."""
    creds = build_credentials()
    gc = gspread.authorize(creds)

    try:
        sh = gc.open(SHEET_NAME)
        log.info(f"Connected to sheet: {SHEET_NAME}")
    except Exception:
        log.warning(f"Sheet '{SHEET_NAME}' not found, creating new one.")
        sh = gc.create(SHEET_NAME)
        if SHARE_EMAIL:
            try:
                sh.share(SHARE_EMAIL, perm_type="user", role="writer")
            except Exception as e:
                log.warning(f"Could not share sheet: {e}")

    headers_watch = [
        "Timestamp","Symbol","TF","Price","RSI","MACD","VolSpike","Breakout%","Gap%",
        "ATR%","Float(M)","Sector","MktCond","A_Score","Rank","Reason","Link"
    ]
    headers_log = [
        "Timestamp","Symbol","TF","Price","RSI","MACD","VolSpike","Breakout%","Gap%",
        "A_Score","Rank","Outcome","R","Notes"
    ]

    # Ensure worksheets
    def setup_worksheet(title, headers):
        try:
            ws = sh.worksheet(title)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=title, rows=2000, cols=len(headers))
            ws.append_row(headers)
            return ws
        first = ws.row_values(1)
        if first != headers:
            ws.delete_rows(1)
            ws.insert_row(headers, 1)
        return ws

    watch = setup_worksheet("Today_Watchlist", headers_watch)
    logsheet = setup_worksheet("Alerts_Log", headers_log)
    return watch, logsheet

# =========================
# Telegram
# =========================
def send_telegram(text: str) -> bool:
    """Sends a Telegram message if credentials exist."""
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        return False
    try:
        from telegram import Bot
        bot = Bot(token=TELEGRAM_TOKEN)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, disable_web_page_preview=True)
        return True
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")
        return False

def telegram_self_test():
    """Sends a startup self-test message."""
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            send_telegram("✅ AI Playbook server is live — Telegram connection OK.")
            log.info("Telegram self-test sent successfully.")
        except Exception as e:
            log.warning(f"Telegram self-test failed: {e}")
    else:
        log.info("Telegram not configured — skipping self-test.")

# =========================
# Data Models
# =========================
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

# =========================
# Helpers
# =========================
DEFAULT_FLOAT_M = 50.0

def calc_atr_percent(sym: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        hist = yf.Ticker(sym).history(period="2mo", interval="1d")
        if len(hist) < 15:
            return None, None
        high, low, close = hist["High"], hist["Low"], hist["Close"]
        prev = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        last_close = close.iloc[-1]
        return float(atr), float(atr / last_close)
    except Exception as e:
        log.warning(f"ATR failed for {sym}: {e}")
        return None, None

def fetch_meta(sym: str) -> Tuple[str, float]:
    try:
        info = yf.Ticker(sym).info
        sector = info.get("sector") or "?"
        flt = info.get("floatShares")
        return sector, (float(flt)/1e6 if flt else DEFAULT_FLOAT_M)
    except Exception as e:
        log.warning(f"Meta fetch failed: {e}")
        return "?", DEFAULT_FLOAT_M

def market_condition() -> str:
    try:
        q = yf.Ticker("QQQ").history(period="5d", interval="1h")["Close"]
        slope = float(q.pct_change().tail(6).mean())
        return "Bull" if slope > 0 else "Bear" if slope < 0 else "Flat"
    except Exception as e:
        log.warning(f"Market condition fetch failed: {e}")
        return "?"

def compute_score(rsi, macd, volSpike, breakoutPct, gapPct, atrp, flt_m, mkt):
    rsi_center = 60.0
    rsi_score = max(0, 1 - abs(rsi - rsi_center) / 10)
    macd_score = np.tanh(max(0, macd) * 3)
    vol_score = np.tanh(volSpike - 1)
    bo_score = np.tanh(max(0, breakoutPct) * 8)
    gap_pen = np.tanh(max(0, abs(gapPct) - 0.03) * 4)
    atr_score = np.tanh(min(0.1, atrp or 0) * 10)
    float_bonus = np.tanh(max(0, (60 - flt_m)) / 20)
    mkt_mult = 1.1 if mkt == "Bull" else 0.95 if mkt == "Bear" else 1
    base = (0.2*rsi_score + 0.18*macd_score + 0.22*vol_score +
            0.22*bo_score + 0.08*atr_score + 0.1*float_bonus)
    score = (base - 0.1*gap_pen) * mkt_mult
    return int(max(0, min(100, round(score*100))))

def rank_from_score(s: int) -> str:
    return "A" if s >= 78 else "B" if s >= 65 else "C"

def upsert_row(ws, data: dict):
    header = ws.row_values(1)
    symbol_idx, tf_idx = header.index("Symbol")+1, header.index("TF")+1
    rows = ws.get_all_values()[1:]
    row_values = [data.get(h, "") for h in header]

    for i, row in enumerate(rows, start=2):
        if len(row) >= max(symbol_idx, tf_idx):
            if row[symbol_idx-1] == data["Symbol"] and row[tf_idx-1] == data["TF"]:
                end_col = gspread.utils.rowcol_to_a1(1, len(header)).split("A")[1]
                ws.update(f"A{i}:{end_col}{i}", [row_values])
                return "updated"
    ws.append_row(row_values)
    return "inserted"

# =========================
# Initialize Sheets + Telegram self-test
# =========================
watch_ws, log_ws = ensure_sheets()
telegram_self_test()

# =========================
# Routes
# =========================
@app.get("/")
def root():
    return {"status": "ok", "message": "Server running successfully"}

@app.get("/health")
def health():
    """Checks connectivity to Sheets and Telegram."""
    try:
        test_symbol = "TEST"
        watch_ws.append_row(["TEST", test_symbol, "1h", 0, 0, 0, 0, 0, 0, 0, 0, "", "", 0, "C", "Health check", ""],
                            value_input_option="USER_ENTERED")
        send_telegram("✅ Health check passed — Sheets and Telegram OK.")
        return {"ok": True}
    except Exception as e:
        send_telegram(f"⚠️ Health check failed:\n{e}")
        return {"ok": False, "error": str(e)}

@app.post("/webhook")
async def webhook(alert: Alert):
    try:
        atr, atrp = calc_atr_percent(alert.symbol)
        sector, flt = fetch_meta(alert.symbol)
        mkt = market_condition()

       score = compute_score(
    alert.rsi,
    alert.macd,
    alert.volSpike,
    alert.breakoutPct,
    alert.gapPct,
    atrp,
    flt,
    mkt
)
