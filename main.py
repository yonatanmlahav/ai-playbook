import os
import time
import math
import logging
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional, Tuple

import numpy as np
import gspread
from fastapi import FastAPI, Request
from google.oauth2.service_account import Credentials
from joblib import load

# ============== CONFIG ==============
SHEET_ID = os.getenv("SPREADSHEET_ID")                      # preferred
SHEET_NAME = os.getenv("SHEET_NAME", "AI_Playbook")         # fallback if no SHEET_ID
SA_PATH = os.getenv("GCP_SA_JSON_PATH", "/app/sa.json")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
]

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

MODEL_PATH = "model.joblib"   # if exists → will be used

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("ai-playbook")

# ============== GLOBAL STATE (CACHED) ==============
gc_client: Optional[gspread.Client] = None
sh: Optional[gspread.Spreadsheet] = None
ws_today: Optional[gspread.Worksheet] = None
ws_log: Optional[gspread.Worksheet] = None
model = None

# simple in-memory metrics
stats = {
    "alerts_total": 0,
    "alerts_today": 0,
    "rank_counts": {"A": 0, "B": 0, "C": 0},
    "latencies_ms": deque(maxlen=500),
    "boot_time": time.time(),
}

def _is_today(ts: float) -> bool:
    return time.gmtime(ts).tm_yday == time.gmtime().tm_yday and time.gmtime(ts).tm_year == time.gmtime().tm_year

# ============== GOOGLE SHEETS ==============
def connect_to_sheets() -> Tuple[Optional[gspread.Client], Optional[gspread.Spreadsheet]]:
    global gc_client, sh
    try:
        creds = Credentials.from_service_account_file(SA_PATH, scopes=SCOPES)
        gc_client = gspread.authorize(creds)
        sh = gc_client.open_by_key(SHEET_ID) if SHEET_ID else gc_client.open(SHEET_NAME)
        titles = [w.title for w in sh.worksheets()]
        log.info(f"✅ Connected to Google Sheets | Worksheets: {titles}")
        return gc_client, sh
    except Exception as e:
        log.error(f"❌ Sheets connection failed: {e}")
        gc_client, sh = None, None
        return None, None

def ensure_worksheets():
    """Ensure Today_Watchlist & Alerts_Log exist and cache them."""
    global ws_today, ws_log
    if not sh:
        return
    # Today_Watchlist
    try:
        ws_today = sh.worksheet("Today_Watchlist")
    except gspread.WorksheetNotFound:
        ws_today = sh.add_worksheet(title="Today_Watchlist", rows=4000, cols=16)
        ws_today.append_row([
            "Timestamp","Symbol","TF","Price","RSI","MACD","VolSpike","Breakout%","Gap%",
            "A_Score","Rank","Reason"
        ])
    # Alerts_Log
    try:
        ws_log = sh.worksheet("Alerts_Log")
    except gspread.WorksheetNotFound:
        ws_log = sh.add_worksheet(title="Alerts_Log", rows=100000, cols=16)
        ws_log.append_row([
            "Timestamp","Symbol","TF","Price","RSI","MACD","VolSpike","Breakout%","Gap%",
            "A_Score","Rank","Outcome","Notes"
        ])
    log.info("🗂️ Worksheets ready: Today_Watchlist, Alerts_Log")

# ============== SCORING ==============
def heuristic_score(rsi: float, macd: float, volSpike: float, breakoutPct: float, gapPct: float) -> int:
    """
    Simple, fast, explainable baseline score (0-100).
    """
    try:
        rsi_center = 60.0
        rsi_score = max(0.0, 1.0 - abs(rsi - rsi_center) / 10.0)
        macd_score = np.tanh(max(0.0, macd) * 3.0)
        vol_score = np.tanh((volSpike - 1.0))
        bo_score = np.tanh(max(0.0, breakoutPct) * 8.0)
        gap_penalty = np.tanh(max(0.0, abs(gapPct) - 0.03) * 4.0)
        base = (0.25*rsi_score + 0.25*macd_score + 0.25*vol_score + 0.25*bo_score)
        score = (base - 0.10*gap_penalty) * 100
        return int(max(0, min(100, round(score))))
    except Exception:
        return 0

def rank_from_score(score: int) -> str:
    return "A" if score >= 78 else "B" if score >= 65 else "C"

def ai_score_or_fallback(rsi, macd, volSpike, breakoutPct, gapPct) -> Tuple[int, str]:
    """
    Compute base heuristic score, and if a trained model exists use it to refine.
    Model was trained on features: [RSI, MACD, VolSpike, Breakout%, Gap%, A_Score]
    where A_Score is the heuristic baseline.
    """
    base = heuristic_score(rsi, macd, volSpike, breakoutPct, gapPct)
    if model:
        try:
            features = np.array([[rsi, macd, volSpike, breakoutPct, gapPct, base]], dtype=float)
            prob = float(model.predict_proba(features)[0][1])
            score = int(max(0, min(100, round(prob * 100))))
            reason = f"AI prob={prob:.2f} | base={base}"
            return score, reason
        except Exception as e:
            log.warning(f"AI scoring failed, fallback to base. Err: {e}")
    # fallback
    return base, f"BASE={base}"

# ============== TELEGRAM (optional) ==============
def notify_telegram(text: str):
    import requests
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=4)
    except Exception as e:
        log.warning(f"Telegram notify failed: {e}")

# ============== FASTAPI APP & LIFESPAN ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    log.info("🚀 Starting AI Playbook service...")
    # load model if exists
    try:
        model = load(MODEL_PATH)
        log.info("🧠 Loaded XGBoost model.")
    except Exception:
        model = None
        log.info("ℹ️ No model found — using heuristic scoring.")

    # connect sheets and ensure worksheets
    connect_to_sheets()
    if sh:
        ensure_worksheets()
    yield
    log.info("🛑 Shutting down AI Playbook service...")

app = FastAPI(title="AI Playbook", version="1.0.0", lifespan=lifespan)

# ============== ROUTES ==============
@app.get("/")
def root():
    up_seconds = int(time.time() - stats["boot_time"])
    return {"status": "ok", "message": "AI Playbook API running", "uptime_sec": up_seconds}

@app.get("/stats")
def get_stats():
    lat = list(stats["latencies_ms"])
    avg = round(sum(lat)/len(lat), 2) if lat else 0.0
    return {
        "alerts_total": stats["alerts_total"],
        "alerts_today": stats["alerts_today"],
        "rank_counts": stats["rank_counts"],
        "avg_latency_ms_last_500": avg
    }

@app.post("/webhook")
async def webhook(request: Request):
    t0 = time.time()
    data = await request.json()
    log.info(f"📩 Alert: {data}")

    try:
        symbol      = data.get("symbol")
        tf          = str(data.get("tf"))
        price       = float(data.get("price", 0))
        rsi         = float(data.get("rsi", 0))
        macd        = float(data.get("macd", 0))
        volSpike    = float(data.get("volSpike", 0))
        breakoutPct = float(data.get("breakoutPct", 0))
        gapPct      = float(data.get("gapPct", 0))
    except Exception as e:
        log.error(f"Bad payload: {e}")
        return {"ok": False, "error": "invalid payload"}

    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())

    # ensure connection (self-heal)
    if not sh:
        connect_to_sheets()
        if sh:
            ensure_worksheets()

    # compute score
    score, reason_detail = ai_score_or_fallback(rsi, macd, volSpike, breakoutPct, gapPct)
    rank = rank_from_score(score)
    reason = f"RSI≈{rsi:.1f}, MACDΔ≈{macd:.2f}, Vol×{volSpike:.1f}, BO {breakoutPct:.1%}, Gap {gapPct:.1%} | {reason_detail}"

    # write rows
    if sh:
        # Today_Watchlist
        try:
            if ws_today is None:
                ensure_worksheets()
            ws_today.append_row([
                ts, symbol, tf, price, rsi, macd, volSpike, breakoutPct, gapPct, score, rank, reason
            ])
            log.info(f"✅ Watchlist {symbol} ({tf}) — {score} ({rank})")
        except Exception as e:
            log.error(f"Error writing Today_Watchlist: {e}")

        # Alerts_Log
        try:
            if ws_log is None:
                ensure_worksheets()
            ws_log.append_row([
                ts, symbol, tf, price, rsi, macd, volSpike, breakoutPct, gapPct, score, rank, "", ""
            ])
            log.info(f"🧠 Logged {symbol} ({tf}) in Alerts_Log")
        except Exception as e:
            log.error(f"Error writing Alerts_Log: {e}")

    # notify for A
    if rank == "A":
        notify_telegram(f"A-Trigger {symbol} ({tf}) — Score {score}\n{reason}")

    # update stats
    stats["alerts_total"] += 1
    if _is_today(time.time()):
        stats["alerts_today"] += 1
    stats["rank_counts"][rank] = stats["rank_counts"].get(rank, 0) + 1

    latency_ms = round((time.time() - t0) * 1000, 2)
    stats["latencies_ms"].append(latency_ms)
    log.info(f"⏱️ Latency: {latency_ms} ms")

    return {"ok": True, "symbol": symbol, "score": score, "rank": rank, "latency_ms": latency_ms}
