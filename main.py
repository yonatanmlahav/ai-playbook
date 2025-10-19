import os
import time
import math
import logging
from typing import Optional, Tuple

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
# Config (Env)
# =========================
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SHEET_NAME = os.getenv("SHEET_NAME", "AI_Playbook")
GCP_SA_JSON_PATH = os.getenv("GCP_SA_JSON_PATH", "/app/sa.json")
GCP_SA_JSON = os.getenv("GCP_SA_JSON")  # Optional: full JSON in env
SHARE_EMAIL = os.getenv("SHARE_EMAIL", "")

# Telegram (optional)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Server
app = FastAPI(title="AI Playbook — Server-Side Market Scanner", version="1.2.0")

# =========================
# Google Sheets Auth
# =========================
def _build_credentials():
    """
    Prefers JSON file at GCP_SA_JSON_PATH.
    Falls back to JSON blob in GCP_SA_JSON env var if file missing.
    """
    if os.path.exists(GCP_SA_JSON_PATH):
        log.info(f"Using service-account file: {GCP_SA_JSON_PATH}")
        return Credentials.from_service_account_file(GCP_SA_JSON_PATH, scopes=SCOPES)
    if GCP_SA_JSON:
        log.info("Using service-account JSON from env GCP_SA_JSON")
        from google.oauth2.service_account import Credentials as Creds
        import json
        info = json.loads(GCP_SA_JSON)
        return Creds.from_service_account_info(info, scopes=SCOPES)
    raise RuntimeError("Service Account credentials not found (GCP_SA_JSON_PATH / GCP_SA_JSON).")

creds = _build_credentials()
gc = gspread.authorize(creds)

def ensure_sheet_and_headers():
    """
    Ensures spreadsheet + two worksheets exist with correct headers.
    Returns (watch_ws, log_ws).
    """
    # Open or create Spreadsheet
    try:
        sh = gc.open(SHEET_NAME)
    except Exception:
        log.warning(f"Spreadsheet '{SHEET_NAME}' not found — creating…")
        sh = gc.create(SHEET_NAME)
        if SHARE_EMAIL:
            try:
                sh.share(SHARE_EMAIL, perm_type="user", role="writer")
                log.info(f"Shared '{SHEET_NAME}' with {SHARE_EMAIL}")
            except Exception as e:
                log.warning(f"Failed sharing with {SHARE_EMAIL}: {e}"

    # Define headers (must match your Blueprint)
    watch_header = [
        "Timestamp","Symbol","TF","Price","RSI","MACD","VolSpike","Breakout%","Gap%",
        "ATR%","Float(M)","Sector","MktCond","A_Score","Rank","Reason","Link"
    ]
    log_header = [
        "Timestamp","Symbol","TF","Price","RSI","MACD","VolSpike","Breakout%","Gap%",
        "A_Score","Rank","Outcome","R","Notes"
    ]

    # Ensure worksheets
    try:
        watch = sh.worksheet("Today_Watchlist")
    except gspread.WorksheetNotFound:
        watch = sh.add_worksheet(title="Today_Watchlist", rows=2000, cols=len(watch_header))
        watch.append_row(watch_header)
    else:
        # If header missing/misaligned, reset header row
        first_row = watch.row_values(1)
        if first_row != watch_header:
            log.info("Realigning headers for Today_Watchlist…")
            watch.delete_rows(1)  # remove old header
            watch.insert_row(watch_header, 1)

    try:
        alerts_log = sh.worksheet("Alerts_Log")
    except gspread.WorksheetNotFound:
        alerts_log = sh.add_worksheet(title="Alerts_Log", rows=50000, cols=len(log_header))
        alerts_log.append_row(log_header)
    else:
        first_row = alerts_log.row_values(1)
        if first_row != log_header:
            log.info("Realigning headers for Alerts_Log…")
            alerts_log.delete_rows(1)
            alerts_log.insert_row(log_header, 1)

    return watch, alerts_log

watch_ws, alerts_ws = ensure_sheet_and_headers()

# =========================
# Telegram (optional)
# =========================
def send_telegram(text: str) -> bool:
    """
    Sends a Telegram message if TELEGRAM_TOKEN + TELEGRAM_CHAT_ID exist.
    Returns True on success, False otherwise.
    """
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        return False
    try:
        # Using python-telegram-bot Bot API
        from telegram import Bot
        bot = Bot(token=TELEGRAM_TOKEN)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, disable_web_page_preview=True)
        return True
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")
        return False
def telegram_self_test():
    """Sends a startup test message to confirm Telegram connectivity."""
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            ok = send_telegram("✅ AI Playbook server is live — Telegram connected successfully.")
            if ok:
                log.info("Telegram self-test sent successfully.")
            else:
                log.warning("Telegram self-test failed to send.")
        except Exception as e:
            log.warning(f"Telegram self-test error: {e}")
    else:
        log.info("Telegram not configured — skipping self-test.")

# =========================
# Models
# =========================
class Alert(BaseModel):
    symbol: str
    tf: str
    price: float
    rsi: float
    macd: float            # MACD delta (line - signal)
    volSpike: float
    breakoutPct: float
    gapPct: float
    ts: Optional[float] = None  # epoch seconds (optional)

# =========================
# Helpers: Data Enrichment
# =========================
DEFAULT_FLOAT_M = 50.0  # fallback float in millions

def calc_atr_percent(sym: str, lookback_days: str = "2mo", atr_len: int = 14) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (ATR, ATR%) for daily timeframe.
    ATR% = ATR / last_close
    """
    try:
        hist = yf.Ticker(sym).history(period=lookback_days, interval="1d")
        if hist is None or hist.empty or len(hist) < atr_len + 1:
            return None, None
        high = hist["High"]
        low = hist["Low"]
        close = hist["Close"]
        prev_close = close.shift(1)

        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(atr_len).mean().iloc[-1]
        last_close = float(close.iloc[-1])
        atrp = float(atr) / last_close if last_close > 0 else None
        return float(atr), (float(atrp) if atrp is not None else None)
    except Exception as e:
        log.warning(f"ATR calc failed for {sym}: {e}")
        return None, None

def fetch_meta(sym: str) -> Tuple[str, float]:
    """
    Returns (sector, float_millions)
    """
    try:
        info = yf.Ticker(sym).info
        sector = info.get("sector") or "?"
        float_shares = info.get("floatShares")
        flt_m = (float(float_shares) / 1e6) if float_shares else DEFAULT_FLOAT_M
        return sector, flt_m
    except Exception as e:
        log.warning(f"Meta fetch failed for {sym}: {e}")
        return "?", DEFAULT_FLOAT_M

def market_condition() -> str:
    """
    Very simple regime check on QQQ hourly slope (last ~6 hours).
    """
    try:
        q = yf.Ticker("QQQ").history(period="5d", interval="1h")["Close"]
        if q is None or len(q) < 6:
            return "?"
        slope = float(q.pct_change().tail(6).mean())
        return "Bull" if slope > 0 else "Bear" if slope < 0 else "Flat"
    except Exception as e:
        log.warning(f"Market condition fetch failed: {e}")
        return "?"

# =========================
# Scoring
# =========================
def compute_score(rsi, macd, volSpike, breakoutPct, gapPct, atrp, flt_m, mkt):
    rsi_center = 60.0
    rsi_score = max(0.0, 1.0 - abs(rsi - rsi_center) / 10.0)              # ±10 סביב 60
    macd_score = np.tanh(max(0.0, macd) * 3.0)
    vol_score  = np.tanh((volSpike - 1.0))
    bo_score   = np.tanh(max(0.0, breakoutPct) * 8.0)
    gap_pen    = np.tanh(max(0.0, abs(gapPct) - 0.03) * 4.0)              # פגיעה בגאפים > 3%
    atr_score  = np.tanh(min(0.10, (atrp or 0.0)) * 10.0)                 # cap ל-10% ATR
    float_bonus = np.tanh(max(0.0, (60.0 - flt_m)) / 20.0)                # float קטן = בונוס
    mkt_mult   = 1.10 if mkt == "Bull" else 0.95 if mkt == "Bear" else 1.0

    base = (0.20*rsi_score + 0.18*macd_score + 0.22*vol_score +
            0.22*bo_score + 0.08*atr_score + 0.10*float_bonus)
    score = (base - 0.10*gap_pen) * mkt_mult
    return int(max(0, min(100, round(score * 100))))

def rank_from_score(s: int) -> str:
    if s >= 78:
        return "A"
    if s >= 65:
        return "B"
    return "C"

# =========================
# Upsert (Symbol + TF)
# =========================
def upsert_watchlist_row(ws: gspread.Worksheet, row_dict: dict):
    """
    Updates row if (Symbol, TF) exists; otherwise appends.
    row_dict keys must match header names in Today_Watchlist.
    """
    header = ws.row_values(1)
    symbol_idx = header.index("Symbol") + 1
    tf_idx = header.index("TF") + 1

    # Pull all current values (except header)
    values = ws.get_all_values()
    updated_row = [row_dict.get(h, "") for h in header]

    # Find existing row by Symbol + TF
    for i, row in enumerate(values[1:], start=2):  # start=2 because header is row 1
        if len(row) >= max(symbol_idx, tf_idx):
            if row[symbol_idx - 1] == row_dict["Symbol"] and row[tf_idx - 1] == row_dict["TF"]:
                # Update entire row
                cell_range = f"A{i}:{gspread.utils.rowcol_to_a1(i, len(header)).split(':')[1]}"
                ws.update(cell_range, [updated_row], value_input_option="USER_ENTERED")
                return "updated"

    # Not found → append
    ws.append_row(updated_row, value_input_option="USER_ENTERED")
    return "inserted"

# =========================
# Routes
# =========================
@app.get("/")
def root():
    return {"status": "ok", "message": "Server-Side Scanner ready"}

@app.get("/health")
def health():
    try:
        _ = watch_ws.title  # touch
        return {"ok": True, "sheet": SHEET_NAME}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/webhook")
async def webhook(alert: Alert, request: Request):
    try:
        # Enrichment
        atr, atrp = calc_atr_percent(alert.symbol)
        sector, flt_m = fetch_meta(alert.symbol)
        mkt = market_condition()

        # Score & rank
        score = compute_score(
            alert.rsi, alert.macd, alert.volSpike, alert.breakoutPct, alert.gapPct,
            atrp, flt_m, mkt
        )
        rank = rank_from_score(score)

        reason = (
            f"RSI≈{alert.rsi:.1f}, MACDΔ≈{alert.macd:.2f}, Vol×{alert.volSpike:.1f}, "
            f"BO {alert.breakoutPct:.1%}, Gap {alert.gapPct:.1%}, "
            f"ATR% {((atrp or 0.0)*100):.1f}%, Float {flt_m:.0f}M, {mkt}"
        )
        link = f"https://www.tradingview.com/chart/?symbol={alert.symbol}"
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        # Log immutable
        alerts_ws.append_row(
            [
                ts, alert.symbol, alert.tf, alert.price, alert.rsi, alert.macd,
                alert.volSpike, alert.breakoutPct, alert.gapPct, score, rank,
                "", "", ""  # Outcome, R, Notes (to be labeled later)
            ],
            value_input_option="USER_ENTERED"
        )

        # Upsert Today_Watchlist
        row_dict = {
            "Timestamp": ts,
            "Symbol": alert.symbol,
            "TF": alert.tf,
            "Price": alert.price,
            "RSI": alert.rsi,
            "MACD": alert.macd,
            "VolSpike": alert.volSpike,
            "Breakout%": alert.breakoutPct,
            "Gap%": alert.gapPct,
            "ATR%": (atrp or 0.0),
            "Float(M)": flt_m,
            "Sector": sector,
            "MktCond": mkt,
            "A_Score": score,
            "Rank": rank,
            "Reason": reason,
            "Link": link,
        }
        action = upsert_watchlist_row(watch_ws, row_dict)

        # Optional: push Telegram on Rank A
        if rank == "A":
            sent = send_telegram(f"🚨 A-Trigger {alert.symbol} ({alert.tf}) — Score {score}\n{reason}\n{link}")
            if sent:
                log.info(f"Telegram sent for {alert.symbol} ({alert.tf})")
            else:
                log.info("Telegram not configured or failed to send.")

        return {"ok": True, "score": score, "rank": rank, "atrp": atrp, "floatM": flt_m, "mkt": mkt, "action": action}
    except Exception as e:
        log.exception("Webhook processing failed")
        raise HTTPException(status_code=500, detail=str(e))
