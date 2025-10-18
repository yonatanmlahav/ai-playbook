import os
import time
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager

# ---------- CONFIG ----------
SHEET_ID = os.getenv("SPREADSHEET_ID")
SA_PATH = os.getenv("GCP_SA_JSON_PATH", "/app/sa.json")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file"
]

# ---------- GOOGLE SHEETS CONNECTION ----------
def connect_to_sheets():
    try:
        creds = Credentials.from_service_account_file(SA_PATH, scopes=SCOPES)
        gc = gspread.authorize(creds)
        if SHEET_ID:
            sh = gc.open_by_key(SHEET_ID)
        else:
            sh = gc.open(os.getenv("SHEET_NAME", "AI_Playbook"))

        worksheets = [ws.title for ws in sh.worksheets()]
        print(f"✅ Connected to Google Sheets successfully")
        print(f"📄 Worksheets: {worksheets}")
        return sh

    except gspread.exceptions.APIError as e:
        print(f"❌ APIError: {e}")
        print("⚠️ Check Drive API, sharing, and scopes.")
        return None
    except Exception as e:
        print(f"⚠️ Google Sheets connection failed: {e}")
        return None

# ---------- SCORING FUNCTIONS ----------
def compute_score(rsi, macd, volSpike, breakoutPct, gapPct):
    try:
        rsi_center = 60.0
        rsi_score = max(0.0, 1.0 - abs(rsi - rsi_center) / 10.0)
        macd_score = np.tanh(max(0.0, macd) * 3.0)
        vol_score = np.tanh((volSpike - 1.0))
        bo_score = np.tanh(max(0.0, breakoutPct) * 8.0)
        gap_penalty = np.tanh(max(0.0, abs(gapPct) - 0.03) * 4.0)

        base = (0.25 * rsi_score +
                0.25 * macd_score +
                0.25 * vol_score +
                0.25 * bo_score)

        score = (base - 0.1 * gap_penalty) * 100
        return int(max(0, min(100, round(score))))
    except Exception:
        return 0

def rank_from_score(score):
    if score >= 78:
        return "A"
    elif score >= 65:
        return "B"
    else:
        return "C"

# ---------- LIFESPAN ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting AI Playbook service...")
    connect_to_sheets()  # runs once at startup
    yield
    print("🛑 Shutting down AI Playbook service...")

# ---------- FASTAPI APP ----------
app = FastAPI(title="AI Playbook", version="1.0.0", lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "ok", "message": "AI Playbook API running"}

# ---------- WEBHOOK ----------
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    print(f"📩 Received alert: {data}")

    try:
        symbol = data.get("symbol")
        tf = data.get("tf")
        price = float(data.get("price", 0))
        rsi = float(data.get("rsi", 0))
        macd = float(data.get("macd", 0))
        volSpike = float(data.get("volSpike", 0))
        breakoutPct = float(data.get("breakoutPct", 0))
        gapPct = float(data.get("gapPct", 0))

        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())

        sh = connect_to_sheets()
        if sh:
            # --- Today_Watchlist ---
            try:
                ws = sh.worksheet("Today_Watchlist")
            except gspread.WorksheetNotFound:
                ws = sh.add_worksheet(title="Today_Watchlist", rows=2000, cols=12)
                ws.append_row(["Timestamp","Symbol","TF","Price","RSI","MACD","VolSpike","Breakout%","Gap%","A_Score","Rank"])

            score = compute_score(rsi, macd, volSpike, breakoutPct, gapPct)
            rank = rank_from_score(score)
            ws.append_row([ts, symbol, tf, price, rsi, macd, volSpike, breakoutPct, gapPct, score, rank])
            print(f"✅ Added {symbol} ({tf}) — Score {score} ({rank})")

            # --- Alerts_Log ---
            try:
                try:
                    log_ws = sh.worksheet("Alerts_Log")
                except gspread.WorksheetNotFound:
                    log_ws = sh.add_worksheet(title="Alerts_Log", rows=2000, cols=12)
                    log_ws.append_row(["Timestamp","Symbol","TF","Price","RSI","MACD","VolSpike","Breakout%","Gap%","A_Score","Rank","Outcome"])

                log_ws.append_row([ts, symbol, tf, price, rsi, macd, volSpike, breakoutPct, gapPct, score, rank, ""])
                print(f"🧠 Logged {symbol} ({tf}) in Alerts_Log")
            except Exception as e:
                print(f"⚠️ Error writing to Alerts_Log: {e}")

        return {"ok": True, "symbol": symbol, "score": score, "rank": rank}

    except Exception as e:
        print(f"❌ Error handling webhook: {e}")
        return {"ok": False, "error": str(e)}


