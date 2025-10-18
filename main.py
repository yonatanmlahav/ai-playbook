import os
import gspread
from google.oauth2.service_account import Credentials
from fastapi import FastAPI
from contextlib import asynccontextmanager

# ---------- Config ----------
SHEET_ID = os.getenv("SPREADSHEET_ID")
SA_PATH = os.getenv("GCP_SA_JSON_PATH", "/app/sa.json")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file"
]

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


# ---------- Lifespan (runs at startup/shutdown) ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting AI Playbook service...")
    connect_to_sheets()  # runs once at startup
    yield
    print("🛑 Shutting down AI Playbook service...")


# ---------- App ----------
app = FastAPI(title="AI Playbook", version="1.0.0", lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "ok", "message": "AI Playbook API running"}
    from fastapi import Request
import time

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    print(f"📩 Received alert: {data}")

    try:
        symbol = data.get("symbol")
        tf = data.get("tf")
        price = data.get("price")
        rsi = data.get("rsi")
        macd = data.get("macd")
        volSpike = data.get("volSpike")
        breakoutPct = data.get("breakoutPct")
        gapPct = data.get("gapPct")

        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())

        # Write to Google Sheet
        sh = connect_to_sheets()
        if sh:
            try:
                ws = sh.worksheet("Today_Watchlist")
            except gspread.WorksheetNotFound:
                ws = sh.add_worksheet(title="Today_Watchlist", rows=2000, cols=10)
                ws.append_row(["Timestamp","Symbol","TF","Price","RSI","MACD","VolSpike","Breakout%","Gap%"])

            ws.append_row([ts, symbol, tf, price, rsi, macd, volSpike, breakoutPct, gapPct])
            print(f"✅ Added {symbol} ({tf}) to Today_Watchlist")

        return {"ok": True, "symbol": symbol}

    except Exception as e:
        print(f"❌ Error handling webhook: {e}")
        return {"ok": False, "error": str(e)}

