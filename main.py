import os, time, math, json
from fastapi import FastAPI
from pydantic import BaseModel
import gspread
from google.oauth2.service_account import Credentials
import numpy as np

app = FastAPI(title="AI Playbook", version="1.0")

# === AUTH ===
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
creds_dict = json.loads(os.getenv("GCP_SA_JSON"))
creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
gc = gspread.authorize(creds)
sh = gc.open(os.getenv("SHEET_NAME", "AI_Playbook"))

try:
    ws = sh.worksheet("Today_Watchlist")
except gspread.WorksheetNotFound:
    ws = sh.add_worksheet(title="Today_Watchlist", rows=2000, cols=15)
    ws.append_row(["Timestamp","Symbol","TF","Price","RSI","MACD","VolSpike","Breakout%","Gap%","A_Score","Rank","Reason"])

class Alert(BaseModel):
    symbol: str
    tf: str
    price: float
    rsi: float
    macd: float
    volSpike: float
    breakoutPct: float
    gapPct: float

def compute_score(rsi, macd, volSpike, breakoutPct):
    base = (0.3*rsi/100) + (0.3*np.tanh(macd*2)) + (0.2*np.tanh(volSpike-1)) + (0.2*np.tanh(breakoutPct*10))
    return int(min(100, base*100))

@app.post("/webhook")
async def webhook(a: Alert):
    score = compute_score(a.rsi, a.macd, a.volSpike, a.breakoutPct)
    rank = "A" if score>=78 else "B" if score>=65 else "C"
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    reason = f"RSI {a.rsi:.1f}, MACD {a.macd:.2f}, Vol×{a.volSpike:.1f}, BO {a.breakoutPct:.1%}"
    ws.append_row([ts,a.symbol,a.tf,a.price,a.rsi,a.macd,a.volSpike,a.breakoutPct,a.gapPct,score,rank,reason])
    return {"ok": True, "score": score, "rank": rank}
