import os
import gspread
from google.oauth2.service_account import Credentials
from fastapi import FastAPI
import threading

app = FastAPI(title="AI Playbook", version="1.0.0")

SHEET_ID = os.getenv("SPREADSHEET_ID")
SA_PATH = "/app/sa.json"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def connect_to_sheets():
    try:
        creds = Credentials.from_service_account_file(SA_PATH, scopes=SCOPES)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SHEET_ID)
        print(f"✅ Connected to Sheet: {SHEET_ID}")
        return sh
    except Exception as e:
        print(f"⚠️ Google Sheets connection failed: {e}")
        return None

@app.on_event("startup")
def startup_event():
    threading.Thread(target=connect_to_sheets, daemon=True).start()

@app.get("/")
def root():
    return {"status": "ok", "message": "AI Playbook API running"}
