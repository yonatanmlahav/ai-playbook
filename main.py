import os
import gspread
from google.oauth2.service_account import Credentials
from fastapi import FastAPI
import threading

app = FastAPI(title="AI Playbook", version="1.0.0")

# ---------- Config ----------
SHEET_ID = os.getenv("SPREADSHEET_ID")   # או None אם תעדיף להשתמש בשם
SA_PATH = os.getenv("GCP_SA_JSON_PATH", "/app/sa.json")

# ---------- Full Scopes ----------
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

        print(f"✅ Connected to Google Sheets successfully")
        return sh

    except gspread.exceptions.APIError as e:
        print(f"❌ APIError: {e}")
        print("⚠️ Check that the Drive API is enabled and the scopes are complete.")
        return None

    except Exception as e:
        print(f"⚠️ Google Sheets connection failed: {e}")
        return None

@app.on_event("startup")
def startup_event():
    threading.Thread(target=connect_to_sheets, daemon=True).start()

@app.get("/")
def root():
    return {"status": "ok", "message": "AI Playbook API running"}
