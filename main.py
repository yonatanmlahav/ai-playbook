import os
import gspread
from google.oauth2.service_account import Credentials

# ==== Google Sheets Auth ====

# Full access to read/write Google Sheets
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# Path to the service account JSON
GCP_JSON_PATH = os.getenv("GCP_SA_JSON_PATH", "/app/sa.json")

# Spreadsheet name
SHEET_NAME = os.getenv("SHEET_NAME", "AI_Playbook")

# Create credentials
creds = Credentials.from_service_account_file(GCP_JSON_PATH, scopes=SCOPES)

# Authorize with gspread
gc = gspread.authorize(creds)

# Open or create the spreadsheet
try:
    sh = gc.open(SHEET_NAME)
except gspread.SpreadsheetNotFound:
    print(f"Spreadsheet '{SHEET_NAME}' not found — creating a new one.")
    sh = gc.create(SHEET_NAME)
    share_email = os.getenv("SHARE_EMAIL")
    if share_email:
        sh.share(share_email, perm_type="user", role="writer")
except Exception as e:
    raise RuntimeError(f"Error accessing Google Sheet: {e}")

print(f"✅ Connected to Google Sheet: {SHEET_NAME}")
