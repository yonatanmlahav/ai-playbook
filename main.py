import os
import json
import gspread
from google.oauth2.service_account import Credentials

# ==== Google Sheets Auth Setup ====

# 1️⃣ Create /app/sa.json dynamically from environment variable
if not os.path.exists("/app/sa.json"):
    sa_content = os.getenv("GCP_SA_JSON")
    if not sa_content:
        raise ValueError("❌ Environment variable GCP_SA_JSON is missing — please add it in Railway Variables.")
    with open("/app/sa.json", "w") as f:
        f.write(sa_content)
    print("✅ Created /app/sa.json from environment variable")

# 2️⃣ Define authentication parameters
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
GCP_JSON_PATH = "/app/sa.json"
SHEET_NAME = os.getenv("SHEET_NAME", "AI_Playbook")

# 3️⃣ Authorize credentials and connect to Google Sheets
try:
    creds = Credentials.from_service_account_file(GCP_JSON_PATH, scopes=SCOPES)
    gc = gspread.authorize(creds)

    try:
        sh = gc.open(SHEET_NAME)
    except gspread.SpreadsheetNotFound:
        print(f"Spreadsheet '{SHEET_NAME}' not found — creating a new one.")
        sh = gc.create(SHEET_NAME)
        share_email = os.getenv("SHARE_EMAIL")
        if share_email:
            sh.share(share_email, perm_type="user", role="writer")

    print(f"✅ Connected to Google Sheet: {SHEET_NAME}")

except Exception as e:
    raise RuntimeError(f"❌ Error connecting to Google Sheets: {e}")
