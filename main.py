# ==== Google Sheets Auth ====

from google.oauth2.service_account import Credentials
import gspread
import os

# דרוש גם Sheets וגם Drive כדי לאפשר יצירה, קריאה ושיתוף
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# קובץ ה־Service Account ייווצר מתוך ה־Secret שלך
GCP_JSON = os.getenv("GCP_SA_JSON_PATH", "/app/sa.json")
SPREADSHEET_NAME = os.getenv("SHEET_NAME", "AI_Playbook")

# יצירת האישורים
creds = Credentials.from_service_account_file(GCP_JSON, scopes=SCOPES)
gc = gspread.authorize(creds)

# ניסיון לפתוח את הגיליון
try:
    sh = gc.open(SPREADSHEET_NAME)
except Exception:
    # אם לא קיים — צור גיליון חדש
    sh = gc.create(SPREADSHEET_NAME)
    # שתף אוטומטית עם המייל שלך (מה־Environment Variable)
    share_email = os.getenv("SHARE_EMAIL", "")
    if share_email:
        sh.share(share_email, perm_type='user', role='writer')

# הבטחת קיום גליונות העבודה
for ws_name, header in {
    "Today_Watchlist": [
        "Timestamp","Symbol","TF","Price","RSI","MACD","VolSpike","Breakout%","Gap%",
        "ATR%","Float(M)","Sector","MktCond","A_Score","Rank","Reason","Link"
    ],
    "Alerts_Log": [
        "Timestamp","Symbol","TF","Price","RSI","MACD","VolSpike","Breakout%","Gap%",
        "A_Score","Rank","Outcome","R","Notes"
    ]
}.items():
    try:
        ws = sh.worksheet(ws_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=ws_name, rows=2000, cols=len(header))
        ws.append_row(header)

watch = sh.worksheet("Today_Watchlist")
log = sh.worksheet("Alerts_Log")
