import os
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from xgboost import XGBClassifier
from joblib import dump

# ---------- CONFIG ----------
SHEET_ID = os.getenv("SPREADSHEET_ID")
SA_PATH = os.getenv("GCP_SA_JSON_PATH", "/app/sa.json")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file"
]

# ---------- CONNECT TO SHEETS ----------
def connect_to_sheets():
    creds = Credentials.from_service_account_file(SA_PATH, scopes=SCOPES)
    gc = gspread.authorize(creds)
    if SHEET_ID:
        sh = gc.open_by_key(SHEET_ID)
    else:
        sh = gc.open(os.getenv("SHEET_NAME", "AI_Playbook"))
    return sh

# ---------- LOAD DATA ----------
print("📊 Loading data from Google Sheets...")
sh = connect_to_sheets()
try:
    ws = sh.worksheet("Alerts_Log")
    data = ws.get_all_records()
except Exception as e:
    raise RuntimeError(f"❌ Could not read Alerts_Log: {e}")

df = pd.DataFrame(data)

if df.empty:
    raise RuntimeError("⚠️ Alerts_Log is empty — need more signals before training.")

# ---------- CLEAN & PREPARE ----------
if "Outcome" not in df.columns:
    raise RuntimeError("⚠️ Missing 'Outcome' column in Alerts_Log (must be Win/Loss).")

# Convert Outcome to binary label
df = df[df["Outcome"].isin(["Win", "Loss"])]
y = (df["Outcome"] == "Win").astype(int)

features = ["RSI","MACD","VolSpike","Breakout%","Gap%","A_Score"]
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

X = df[features]

print(f"✅ Loaded {len(df)} samples for training.")

# ---------- TRAIN MODEL ----------
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X, y)

# ---------- FEATURE IMPORTANCE ----------
imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\n🎯 Feature importance:\n")
print(imp)
print("\n💾 Saving model to model.joblib ...")
dump(model, "model.joblib")

print("✅ Training complete! Model saved as model.joblib")
