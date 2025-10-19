# 🧠 AI-Driven Playbook — Server-Side Market Scanner

A fully automated **FastAPI + Google Sheets + Telegram** backend that receives alerts from **TradingView**, scores them using your *Top-Down Hybrid* playbook, ranks them (A/B/C), and logs everything to Google Sheets.

---

## 🚀 System Overview


TradingView (Pine v5)
↓ JSON webhook
FastAPI server (Railway)
↓ enriches data via yfinance
↓ computes A_Score (0–100)
↓ writes to Google Sheets
↓ sends Telegram alerts for Rank A

---

## ⚙️ Features

✅ FastAPI webhook that receives live alerts from TradingView  
✅ Computes an **A_Score** based on RSI, MACD, Volume Spike, Breakout %, Gap %, ATR %, Float, and Market Trend  
✅ Writes data into two sheets:
- `Today_Watchlist`
- `Alerts_Log`

✅ Auto-shares your sheet with your Google account  
✅ Sends **Telegram alerts** when Rank = A  
✅ `/health` endpoint checks connectivity to Sheets + Telegram  
✅ Clean, production-ready, Railway-friendly

---

## 📁 Project Files

| File | Description |
|------|--------------|
| `main.py` | FastAPI backend — receives alerts, scores, logs, sends alerts |
| `requirements.txt` | Python dependencies |
| `README.md` | This file — setup guide |
| *(optional)* `train.py` | (future) ML learning loop with XGBoost |

---

## 🧩 Environment Variables (set in Railway)

| Name | Example | Description |
|------|----------|-------------|
| `SHEET_NAME` | `AI_Playbook` | Your Google Sheet name |
| `SHARE_EMAIL` | `youremail@gmail.com` | Account to share the sheet with |
| `GCP_SA_JSON` | `{ "type": "service_account", ... }` | Full service account JSON |
| `GCP_SA_JSON_PATH` | `/app/sa.json` | Default service account path |
| `TELEGRAM_TOKEN` | `8111708783:AAE...` | Telegram Bot token |
| `TELEGRAM_CHAT_ID` | `123456789` | Your Telegram chat ID |

💡 *All variables must be marked as “Build & Runtime” in Railway.*

---

## 🧱 Deploy to Railway

1. Create a new repo (or use your existing one):
main.py
requirements.txt
README.md
2. Push to GitHub:
```bash
git add .
git commit -m "Initial AI Playbook backend"
git push origin main
