# 🧠 AI-Driven Trading Playbook

An automated pipeline that detects *A-Ready* swing and intraday setups using technical and contextual signals, scores them via AI, and logs them to Google Sheets — ready for trade execution.

---

## ⚙️ Architecture

**TradingView → FastAPI → Google Sheets → AI Learning Loop**

1. **TradingView Indicator (Pine v5)**  
   Detects high-probability breakouts based on:
   - RSI 55–65 rising  
   - MACD > 0 and > signal  
   - Breakout > recent high  
   - Volume spike vs 20-SMA  
   - Optional gap and price filters  

2. **Webhook API (FastAPI + Railway)**  
   Receives alerts, enriches with live Yahoo Finance data, computes an `A_Score (0–100)` and ranks each signal (A/B/C).  
   Writes the data into Google Sheets (`Today_Watchlist` and `Alerts_Log`).

3. **Google Sheets Dashboard**  
   Displays real-time ranked tickers, market context, and historical performance for training.

4. **Learning Loop (XGBoost)**  
   Uses labeled trade outcomes (Win/Loss, R-multiple) to retrain weights weekly and improve precision.

---

## 🚀 Deployment Guide

### 1️⃣ Prerequisites
- A [Google Cloud Service Account JSON](https://console.cloud.google.com/iam-admin/serviceaccounts)
- A Google Sheet named `AI_Playbook`
- Your TradingView indicator (included in this repo)
- Railway account connected to GitHub

### 2️⃣ Environment Variables
In Railway → Project → **Variables**, set:

| Variable | Example |
|-----------|----------|
| `GCP_SA_JSON_PATH` | `/app/sa.json` |
| `SHEET_NAME` | `AI_Playbook` |
| `SHARE_EMAIL` | your@email.com |

Then upload your `sa.json` via Railway → **Files** → `/app/sa.json`.

### 3️⃣ Webhook URL
After deployment, Railway will give you a public URL such as:
