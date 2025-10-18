# AI-Driven Trading Playbook (FastAPI + Google Sheets)

This project automates your **Top-Down Hybrid Playbook** using an AI-driven scoring pipeline.  
It receives real-time alerts from **TradingView**, computes an **A-Score (0–100)**,  
logs the data to **Google Sheets**, and ranks tickers (A/B/C) for trading decisions.

---

## 🧩 Features
- ✅ Receives TradingView webhook alerts in real-time  
- ✅ Computes dynamic A-Score based on RSI, MACD, Volume Spike & Breakout strength  
- ✅ Writes signals to Google Sheets automatically  
- ✅ Works entirely from environment variables (no files needed)  
- ✅ Ready for deployment on **Railway**, **Render**, or **Google Cloud Run**

---

## ⚙️ Architecture Overview
1. **TradingView Pine Script** detects "A-Ready" setups and sends JSON alerts.
2. **FastAPI endpoint** (`/webhook`) receives and scores the signal.
3. **Google Sheets integration** logs and ranks each alert.
4. (Optional) Add a Telegram bot to push Rank A alerts.

---

## 🧠 Example JSON Payload
```json
{
  "symbol": "GWH",
  "tf": "15m",
  "price": 11.04,
  "rsi": 60.1,
  "macd": 0.23,
  "volSpike": 2.4,
  "breakoutPct": 0.035,
  "gapPct": 0.02
}
