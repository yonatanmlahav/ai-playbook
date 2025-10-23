# stock_scanner/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from .scanner import StockScanner

app = FastAPI(title="Stock Breakout Scanner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scanner = StockScanner()

class ScanRequest(BaseModel):
    symbols: List[str]
    threshold: Optional[float] = 0.5

@app.get("/")
async def root():
    return {
        "message": "Stock Breakout Scanner API",
        "version": "1.0.0",
        "model_loaded": scanner.model.is_loaded()
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": scanner.model.is_loaded()}

@app.get("/api/stocks/{symbol}")
async def get_stock(symbol: str):
    try:
        data = await scanner.fetch_stock_data(symbol.upper())
        result = await scanner.scan_single(symbol.upper())
        return {
            "symbol": symbol.upper(),
            "data": data[-20:],  # Last 20 days only
            "prediction": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scanner/scan")
async def scan_stocks(request: ScanRequest):
    try:
        results = await scanner.scan_multiple(request.symbols, request.threshold)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scanner/watchlist")
async def scan_watchlist():
    """סורק watchlist ידוע"""
    default_symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AMD",
        "NFLX", "DIS", "PYPL", "SQ", "SHOP", "ROKU", "ZM", "DOCU"
    ]
    
    try:
        results = await scanner.scan_multiple(default_symbols, threshold=0.6)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
