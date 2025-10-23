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
# בתחילת הקובץ הוסף:
from .tradingview import TradingViewWebhook

# הוסף לפני if __name__ == "__main__":

@app.post("/api/tradingview/webhook")
async def tradingview_webhook(request: Request):
    """
    מקבל webhook מ-TradingView
    """
    try:
        # פרס את הנתונים מ-TradingView
        tv_data = await TradingViewWebhook.parse_alert(request)
        symbol = tv_data.get("symbol", "").upper()
        
        if not symbol:
            raise HTTPException(status_code=400, detail="Missing symbol")
        
        # מושך נתוני מניה
        stock_data = await scanner.fetch_stock_data(symbol)
        
        # מריץ prediction
        result = await scanner.scan_single(symbol, threshold=0.5)
        
        if result:
            # מעצב את התשובה
            signal = TradingViewWebhook.format_signal(
                result["prediction"],
                symbol,
                result["last_price"]
            )
            
            return {
                "success": True,
                "symbol": symbol,
                "signal": signal,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "price": result["last_price"],
                "recommendation": "BUY" if result["prediction"] >= 0.55 else "HOLD" if result["prediction"] >= 0.45 else "AVOID"
            }
        else:
            return {
                "success": True,
                "symbol": symbol,
                "signal": "⚠️ NO SIGNAL",
                "recommendation": "HOLD"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tradingview/batch-webhook")
async def tradingview_batch_webhook(request: Request):
    """
    מקבל מספר סימנלים בבת אחת
    """
    try:
        data = await request.json()
        symbols = data.get("symbols", [])
        
        if not symbols:
            raise HTTPException(status_code=400, detail="No symbols provided")
        
        results = []
        for symbol in symbols:
            try:
                result = await scanner.scan_single(symbol.upper(), threshold=0.5)
                if result:
                    results.append(result)
            except:
                continue
        
        return {
            "success": True,
            "count": len(results),
            "signals": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
