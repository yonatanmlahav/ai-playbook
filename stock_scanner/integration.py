# stock_scanner/integration.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from train import engineer_features
from joblib import load
import pandas as pd
import numpy as np

class IntegratedModel:
    """
    משתמש במודלים שאתה מאמן ב-train.py
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_models()
    
    def load_models(self):
        """טוען את המודלים שנשמרו"""
        try:
            # נסה לטעון את המודל הטוב ביותר
            if os.path.exists('model.joblib'):
                self.model = load('model.joblib')
                print("✅ Loaded main model")
            elif os.path.exists('model_ensemble.joblib'):
                self.model = load('model_ensemble.joblib')
                print("✅ Loaded ensemble model")
            else:
                print("⚠️ No trained model found. Running without predictions.")
                return False
            
            # טען scaler
            if os.path.exists('scaler.joblib'):
                self.scaler = load('scaler.joblib')
                print("✅ Loaded scaler")
            
            return True
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            print("Running without ML predictions")
            return False
    
    def is_loaded(self):
        return self.model is not None
    
    def prepare_features(self, stock_data):
        """
        ממיר נתוני מניה לפורמט שהמודל שלך מצפה לו
        """
        # המר את נתוני yfinance לפורמט שלך
        df = pd.DataFrame(stock_data)
        
        # חשב אינדיקטורים טכניים
        features_dict = self._calculate_technical_indicators(df)
        
        # השתמש באותה פונקציית feature engineering מ-train.py
        features_df = pd.DataFrame([features_dict])
        engineered = engineer_features(features_df)
        
        return engineered
    
    def _calculate_technical_indicators(self, df):
        """מחשב RSI, MACD, וכו' מנתוני מניה"""
        import ta
        
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # RSI
        rsi_indicator = ta.momentum.RSIIndicator(df['close'], window=14)
        rsi = rsi_indicator.rsi().iloc[-1]
        if pd.isna(rsi):
            rsi = 50.0
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        macd_value = macd.macd_diff().iloc[-1]
        if pd.isna(macd_value):
            macd_value = 0.0
        
        # Volume Spike
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        vol_spike = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Breakout %
        high_20 = df['high'].rolling(20).max().iloc[-1]
        current_price = df['close'].iloc[-1]
        breakout_pct = ((current_price - high_20) / high_20 * 100) if high_20 > 0 else 0.0
        
        # Gap %
        if len(df) > 1:
            gap_pct = ((df['open'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100)
        else:
            gap_pct = 0.0
        
        # A_Score (ציון משוקלל)
        score = (
            (rsi / 100 * 30) +
            (min(vol_spike / 3, 1) * 30) +
            (1 if macd_value > 0 else 0) * 20 +
            (1 if breakout_pct > 0 else 0) * 20
        )
        
        return {
            'RSI': float(rsi),
            'MACD': float(macd_value),
            'VolSpike': float(vol_spike),
            'Breakout%': float(breakout_pct),
            'Gap%': float(gap_pct),
            'A_Score': float(score)
        }
    
    def predict(self, stock_data):
        """
        חיזוי אם תהיה פריצה
        מחזיר: (probability, prediction_label)
        """
        if not self.is_loaded():
            # אם אין מודל, החזר תחזית ברירת מחדל מבוססת על Score
            features_dict = self._calculate_technical_indicators(pd.DataFrame(stock_data))
            score = features_dict['A_Score']
            probability = score / 100.0
            return float(probability), 'Win' if probability > 0.5 else 'Loss'
        
        # הכן features
        X = self.prepare_features(stock_data)
        
        # Scale אם יש scaler
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # חזה
        try:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_scaled)[0]
                return float(proba[1]), 'Win' if proba[1] > 0.5 else 'Loss'
            else:
                pred = self.model.predict(X_scaled)[0]
                return float(pred), 'Win' if pred == 1 else 'Loss'
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            # Fallback to score-based prediction
            features_dict = self._calculate_technical_indicators(pd.DataFrame(stock_data))
            score = features_dict['A_Score']
            probability = score / 100.0
            return float(probability), 'Win' if probability > 0.5 else 'Loss'
