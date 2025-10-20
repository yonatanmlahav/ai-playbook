# train.py - Optimized ML Training with Ensemble Models
# ======================================================
# XGBoost + Random Forest + LightGBM + Voting Ensemble

import os
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
import warnings
warnings.filterwarnings('ignore')

# Try importing LightGBM
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("⚠️ LightGBM not installed - using XGBoost and RandomForest")
    LIGHTGBM_AVAILABLE = False

# ========== CONFIG ==========
SHEET_ID = os.getenv("SPREADSHEET_ID")
SA_PATH = os.getenv("GCP_SA_JSON_PATH", "/app/sa.json")
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file"
]

# ========== CONNECT TO SHEETS ==========
def connect_to_sheets():
    creds = Credentials.from_service_account_file(SA_PATH, scopes=SCOPES)
    gc = gspread.authorize(creds)
    if SHEET_ID:
        sh = gc.open_by_key(SHEET_ID)
    else:
        sh = gc.open(os.getenv("SHEET_NAME", "AI_Playbook"))
    return sh

# ========== FEATURE ENGINEERING ==========
def engineer_features(df):
    """Create enhanced features for better model performance"""
    
    # Original features
    features = ["RSI", "MACD", "VolSpike", "Breakout%", "Gap%", "A_Score"]
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # Create new features
    feature_df = pd.DataFrame()
    
    # Basic features
    feature_df['rsi'] = df['RSI']
    feature_df['macd'] = df['MACD']
    feature_df['vol_spike'] = df['VolSpike']
    feature_df['breakout'] = df['Breakout%']
    feature_df['gap'] = df['Gap%']
    feature_df['score'] = df['A_Score']
    
    # Enhanced features for better prediction
    feature_df['rsi_optimal'] = ((df['RSI'] >= 55) & (df['RSI'] <= 65)).astype(int)
    feature_df['rsi_distance_60'] = abs(df['RSI'] - 60)
    feature_df['macd_positive'] = (df['MACD'] > 0).astype(int)
    feature_df['vol_high'] = (df['VolSpike'] > 2).astype(int)
    feature_df['breakout_strong'] = (df['Breakout%'] > 3).astype(int)
    feature_df['gap_small'] = (abs(df['Gap%']) < 2).astype(int)
    
    # Interaction features
    feature_df['rsi_macd_combo'] = df['RSI'] * df['MACD']
    feature_df['vol_breakout_combo'] = df['VolSpike'] * df['Breakout%']
    feature_df['momentum'] = (df['RSI'] - 50) * df['MACD']
    
    # Score categories
    feature_df['high_score'] = (df['A_Score'] >= 85).astype(int)
    feature_df['medium_score'] = ((df['A_Score'] >= 70) & (df['A_Score'] < 85)).astype(int)
    
    return feature_df

# ========== TRAIN ENSEMBLE MODEL ==========
def train_ensemble_model(X_train, y_train, X_test, y_test):
    """Train multiple models and create ensemble"""
    
    print("\n🤖 Training individual models...")
    models = {}
    scores = {}
    
    # 1. XGBoost
    print("Training XGBoost...")
    models['xgboost'] = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42
    )
    models['xgboost'].fit(X_train, y_train)
    scores['xgboost'] = accuracy_score(y_test, models['xgboost'].predict(X_test))
    print(f"  XGBoost accuracy: {scores['xgboost']:.4f}")
    
    # 2. Random Forest
    print("Training Random Forest...")
    models['random_forest'] = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=10,
        random_state=42
    )
    models['random_forest'].fit(X_train, y_train)
    scores['random_forest'] = accuracy_score(y_test, models['random_forest'].predict(X_test))
    print(f"  Random Forest accuracy: {scores['random_forest']:.4f}")
    
    # 3. LightGBM (if available)
    if LIGHTGBM_AVAILABLE:
        print("Training LightGBM...")
        models['lightgbm'] = LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbose=-1
        )
        models['lightgbm'].fit(X_train, y_train)
        scores['lightgbm'] = accuracy_score(y_test, models['lightgbm'].predict(X_test))
        print(f"  LightGBM accuracy: {scores['lightgbm']:.4f}")
    
    # 4. Create Voting Ensemble
    print("\n🎯 Creating ensemble model...")
    estimators = [(name, model) for name, model in models.items()]
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    ensemble.fit(X_train, y_train)
    ensemble_score = accuracy_score(y_test, ensemble.predict(X_test))
    print(f"  Ensemble accuracy: {ensemble_score:.4f}")
    
    # Find best model
    scores['ensemble'] = ensemble_score
    best_model_name = max(scores, key=scores.get)
    best_score = scores[best_model_name]
    
    print(f"\n🏆 Best model: {best_model_name} (accuracy: {best_score:.4f})")
    
    # Save all models
    for name, model in models.items():
        dump(model, f'model_{name}.joblib')
    dump(ensemble, 'model_ensemble.joblib')
    
    # Save best as main model
    if best_model_name == 'ensemble':
        best_model = ensemble
    else:
        best_model = models[best_model_name]
    dump(best_model, 'model.joblib')
    
    return best_model, best_model_name, scores

# ========== FEATURE IMPORTANCE ==========
def analyze_feature_importance(models, feature_names):
    """Get feature importance from all models"""
    
    print("\n📊 Feature importance:")
    importance_dict = {}
    
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for i, feature in enumerate(feature_names):
                if feature not in importance_dict:
                    importance_dict[feature] = []
                importance_dict[feature].append(importances[i])
    
    # Average importance
    avg_importance = []
    for feature in feature_names:
        if feature in importance_dict:
            avg = np.mean(importance_dict[feature])
            avg_importance.append((feature, avg))
    
    # Sort by importance
    avg_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop features:")
    for feature, importance in avg_importance[:10]:
        print(f"  {feature}: {importance:.4f}")
    
    return avg_importance

# ========== MAIN TRAINING ==========
def main():
    print("="*60)
    print("🚀 ML Training Pipeline - Ensemble Models")
    print("="*60)
    
    # 1. Load data
    print("\n📊 Loading data from Google Sheets...")
    sh = connect_to_sheets()
    
    try:
        ws = sh.worksheet("Alerts_Log")
        data = ws.get_all_records()
    except Exception as e:
        raise RuntimeError(f"❌ Could not read Alerts_Log: {e}")
    
    df = pd.DataFrame(data)
    if df.empty:
        raise RuntimeError("⚠️ Alerts_Log is empty")
    
    # 2. Filter labeled data
    if "Outcome" not in df.columns:
        if "Result" in df.columns:
            df["Outcome"] = df["Result"]
        else:
            raise RuntimeError("⚠️ Missing 'Outcome' or 'Result' column")
    
    df = df[df["Outcome"].isin(["Win", "Loss"])]
    
    if len(df) < 30:
        raise RuntimeError(f"⚠️ Need at least 30 samples. Found: {len(df)}")
    
    print(f"✅ Loaded {len(df)} labeled samples")
    
    # 3. Create target
    y = (df["Outcome"] == "Win").astype(int)
    print(f"  Win rate: {y.mean():.1%}")
    
    # 4. Engineer features
    print("\n🔧 Engineering features...")
    X = engineer_features(df)
    print(f"✅ Created {len(X.columns)} features")
    
    # 5. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📊 Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    
    # 6. Scale features
    print("\n⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    dump(scaler, 'scaler.joblib')
    
    # 7. Train ensemble
    models_dict = {}
    
    # Train XGBoost
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, random_state=42
    )
    xgb.fit(X_train_scaled, y_train)
    models_dict['xgboost'] = xgb
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_split=10, random_state=42
    )
    rf.fit(X_train_scaled, y_train)
    models_dict['random_forest'] = rf
    
    # Train LightGBM if available
    if LIGHTGBM_AVAILABLE:
        lgb = LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42, verbose=-1
        )
        lgb.fit(X_train_scaled, y_train)
        models_dict['lightgbm'] = lgb
    
    best_model, best_name, scores = train_ensemble_model(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # 8. Feature importance
    importance = analyze_feature_importance(models_dict, X.columns.tolist())
    
    # 9. Final evaluation
    y_pred = best_model.predict(X_test_scaled)
    print("\n📈 Final Results:")
    print(classification_report(y_test, y_pred, target_names=['Loss', 'Win']))
    
    # 10. Summary
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"Best model: {best_name}")
    print(f"Test accuracy: {scores[best_name]:.2%}")
    print("\nFiles saved:")
    print("  - model.joblib (best model)")
    print("  - model_ensemble.joblib")
    print("  - model_xgboost.joblib")
    print("  - model_random_forest.joblib")
    if LIGHTGBM_AVAILABLE:
        print("  - model_lightgbm.joblib")
    print("  - scaler.joblib")
    
    return scores[best_name]

if __name__ == "__main__":
    try:
        accuracy = main()
        exit(0 if accuracy > 0.6 else 1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        exit(1)
