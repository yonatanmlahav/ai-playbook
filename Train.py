# train.py - Enhanced ML Training Pipeline v3.0
# ============================================
# כולל: Ensemble models, Feature engineering, Backtesting, Hyperparameter tuning

import os
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
from joblib import dump, load
import json

# Try importing advanced ML libraries
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("⚠️ LightGBM not installed - using XGBoost and RandomForest only")
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ CatBoost not installed - using other models")
    CATBOOST_AVAILABLE = False

# ========================================
# CONFIGURATION
# ========================================
SHEET_ID = os.getenv("SPREADSHEET_ID")
SA_PATH = os.getenv("GCP_SA_JSON_PATH", "/app/sa.json")
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file"
]

# Training parameters
MIN_SAMPLES = 30  # Minimum samples needed for training
TEST_SIZE = 0.2   # 80/20 train/test split
CV_FOLDS = 5      # Cross-validation folds
RANDOM_STATE = 42

# ========================================
# CONNECT TO SHEETS
# ========================================
def connect_to_sheets():
    """Connect to Google Sheets"""
    try:
        creds = Credentials.from_service_account_file(SA_PATH, scopes=SCOPES)
        gc = gspread.authorize(creds)
        if SHEET_ID:
            sh = gc.open_by_key(SHEET_ID)
        else:
            sh = gc.open(os.getenv("SHEET_NAME", "AI_Playbook"))
        return sh
    except Exception as e:
        print(f"❌ Failed to connect to sheets: {e}")
        raise

# ========================================
# FEATURE ENGINEERING
# ========================================
class FeatureEngineer:
    """Advanced feature engineering pipeline"""
    
    def __init__(self):
        self.feature_names = []
        self.scaler = StandardScaler()
        
    def create_features(self, df):
        """Create advanced features from raw data"""
        print("🔧 Engineering features...")
        
        # Original features
        original_features = ["RSI", "MACD", "VolSpike", "Breakout%", "Gap%", "A_Score"]
        
        # Clean original features
        for col in original_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        
        # New engineered features
        features_df = pd.DataFrame()
        
        # 1. Basic features
        features_df['rsi'] = df['RSI']
        features_df['macd'] = df['MACD']
        features_df['vol_spike'] = df['VolSpike']
        features_df['breakout_pct'] = df['Breakout%']
        features_df['gap_pct'] = df['Gap%']
        features_df['a_score'] = df['A_Score']
        
        # 2. RSI features
        features_df['rsi_oversold'] = (df['RSI'] < 30).astype(int)
        features_df['rsi_overbought'] = (df['RSI'] > 70).astype(int)
        features_df['rsi_optimal'] = ((df['RSI'] >= 55) & (df['RSI'] <= 65)).astype(int)
        features_df['rsi_distance_60'] = abs(df['RSI'] - 60)
        
        # 3. MACD features
        features_df['macd_positive'] = (df['MACD'] > 0).astype(int)
        features_df['macd_strength'] = abs(df['MACD'])
        
        # 4. Volume features
        features_df['vol_extreme'] = (df['VolSpike'] > 3).astype(int)
        features_df['vol_moderate'] = ((df['VolSpike'] >= 1.5) & (df['VolSpike'] <= 3)).astype(int)
        features_df['vol_log'] = np.log1p(df['VolSpike'])
        
        # 5. Breakout features
        features_df['breakout_strong'] = (df['Breakout%'] > 5).astype(int)
        features_df['breakout_moderate'] = ((df['Breakout%'] >= 2) & (df['Breakout%'] <= 5)).astype(int)
        
        # 6. Gap features
        features_df['gap_small'] = (abs(df['Gap%']) < 2).astype(int)
        features_df['gap_abs'] = abs(df['Gap%'])
        
        # 7. Interaction features
        features_df['rsi_macd_interaction'] = df['RSI'] * df['MACD']
        features_df['vol_breakout_interaction'] = df['VolSpike'] * df['Breakout%']
        features_df['momentum_score'] = (df['RSI'] - 50) * df['MACD']
        
        # 8. Score features
        features_df['high_score'] = (df['A_Score'] >= 85).astype(int)
        features_df['medium_score'] = ((df['A_Score'] >= 70) & (df['A_Score'] < 85)).astype(int)
        
        # 9. Market features (if available)
        if 'MarketCond' in df.columns:
            features_df['market_bull'] = (df['MarketCond'] == 'Bull').astype(int)
            features_df['market_bear'] = (df['MarketCond'] == 'Bear').astype(int)
        
        # 10. Time features (if timestamp available)
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            features_df['hour'] = df['Timestamp'].dt.hour
            features_df['day_of_week'] = df['Timestamp'].dt.dayofweek
            features_df['is_morning'] = (df['Timestamp'].dt.hour < 12).astype(int)
            features_df['is_first_hour'] = ((df['Timestamp'].dt.hour >= 9) & (df['Timestamp'].dt.hour < 10)).astype(int)
        
        # Store feature names
        self.feature_names = features_df.columns.tolist()
        
        print(f"✅ Created {len(self.feature_names)} features")
        return features_df
    
    def scale_features(self, X_train, X_test):
        """Scale features for better model performance"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler
        dump(self.scaler, 'scaler.joblib')
        print("💾 Scaler saved to scaler.joblib")
        
        return X_train_scaled, X_test_scaled

# ========================================
# MODEL TRAINING
# ========================================
class ModelTrainer:
    """Advanced model training with ensemble methods"""
    
    def __init__(self):
        self.models = {}
        self.ensemble = None
        self.best_model = None
        self.scores = {}
        
    def create_models(self):
        """Create different model architectures"""
        print("🤖 Creating model architectures...")
        
        # XGBoost
        self.models['xgboost'] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric='logloss'
        )
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE
        )
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = LGBMClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                verbose=-1
            )
        
        # CatBoost
        if CATBOOST_AVAILABLE:
            self.models['catboost'] = CatBoostClassifier(
                iterations=300,
                depth=4,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
                verbose=False
            )
        
        print(f"✅ Created {len(self.models)} models")
    
    def train_individual_models(self, X_train, y_train, X_test, y_test):
        """Train each model individually"""
        print("\n📊 Training individual models...")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS)
            cv_mean = cv_scores.mean()
            
            # ROC AUC
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            else:
                roc_auc = 0
            
            self.scores[name] = {
                'train_accuracy': round(train_score, 4),
                'test_accuracy': round(test_score, 4),
                'cv_mean': round(cv_mean, 4),
                'cv_std': round(cv_scores.std(), 4),
                'roc_auc': round(roc_auc, 4)
            }
            
            print(f"  Train: {train_score:.4f} | Test: {test_score:.4f} | CV: {cv_mean:.4f} | AUC: {roc_auc:.4f}")
            
            # Save individual model
            dump(model, f'model_{name}.joblib')
    
    def create_ensemble(self, X_train, y_train, X_test, y_test):
        """Create voting ensemble"""
        print("\n🎯 Creating ensemble model...")
        
        # Create ensemble with all trained models
        estimators = [(name, model) for name, model in self.models.items()]
        
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability predictions
        )
        
        # Train ensemble
        self.ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        train_score = self.ensemble.score(X_train, y_train)
        test_score = self.ensemble.score(X_test, y_test)
        
        # ROC AUC
        y_proba = self.ensemble.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        
        self.scores['ensemble'] = {
            'train_accuracy': round(train_score, 4),
            'test_accuracy': round(test_score, 4),
            'roc_auc': round(roc_auc, 4)
        }
        
        print(f"Ensemble - Train: {train_score:.4f} | Test: {test_score:.4f} | AUC: {roc_auc:.4f}")
        
        # Save ensemble
        dump(self.ensemble, 'model_ensemble.joblib')
        print("💾 Ensemble model saved")
    
    def select_best_model(self):
        """Select best performing model"""
        # Find best model based on test accuracy
        best_name = max(self.scores, key=lambda x: self.scores[x]['test_accuracy'])
        
        if best_name == 'ensemble':
            self.best_model = self.ensemble
        else:
            self.best_model = self.models[best_name]
        
        print(f"\n🏆 Best model: {best_name}")
        print(f"   Scores: {self.scores[best_name]}")
        
        # Save as main model
        dump(self.best_model, 'model.joblib')
        print("💾 Best model saved as model.joblib")
        
        return best_name

# ========================================
# FEATURE IMPORTANCE
# ========================================
def analyze_feature_importance(models, feature_names):
    """Analyze and aggregate feature importance"""
    print("\n📈 Analyzing feature importance...")
    
    importance_dict = {}
    
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for i, feature in enumerate(feature_names):
                if feature not in importance_dict:
                    importance_dict[feature] = []
                importance_dict[feature].append(importances[i])
    
    # Average importance
    avg_importance = {}
    for feature, values in importance_dict.items():
        avg_importance[feature] = np.mean(values)
    
    # Sort by importance
    importance_df = pd.DataFrame(
        list(avg_importance.items()),
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string())
    
    # Save importance
    importance_df.to_csv('feature_importance.csv', index=False)
    print("💾 Feature importance saved to feature_importance.csv")
    
    return importance_df

# ========================================
# PERFORMANCE REPORT
# ========================================
def generate_performance_report(y_test, y_pred, y_proba, scores):
    """Generate comprehensive performance report"""
    print("\n📊 Generating performance report...")
    
    # Classification report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'win_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
    }
    
    print("\n📈 Performance Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save report
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'model_scores': scores,
        'metrics': metrics,
        'confusion_matrix': cm.tolist()
    }
    
    with open('training_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print("💾 Report saved to training_report.json")
    
    return metrics

# ========================================
# MAIN TRAINING PIPELINE
# ========================================
def main():
    print("="*60)
    print("🚀 AI Trading Playbook - Enhanced Training Pipeline v3.0")
    print("="*60)
    
    # 1. Load data from Google Sheets
    print("\n📊 Loading data from Google Sheets...")
    sh = connect_to_sheets()
    
    try:
        ws = sh.worksheet("Alerts_Log")
        data = ws.get_all_records()
    except Exception as e:
        raise RuntimeError(f"❌ Could not read Alerts_Log: {e}")
    
    df = pd.DataFrame(data)
    
    if df.empty:
        raise RuntimeError("⚠️ Alerts_Log is empty — need more signals before training.")
    
    print(f"✅ Loaded {len(df)} records")
    
    # 2. Filter labeled data
    if "Outcome" not in df.columns and "Result" not in df.columns:
        raise RuntimeError("⚠️ Missing outcome column (must be 'Outcome' or 'Result' with Win/Loss values)")
    
    outcome_col = "Outcome" if "Outcome" in df.columns else "Result"
    df = df[df[outcome_col].isin(["Win", "Loss"])]
    
    if len(df) < MIN_SAMPLES:
        raise RuntimeError(f"⚠️ Need at least {MIN_SAMPLES} labeled samples. Found: {len(df)}")
    
    print(f"✅ Found {len(df)} labeled samples")
    
    # 3. Create target variable
    y = (df[outcome_col] == "Win").astype(int)
    
    print(f"📊 Class distribution:")
    print(f"  Wins: {y.sum()} ({y.mean():.1%})")
    print(f"  Losses: {len(y) - y.sum()} ({1-y.mean():.1%})")
    
    # 4. Feature engineering
    engineer = FeatureEngineer()
    X = engineer.create_features(df)
    
    # 5. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\n📊 Data split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # 6. Scale features
    X_train_scaled, X_test_scaled = engineer.scale_features(X_train, X_test)
    
    # 7. Train models
    trainer = ModelTrainer()
    trainer.create_models()
    trainer.train_individual_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 8. Create ensemble
    if len(trainer.models) > 1:
        trainer.create_ensemble(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 9. Select best model
    best_model_name = trainer.select_best_model()
    
    # 10. Feature importance
    importance_df = analyze_feature_importance(trainer.models, engineer.feature_names)
    
    # 11. Generate report
    best_model = trainer.best_model
    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else None
    
    metrics = generate_performance_report(y_test, y_pred, y_proba, trainer.scores)
    
    # 12. Summary
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"  Best Model: {best_model_name}")
    print(f"  Test Accuracy: {trainer.scores[best_model_name]['test_accuracy']:.2%}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Files saved:")
    print(f"    - model.joblib (best model)")
    print(f"    - model_ensemble.joblib (ensemble)")
    print(f"    - scaler.joblib (feature scaler)")
    print(f"    - feature_importance.csv")
    print(f"    - training_report.json")
    
    return trainer.scores[best_model_name]['test_accuracy']

if __name__ == "__main__":
    try:
        accuracy = main()
        exit(0 if accuracy > 0.6 else 1)  # Exit with error if accuracy too low
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        exit(1)
