import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
import os

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'ml_models')

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# File paths
FERTILIZER_DATA_PATH = os.path.join(DATASET_DIR, 'fertilizer_data.csv')
IRRIGATION_DATA_PATH = os.path.join(DATASET_DIR, 'irrigation_data.csv')
FERTILIZER_MODEL_PATH = os.path.join(MODEL_DIR, 'fertilizer_model.joblib')
IRRIGATION_MODEL_PATH = os.path.join(MODEL_DIR, 'irrigation_model.joblib')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

def load_and_preprocess_data():
    """Load and preprocess the datasets"""
    
    # Load fertilizer data
    print("Loading fertilizer data...")
    fert_df = pd.read_csv(FERTILIZER_DATA_PATH)
    
    # Load irrigation data
    print("Loading irrigation data...")
    irr_df = pd.read_csv(IRRIGATION_DATA_PATH)
    
    # Encode categorical variables for fertilizer data
    label_encoder = LabelEncoder()
    
    # Encode crop type
    fert_df['crop_encoded'] = label_encoder.fit_transform(fert_df['crop_type'])
    irr_df['crop_encoded'] = label_encoder.transform(irr_df['crop_type'])
    
    # Encode season
    season_encoder = LabelEncoder()
    fert_df['season_encoded'] = season_encoder.fit_transform(fert_df['season'])
    irr_df['season_encoded'] = season_encoder.transform(irr_df['season'])
    
    # Encode fertilizer type
    fertilizer_encoder = LabelEncoder()
    fert_df['fertilizer_encoded'] = fertilizer_encoder.fit_transform(fert_df['fertilizer_type'])
    
    return fert_df, irr_df, fertilizer_encoder, label_encoder, season_encoder

def train_fertilizer_model(fert_df, fertilizer_encoder):
    """Train the fertilizer recommendation model"""
    print("\nTraining fertilizer recommendation model...")
    
    # Features for fertilizer prediction
    X_fert = fert_df[[
        'nitrogen', 'phosphorus', 'potassium', 'ph_level',
        'temperature', 'humidity', 'rainfall', 'soil_moisture',
        'crop_encoded', 'season_encoded'
    ]].values
    
    # Target: fertilizer type
    y_fert = fert_df['fertilizer_encoded'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_fert, y_fert, test_size=0.2, random_state=42
    )
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Fertilizer Model Accuracy: {accuracy:.2%}")
    
    # Feature importance
    feature_names = ['N', 'P', 'K', 'pH', 'Temp', 'Humidity', 'Rainfall', 'Soil Moisture', 'Crop', 'Season']
    importances = model.feature_importances_
    
    print("\nFeature Importances:")
    for name, importance in zip(feature_names, importances):
        print(f"  {name}: {importance:.3f}")
    
    return model

def train_irrigation_model(irr_df):
    """Train the irrigation water requirement model"""
    print("\nTraining irrigation water requirement model...")
    
    # Features for irrigation prediction
    X_irr = irr_df[[
        'nitrogen', 'phosphorus', 'potassium', 'ph_level',
        'temperature', 'humidity', 'rainfall', 'soil_moisture',
        'crop_encoded', 'season_encoded'
    ]].values
    
    # Target: water requirement
    y_irr = irr_df['water_required_liters_per_day_per_acre'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_irr, y_irr, test_size=0.2, random_state=42
    )
    
    # Train Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Irrigation Model RMSE: {rmse:.2f} liters/day")
    print(f"Average prediction error: {rmse/np.mean(y_test):.2%}")
    
    return model

def save_models(fert_model, irr_model, fertilizer_encoder, label_encoder, season_encoder):
    """Save trained models and encoders"""
    print("\nSaving models...")
    
    # Save models
    joblib.dump(fert_model, FERTILIZER_MODEL_PATH)
    joblib.dump(irr_model, IRRIGATION_MODEL_PATH)
    
    # Save encoders in a dictionary
    encoders = {
        'fertilizer_encoder': fertilizer_encoder,
        'crop_encoder': label_encoder,
        'season_encoder': season_encoder
    }
    joblib.dump(encoders, ENCODER_PATH)
    
    print(f"Models saved to {MODEL_DIR}")

def create_sample_predictions():
    """Create sample predictions to verify model works"""
    print("\nCreating sample predictions...")
    
    # Load models
    fert_model = joblib.load(FERTILIZER_MODEL_PATH)
    irr_model = joblib.load(IRRIGATION_MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    
    # Sample input data
    samples = [
        {
            'crop_type': 'rice',
            'season': 'kharif',
            'nitrogen': 30,
            'phosphorus': 25,
            'potassium': 35,
            'ph_level': 6.5,
            'temperature': 28,
            'humidity': 75,
            'rainfall': 150,
            'soil_moisture': 45
        },
        {
            'crop_type': 'wheat',
            'season': 'rabi',
            'nitrogen': 40,
            'phosphorus': 35,
            'potassium': 45,
            'ph_level': 7.0,
            'temperature': 22,
            'humidity': 65,
            'rainfall': 50,
            'soil_moisture': 60
        }
    ]
    
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i} Prediction:")
        print(f"Crop: {sample['crop_type']}, Season: {sample['season']}")
        
        # Prepare features
        crop_encoded = encoders['crop_encoder'].transform([sample['crop_type']])[0]
        season_encoded = encoders['season_encoder'].transform([sample['season']])[0]
        
        features = np.array([
            sample['nitrogen'],
            sample['phosphorus'],
            sample['potassium'],
            sample['ph_level'],
            sample['temperature'],
            sample['humidity'],
            sample['rainfall'],
            sample['soil_moisture'],
            crop_encoded,
            season_encoded
        ]).reshape(1, -1)
        
        # Fertilizer prediction
        fert_pred_encoded = fert_model.predict(features)[0]
        fertilizer_type = encoders['fertilizer_encoder'].inverse_transform([fert_pred_encoded])[0]
        
        # Irrigation prediction
        water_required = irr_model.predict(features)[0]
        
        print(f"  Recommended Fertilizer: {fertilizer_type}")
        print(f"  Water Required: {water_required:.0f} liters/day/acre")

def main():
    """Main training function"""
    print("=" * 50)
    print("AGRISMART ML MODEL TRAINING")
    print("=" * 50)
    
    try:
        # Load and preprocess data
        fert_df, irr_df, fertilizer_encoder, label_encoder, season_encoder = load_and_preprocess_data()
        
        print(f"\nDataset Statistics:")
        print(f"Fertilizer samples: {len(fert_df)}")
        print(f"Irrigation samples: {len(irr_df)}")
        
        # Train models
        fert_model = train_fertilizer_model(fert_df, fertilizer_encoder)
        irr_model = train_irrigation_model(irr_df)
        
        # Save models
        save_models(fert_model, irr_model, fertilizer_encoder, label_encoder, season_encoder)
        
        # Create sample predictions
        create_sample_predictions()
        
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()