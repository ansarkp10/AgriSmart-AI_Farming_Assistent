import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'ml_models')

class AgriSmartML:
    def __init__(self):
        self.fertilizer_model = None
        self.irrigation_model = None
        self.encoders = None
        self.load_models()
    
    def load_models(self):
        """Load trained models if they exist, otherwise create fallback"""
        try:
            if os.path.exists(os.path.join(MODEL_DIR, 'fertilizer_model.joblib')):
                self.fertilizer_model = joblib.load(os.path.join(MODEL_DIR, 'fertilizer_model.joblib'))
            if os.path.exists(os.path.join(MODEL_DIR, 'irrigation_model.joblib')):
                self.irrigation_model = joblib.load(os.path.join(MODEL_DIR, 'irrigation_model.joblib'))
            if os.path.exists(os.path.join(MODEL_DIR, 'label_encoder.joblib')):
                self.encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
            print("ML models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Creating fallback models...")
            self.create_fallback_models()
    
    def create_fallback_models(self):
        """Create simple models for demonstration"""
        # Create a simple Random Forest classifier for fertilizer
        self.fertilizer_model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Create a simple Random Forest regressor for irrigation
        self.irrigation_model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        # Create encoders
        self.encoders = {
            'crop_encoder': LabelEncoder(),
            'season_encoder': LabelEncoder(),
            'fertilizer_encoder': LabelEncoder()
        }
        
        # Fit with sample data
        self.encoders['crop_encoder'].fit(['rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'vegetables', 'fruits'])
        self.encoders['season_encoder'].fit(['kharif', 'rabi', 'zaid'])
        self.encoders['fertilizer_encoder'].fit(['Urea', 'DAP', 'MOP', 'NPK', 'SSP'])
    
    def calculate_fertilizer_quantity(self, data, fertilizer_type):
        """Calculate fertilizer quantity based on soil parameters"""
        base_quantity = 50  # Base quantity in kg/acre
        
        # Adjust based on nutrient levels
        if data['nitrogen'] < 20:
            base_quantity *= 1.3
        elif data['nitrogen'] > 40:
            base_quantity *= 0.7
            
        if data['phosphorus'] < 10:
            base_quantity *= 1.2
        elif data['phosphorus'] > 30:
            base_quantity *= 0.8
            
        if data['potassium'] < 15:
            base_quantity *= 1.1
        elif data['potassium'] > 35:
            base_quantity *= 0.9
        
        # Adjust based on crop type
        crop_factors = {
            'rice': 1.2, 'wheat': 1.0, 'maize': 1.1,
            'cotton': 0.9, 'sugarcane': 1.3, 'vegetables': 0.8, 'fruits': 0.7
        }
        crop_factor = crop_factors.get(data['crop_type'], 1.0)
        
        quantity = round(base_quantity * crop_factor, 2)
        
        return quantity
    
    def fallback_fertilizer_prediction(self, data):
        """Fallback fertilizer prediction if ML model fails"""
        # Simple rule-based prediction
        n = data['nitrogen']
        
        if n < 25:
            return 'Urea', 60
        elif n < 35:
            return 'DAP', 55
        elif n < 45:
            return 'NPK', 50
        else:
            return 'SSP', 45
    
    def fallback_irrigation_prediction(self, data):
        """Fallback irrigation prediction if ML model fails"""
        # Simple rule-based prediction
        crop_water_needs = {
            'rice': 8500, 'wheat': 4500, 'maize': 6000,
            'cotton': 5500, 'sugarcane': 12000, 'vegetables': 3500, 'fruits': 3000
        }
        crop_factor = crop_water_needs.get(data['crop_type'], 5000)
        
        # Adjust for temperature
        temp_factor = 1 + (data['temperature'] - 25) * 0.02
        
        return max(0, round(crop_factor * temp_factor, 2))
    
    def predict_fertilizer(self, data):
        """Predict fertilizer type and quantity"""
        try:
            if self.fertilizer_model is not None and self.encoders is not None:
                # Encode categorical features
                crop_encoded = self.encoders['crop_encoder'].transform([data['crop_type']])[0]
                season_encoded = self.encoders['season_encoder'].transform([data['season']])[0]
                
                # Prepare features
                features = np.array([
                    data['nitrogen'], data['phosphorus'], data['potassium'],
                    data['ph_level'], data['temperature'], data['humidity'],
                    data['rainfall'], data['soil_moisture'],
                    crop_encoded, season_encoded
                ]).reshape(1, -1)
                
                # Predict
                fert_pred = self.fertilizer_model.predict(features)[0]
                fertilizer_type = self.encoders['fertilizer_encoder'].inverse_transform([fert_pred])[0]
                
                # Calculate quantity
                quantity = self.calculate_fertilizer_quantity(data, fertilizer_type)
                
                return fertilizer_type, quantity
            else:
                return self.fallback_fertilizer_prediction(data)
                
        except Exception as e:
            print(f"Fertilizer prediction error: {e}")
            return self.fallback_fertilizer_prediction(data)
    
    def predict_irrigation(self, data):
        """Predict water requirement"""
        try:
            if self.irrigation_model is not None and self.encoders is not None:
                # Encode categorical features
                crop_encoded = self.encoders['crop_encoder'].transform([data['crop_type']])[0]
                season_encoded = self.encoders['season_encoder'].transform([data['season']])[0]
                
                # Prepare features
                features = np.array([
                    data['nitrogen'], data['phosphorus'], data['potassium'],
                    data['ph_level'], data['temperature'], data['humidity'],
                    data['rainfall'], data['soil_moisture'],
                    crop_encoded, season_encoded
                ]).reshape(1, -1)
                
                # Predict
                water_required = self.irrigation_model.predict(features)[0]
                
                return max(0, round(water_required, 2))
            else:
                return self.fallback_irrigation_prediction(data)
                
        except Exception as e:
            print(f"Irrigation prediction error: {e}")
            return self.fallback_irrigation_prediction(data)

# Create a global instance - THIS IS IMPORTANT
ml_model = AgriSmartML()