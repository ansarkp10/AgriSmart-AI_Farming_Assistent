import os
import pandas as pd
import numpy as np

# Create dataset directory
os.makedirs('dataset', exist_ok=True)

# Generate fertilizer dataset
print("Creating fertilizer dataset...")

# Define crop parameters
crop_params = {
    'rice': {'n_range': (25, 35), 'p_range': (20, 28), 'k_range': (30, 40), 'fertilizer': 'Urea', 'base_qty': 60},
    'wheat': {'n_range': (38, 45), 'p_range': (32, 38), 'k_range': (42, 48), 'fertilizer': 'DAP', 'base_qty': 55},
    'maize': {'n_range': (30, 38), 'p_range': (26, 32), 'k_range': (36, 42), 'fertilizer': 'NPK', 'base_qty': 52},
    'cotton': {'n_range': (40, 45), 'p_range': (33, 38), 'k_range': (48, 52), 'fertilizer': 'MOP', 'base_qty': 48},
    'sugarcane': {'n_range': (48, 55), 'p_range': (42, 48), 'k_range': (52, 58), 'fertilizer': 'SSP', 'base_qty': 70},
    'vegetables': {'n_range': (28, 32), 'p_range': (22, 28), 'k_range': (32, 38), 'fertilizer': 'NPK', 'base_qty': 40},
    'fruits': {'n_range': (22, 28), 'p_range': (18, 23), 'k_range': (28, 33), 'fertilizer': 'SSP', 'base_qty': 35}
}

# Season parameters
season_params = {
    'kharif': {'temp_range': (26, 32), 'humidity_range': (70, 82), 'rainfall_range': (80, 220)},
    'rabi': {'temp_range': (18, 26), 'humidity_range': (60, 75), 'rainfall_range': (20, 60)},
    'zaid': {'temp_range': (30, 36), 'humidity_range': (65, 72), 'rainfall_range': (180, 250)}
}

# Crop-season mapping
crop_season_map = {
    'rice': ['kharif', 'rabi'],
    'wheat': ['rabi', 'kharif'],
    'maize': ['kharif', 'rabi'],
    'cotton': ['kharif', 'rabi'],
    'sugarcane': ['zaid', 'kharif'],
    'vegetables': ['rabi', 'zaid'],
    'fruits': ['kharif', 'rabi']
}

# Generate data
data = []
np.random.seed(42)

for crop, seasons in crop_season_map.items():
    crop_info = crop_params[crop]
    
    for season in seasons:
        season_info = season_params[season]
        
        # Generate 6-8 samples per crop-season combination
        for _ in range(np.random.randint(6, 9)):
            # Generate soil nutrient values
            n = np.random.uniform(*crop_info['n_range'])
            p = np.random.uniform(*crop_info['p_range'])
            k = np.random.uniform(*crop_info['k_range'])
            
            # Generate pH based on crop
            if crop in ['rice', 'maize']:
                ph = np.random.uniform(6.2, 6.8)
            elif crop in ['wheat', 'cotton']:
                ph = np.random.uniform(6.9, 7.3)
            else:
                ph = np.random.uniform(6.0, 7.6)
            
            # Generate environmental conditions
            temp = np.random.uniform(*season_info['temp_range'])
            humidity = np.random.uniform(*season_info['humidity_range'])
            rainfall = np.random.uniform(*season_info['rainfall_range'])
            soil_moisture = np.random.uniform(40, 72)
            
            # Calculate fertilizer quantity with variation
            base_qty = crop_info['base_qty']
            # Adjust based on nutrient levels
            if n < np.mean(crop_info['n_range']):
                qty_multiplier = 1.1
            elif n > np.mean(crop_info['n_range']) * 1.2:
                qty_multiplier = 0.9
            else:
                qty_multiplier = 1.0
            
            fertilizer_qty = round(base_qty * qty_multiplier * np.random.uniform(0.9, 1.1), 1)
            
            data.append([
                crop, season,
                round(n, 1), round(p, 1), round(k, 1), round(ph, 1),
                round(temp, 1), round(humidity, 1), round(rainfall, 1), round(soil_moisture, 1),
                crop_info['fertilizer'], fertilizer_qty
            ])

# Create DataFrame
columns = [
    'crop_type', 'season', 'nitrogen', 'phosphorus', 'potassium',
    'ph_level', 'temperature', 'humidity', 'rainfall', 'soil_moisture',
    'fertilizer_type', 'fertilizer_quantity_kg_per_acre'
]

fert_df = pd.DataFrame(data, columns=columns)

# Save fertilizer dataset
fert_df.to_csv('dataset/fertilizer_data.csv', index=False)
print(f"Fertilizer dataset created with {len(fert_df)} samples")

# Create irrigation dataset
print("\nCreating irrigation dataset...")

irr_data = []

for _, row in fert_df.iterrows():
    # Water requirement base on crop type
    crop_water_base = {
        'rice': 8500,
        'wheat': 4500,
        'maize': 6000,
        'cotton': 5500,
        'sugarcane': 12000,
        'vegetables': 3500,
        'fruits': 3000
    }
    
    base_water = crop_water_base[row['crop_type']]
    
    # Adjust based on conditions
    # More water if temperature is high
    temp_factor = 1 + (row['temperature'] - 25) * 0.02
    
    # Less water if humidity is high
    humidity_factor = 1 - (row['humidity'] - 50) * 0.005
    
    # Less water if rainfall is high
    rainfall_factor = max(0.7, 1 - (row['rainfall'] / 300))
    
    # Less water if soil moisture is high
    soil_factor = 1 - (row['soil_moisture'] / 100) * 0.3
    
    # Calculate final water requirement
    water_required = base_water * temp_factor * humidity_factor * rainfall_factor * soil_factor
    water_required = round(water_required * np.random.uniform(0.95, 1.05))
    
    irr_data.append([
        row['crop_type'], row['season'],
        row['nitrogen'], row['phosphorus'], row['potassium'], row['ph_level'],
        row['temperature'], row['humidity'], row['rainfall'], row['soil_moisture'],
        water_required
    ])

# Create irrigation DataFrame
irr_columns = [
    'crop_type', 'season', 'nitrogen', 'phosphorus', 'potassium',
    'ph_level', 'temperature', 'humidity', 'rainfall', 'soil_moisture',
    'water_required_liters_per_day_per_acre'
]

irr_df = pd.DataFrame(irr_data, columns=irr_columns)

# Save irrigation dataset
irr_df.to_csv('dataset/irrigation_data.csv', index=False)
print(f"Irrigation dataset created with {len(irr_df)} samples")

print("\n" + "=" * 50)
print("DATASETS CREATED SUCCESSFULLY!")
print("=" * 50)
print("\nDataset Summary:")
print(f"Fertilizer dataset: {len(fert_df)} records")
print(f"Irrigation dataset: {len(irr_df)} records")
print("\nSample from fertilizer dataset:")
print(fert_df.head())
print("\nSample from irrigation dataset:")
print(irr_df.head())