from django.db import models
from django.contrib.auth.models import User

class FarmerProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    farm_name = models.CharField(max_length=100)
    location = models.CharField(max_length=100)
    farm_size = models.FloatField(help_text="Size in acres")
    contact_number = models.CharField(max_length=15)
    
    def __str__(self):
        return f"{self.user.username} - {self.farm_name}"

class SoilData(models.Model):
    CROP_CHOICES = [
        ('rice', 'Rice'),
        ('wheat', 'Wheat'),
        ('maize', 'Maize'),
        ('cotton', 'Cotton'),
        ('sugarcane', 'Sugarcane'),
        ('vegetables', 'Vegetables'),
        ('fruits', 'Fruits'),
    ]
    
    SEASON_CHOICES = [
        ('kharif', 'Kharif'),
        ('rabi', 'Rabi'),
        ('zaid', 'Zaid'),
    ]
    
    farmer = models.ForeignKey(User, on_delete=models.CASCADE)
    crop_type = models.CharField(max_length=20, choices=CROP_CHOICES)
    season = models.CharField(max_length=20, choices=SEASON_CHOICES)
    nitrogen = models.FloatField(help_text="N value in mg/kg")
    phosphorus = models.FloatField(help_text="P value in mg/kg")
    potassium = models.FloatField(help_text="K value in mg/kg")
    ph_level = models.FloatField(help_text="pH value")
    temperature = models.FloatField(help_text="Temperature in °C")
    humidity = models.FloatField(help_text="Humidity in %")
    rainfall = models.FloatField(help_text="Rainfall in mm")
    soil_moisture = models.FloatField(help_text="Soil moisture in %")
    recorded_date = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.farmer.username} - {self.crop_type} - {self.recorded_date.date()}"

class Recommendation(models.Model):
    soil_data = models.OneToOneField(SoilData, on_delete=models.CASCADE)
    fertilizer_type = models.CharField(max_length=50)
    fertilizer_quantity = models.FloatField(help_text="Quantity in kg/acre")
    water_required = models.FloatField(help_text="Water required in liters/day")
    recommendation_date = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True)
    
    def __str__(self):
        return f"Recommendation for {self.soil_data}"