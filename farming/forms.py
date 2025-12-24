from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import FarmerProfile, SoilData

class UserRegisterForm(UserCreationForm):
    email = forms.EmailField()
    farm_name = forms.CharField(max_length=100)
    location = forms.CharField(max_length=100)
    farm_size = forms.FloatField()
    contact_number = forms.CharField(max_length=15)
    
    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2', 
                 'farm_name', 'location', 'farm_size', 'contact_number']

class SoilDataForm(forms.ModelForm):
    class Meta:
        model = SoilData
        fields = ['crop_type', 'season', 'nitrogen', 'phosphorus', 
                 'potassium', 'ph_level', 'temperature', 'humidity', 
                 'rainfall', 'soil_moisture']
        widgets = {
            'crop_type': forms.Select(attrs={'class': 'form-control'}),
            'season': forms.Select(attrs={'class': 'form-control'}),
            'nitrogen': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'phosphorus': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'potassium': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'ph_level': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'temperature': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'humidity': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'rainfall': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'soil_moisture': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
        }