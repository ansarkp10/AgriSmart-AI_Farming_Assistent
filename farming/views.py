from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth import login, logout
from .forms import UserRegisterForm, SoilDataForm
from .models import SoilData, Recommendation, FarmerProfile
from .ml_models import ml_model
from django.db.models import Avg

def index(request):
    """Home page"""
    return render(request, 'farming/index.html')

def register(request):
    """User registration view"""
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            
            # Create farmer profile
            FarmerProfile.objects.create(
                user=user,
                farm_name=form.cleaned_data['farm_name'],
                location=form.cleaned_data['location'],
                farm_size=form.cleaned_data['farm_size'],
                contact_number=form.cleaned_data['contact_number']
            )
            
            messages.success(request, 'Account created successfully! Please log in.')
            return redirect('login')
    else:
        form = UserRegisterForm()
    
    return render(request, 'farming/register.html', {'form': form})

@login_required
def dashboard(request):
    """User dashboard"""
    user = request.user
    soil_data_list = SoilData.objects.filter(farmer=user).order_by('-recorded_date')[:5]
    
    try:
        profile = FarmerProfile.objects.get(user=user)
    except FarmerProfile.DoesNotExist:
        profile = None
    
    return render(request, 'farming/dashboard.html', {
        'profile': profile,
        'soil_data_list': soil_data_list
    })

@login_required
def fertilizer_recommendation(request):
    """Fertilizer recommendation view"""
    if request.method == 'POST':
        form = SoilDataForm(request.POST)
        if form.is_valid():
            # Save soil data
            soil_data = form.save(commit=False)
            soil_data.farmer = request.user
            soil_data.save()
            
            # Prepare data for ML prediction
            data = {
                'nitrogen': soil_data.nitrogen,
                'phosphorus': soil_data.phosphorus,
                'potassium': soil_data.potassium,
                'ph_level': soil_data.ph_level,
                'temperature': soil_data.temperature,
                'humidity': soil_data.humidity,
                'rainfall': soil_data.rainfall,
                'soil_moisture': soil_data.soil_moisture,
                'crop_type': soil_data.crop_type,
                'season': soil_data.season,
            }
            
            # Get predictions
            fertilizer_type, fertilizer_quantity = ml_model.predict_fertilizer(data)
            water_required = ml_model.predict_irrigation(data)
            
            # Save recommendation
            recommendation = Recommendation.objects.create(
                soil_data=soil_data,
                fertilizer_type=fertilizer_type,
                fertilizer_quantity=fertilizer_quantity,
                water_required=water_required,
                notes=f"Recommendation for {soil_data.crop_type} in {soil_data.season} season."
            )
            
            return render(request, 'farming/result.html', {
                'soil_data': soil_data,
                'fertilizer_type': fertilizer_type,
                'fertilizer_quantity': fertilizer_quantity,
                'water_required': water_required,
                'recommendation': recommendation,
            })
    else:
        form = SoilDataForm()
    
    return render(request, 'farming/fertilizer.html', {'form': form})

@login_required
def irrigation_recommendation(request):
    """Irrigation recommendation view"""
    # This can be a simplified version or redirect to fertilizer page
    return redirect('fertilizer_recommendation')

@login_required
def history(request):
    """View recommendation history"""
    user = request.user
    recommendations = Recommendation.objects.filter(
        soil_data__farmer=user
    ).select_related('soil_data').order_by('-recommendation_date')
    
    # Calculate averages
    if recommendations.exists():
        avg_fertilizer = round(recommendations.aggregate(Avg('fertilizer_quantity'))['fertilizer_quantity__avg'], 2)
        avg_water = round(recommendations.aggregate(Avg('water_required'))['water_required__avg'], 2)
    else:
        avg_fertilizer = None
        avg_water = None
    
    return render(request, 'farming/history.html', {
        'recommendations': recommendations,
        'avg_fertilizer': avg_fertilizer,
        'avg_water': avg_water,
    })

@login_required
def logout_view(request):
    """Logout view"""
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('index')