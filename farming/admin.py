from django.contrib import admin
from .models import FarmerProfile, SoilData, Recommendation

@admin.register(FarmerProfile)
class FarmerProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'farm_name', 'location', 'farm_size']
    search_fields = ['user__username', 'farm_name', 'location']

@admin.register(SoilData)
class SoilDataAdmin(admin.ModelAdmin):
    list_display = ['farmer', 'crop_type', 'season', 'nitrogen', 'phosphorus', 'recorded_date']
    list_filter = ['crop_type', 'season', 'recorded_date']
    search_fields = ['farmer__username']

@admin.register(Recommendation)
class RecommendationAdmin(admin.ModelAdmin):
    list_display = ['soil_data', 'fertilizer_type', 'fertilizer_quantity', 'water_required', 'recommendation_date']
    list_filter = ['fertilizer_type', 'recommendation_date']