from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('register/', views.register, name='register'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('fertilizer/', views.fertilizer_recommendation, name='fertilizer_recommendation'),
    path('irrigation/', views.irrigation_recommendation, name='irrigation_recommendation'),
    path('history/', views.history, name='history'),
    path('logout/', views.logout_view, name='logout'),
]