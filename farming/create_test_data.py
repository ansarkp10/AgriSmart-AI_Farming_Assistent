import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agrismart_project.settings')
django.setup()

from django.contrib.auth.models import User
from farming.models import FarmerProfile, SoilData, Recommendation

# Create test user
user, created = User.objects.get_or_create(
    username='testfarmer',
    defaults={
        'email': 'farmer@test.com',
        'is_staff': False,
        'is_superuser': False
    }
)
if created:
    user.set_password('test123')
    user.save()

# Create farmer profile
profile, created = FarmerProfile.objects.get_or_create(
    user=user,
    defaults={
        'farm_name': 'Green Valley Farm',
        'location': 'Punjab, India',
        'farm_size': 10.5,
        'contact_number': '+91 9876543210'
    }
)

print("Test data created successfully!")
print(f"Username: testfarmer")
print(f"Password: test123")