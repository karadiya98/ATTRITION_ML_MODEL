# Assuming this file is your_app_name/models/signup.py or your_app_name/models.py
# If it's a subfolder, make sure there's an __init__.py inside 'models' folder.

from django.db import models
from django.contrib.auth.hashers import make_password, check_password # Import for hashing/checking passwords

class User_Detail(models.Model):

    user_id = models.AutoField(primary_key=True) 
    image = models.ImageField(upload_to='profile_pics/', blank=True, null=True)
    companyname = models.CharField(max_length=100)
    companyphone = models.CharField(max_length=20)
    gmail = models.EmailField(max_length=255, unique=True) # Emails should be unique
    user_name = models.CharField(max_length=100, unique=True)
    password = models.CharField(max_length=128) # Django's default hash length is 128
    department = models.CharField(max_length=100, blank=True, null=True) # Assuming optional



    def __str__(self):
        return self.user_name

    # Optional: Custom method to set password safely
    def set_password(self, raw_password):
        self.password = make_password(raw_password)

    # Optional: Custom method to check password safely
    def check_password(self, raw_password):
        return check_password(raw_password, self.password)

    # If your model file is NOT named models.py directly inside the app folder,
    # you might need this Meta class to help Django find it.
    # Replace 'your_app_name' with the actual name of your Django app.
    # class Meta:
    #     app_label = 'your_app_name'