from django.contrib import admin
from .models.signup import User_Detail


class usertableview(admin.ModelAdmin):
    list=['user_id','image','companyname','companyphone','gmail','user_name','password','department']





admin.site.register(User_Detail,usertableview)