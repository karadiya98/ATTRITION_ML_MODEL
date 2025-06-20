from django.contrib import admin # type: ignore
from django.urls import path # type: ignore
from .views import home,HOME,DATA,signup,login,about_logout,userdata_view,logout,predict_attrition


urlpatterns = [
    path('admin/', admin.site.urls),
    path('',home,name='home'),
    path('HOME',HOME,name='HOME1'),
    path('DATA',DATA,name='DATA'),
    path('signup',signup,name='signup'),
    path('login',login,name='login'),
    path('about',about_logout,name='login'),
    path('datainserted',userdata_view,name='datainserted'),
    path('logout',logout,name='logout'),
    path('predict/',predict_attrition, name='predict_attrition')
]
