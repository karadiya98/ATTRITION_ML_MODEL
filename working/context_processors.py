# your_app/context_processors.py

def user_profile(request):
    profile_name = request.session.get('profile_name', None)
    companyname= request.session.get('companyname', None)
    return {'profile_name': profile_name,'companyname':companyname}
