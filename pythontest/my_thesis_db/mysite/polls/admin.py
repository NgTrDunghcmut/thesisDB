from django.contrib import admin
from .models import *
from django.contrib.admin.sites import AdminSite


# Create a custom admin site
custom_admin_site = AdminSite(name='customadmin')
# Register your models here.
custom_admin_site.register(Device)
custom_admin_site.register(Data)
#admin.site.urls()