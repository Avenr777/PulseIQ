from django.urls import path
from . import views
from django.shortcuts import render, redirect
urlpatterns = [
    path('', views.land, name='land'),
    path('home/', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('water/', views.water, name='water'),
    path('electricity/', views.electricity, name='electricity'),
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    path("forecast/", views.electricity_forecast, name="forecast"),
]