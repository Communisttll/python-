from django.urls import path
from . import views

app_name = 'retrieval'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload, name='upload'),
    path('gallery/', views.gallery, name='gallery'),
    path('api/search/', views.api_search, name='api_search'),
    path('api/analyze/', views.api_analyze, name='api_analyze'),
]