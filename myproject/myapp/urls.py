# filepath: d:\Major project\myproject\myapp\urls.py
from django.urls import path
from .views import predict_image
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),  # Home page
    path('login/', views.login_view, name='login'),  # Login page
    path('signin/', views.signin_view, name='signin'),  # Sign-in page
    path('upload/', views.predict_image, name='prediction'),  # Prediction page
    path('predict/', predict_image, name='predict_image'),  # Prediction logic
    path('register_farmer/', views.register_farmer, name='register_farmer'),  # Farmer registration
    path('welcome/<str:name>/', views.welcome_page, name='welcome_page'),  # Welcome page
    path('about/', views.about, name='about'),  # About page
    path('feedback/', views.feedback, name='feedback'),
    path('feedback-stats/', views.feedback_stats, name='feedback_stats'),
    path('feedback-page/', views.feedback_page, name='feedback_page'),
    path('chatbot/', views.chatbot_view, name='chatbot'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)