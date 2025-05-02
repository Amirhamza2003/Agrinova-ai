# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.utils import load_img, img_to_array

# Define model paths based on crop type
# model_paths = {
#     'corn': r'D:\Major project\myproject\models\corn_disease_model (2).h5',
#     'potato': r'D:\Major project\myproject\models\potato_model.h5',
#     'rice': r'myproject/models/rice1_disease_model.h5',
#     'wheat': r'D:\Major project\myproject\models\wheat_disease_model.h5',
#     'sugarcane': r'D:\Major project\myproject\models\sugarcane1_disease_model.h5',
# }
model_paths = {
    'corn': r'D:\Major project\myproject\models\corn_disease_model (2).h5',
    'potato': r'D:\Major project\myproject\models\potato_model.h5',
    'rice': r'D:\Major project\myproject\models\rice1_disease_model.h5',  # Updated to absolute path
    'wheat': r'D:\Major project\myproject\models\wheat1_disease_model.h5',  # Corrected file name
    'sugarcane': r'D:\Major project\myproject\models\sugarcane1_disease_model.h5',
}
# Define class labels per crop
class_labels = {
    'corn': [
        {"label": "Corn - Common Rust", "description": "Common rust in corn...", "treatment": "To treat common rust..."},
        {"label": "Corn - Gray Leaf Spot", "description": "Gray leaf spot...", "treatment": "Gray leaf spot in corn..."},
        {"label": "Corn - Healthy", "description": "Perfect, ready for selling.", "treatment": None},
        {"label": "Corn - Northern Leaf Blight", "description": "Northern leaf blight...", "treatment": "To treat Northern..."}
    ],
    'potato': [
        {"label": "Potato - Early Blight", "description": "Potato early blight...", "treatment": "Managed by using..."},
        {"label": "Potato - Healthy", "description": "Perfect, ready to sell.", "treatment": None},
        {"label": "Potato - Late Blight", "description": "It affects leaves...", "treatment": "Controlled by eliminating..."}
     ], 
    'rice': [
        {"label": "Rice - Brown Spot", "description": "Rice brown spot, caused by the fungus Cochliobolus miyabeanus, is a fungal disease that affects rice plants, leading to significant yield losses.", "treatment": "Manage brown spot disease in rice by using resistant varieties, maintaining good field hygiene, and applying fungicides if necessary."},
        {"label": "Rice - Healthy", "description": "Perfect, ready for selling.", "treatment": None},
        {"label": "Rice - Leaf Blast", "description": "Blast is caused by the fungus Magnaporthe oryzae and can affect all above-ground parts of a rice plant.", "treatment": "Manage through cultural practices, fungicide applications, and the use of blast-resistant rice varieties."},
        {"label": "Rice - Neck Blast", "description": "Blast caused by the fungus Magnaporthe oryzae can affect the neck and other parts of the rice plant.", "treatment": "Preventive measures include planting less susceptible varieties, burning straw and stubbles, and applying fungicides."}
    ],
    'wheat': [
        {"label": "Wheat - Brown Rust", "description": "Brown rust in wheat is caused by Puccinia triticina and occurs wherever wheat is grown.", "treatment": "Managing leaf rust in wheat involves using resistant varieties, proper fertilization, and timely fungicide applications."},
        {"label": "Wheat - Healthy", "description": "Perfect, ready for selling.", "treatment": None},
        {"label": "Wheat - Yellow Rust", "description": "Wheat yellow rust, also known as stripe rust, is a fungal disease caused by Puccinia striiformis f. sp. tritici.", "treatment": "Preventative measures include using resistant wheat varieties, crop rotation, and fungicide applications when the disease is detected early."}
    ],
    
    'sugarcane': [
        {"label": "Sugarcane - Red Rot", "description": "Red rot is a devastating fungal disease in sugarcane caused by Colletotrichum falcatum.", "treatment": "Treat red rot with cultural practices, sanitation, and fungicides like thiophanate methyl or Carbendazim."},
        {"label": "Sugarcane - Healthy", "description": "Perfect, ready for selling.", "treatment": None},
        {"label": "Sugarcane - Bacterial Blight", "description": "Sugarcane bacterial blight is caused by Acidovorax avenae and other bacteria.", "treatment": "Treat bacterial blight with resistant varieties, proper drainage, and copper-based fungicides or antibiotics like streptomycin."}
    ]
}

# Home page view
def home(request):
    return render(request, 'home.html')

# Login view
from django.shortcuts import redirect

def login_view(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']
        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)
            next_url = request.GET.get('next', 'home')  # Redirect to 'next' or home page
            return redirect(next_url)
        else:
            messages.error(request, 'Invalid email or password.')
    return render(request, 'login.html')

# Sign-in view
def signin_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Account created successfully. You can now log in.')
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'signin.html', {'form': form})

# Prediction view (requires login)
@login_required
def prediction_view(request):
    return render(request, 'upload.html')  # Render the upload page for crop status prediction

# Upload and predict image
@login_required
def predict_image(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_obj = form.save()
            img_path = os.path.join(settings.MEDIA_ROOT, str(img_obj.image))
            crop_type = form.cleaned_data['crop_type']

            selected_model_path = model_paths.get(crop_type)
            if not selected_model_path or not os.path.exists(selected_model_path):
                return render(request, "upload.html", {"form": form, "error": f"Model for '{crop_type}' not found."})

            try:
                model = tf.keras.models.load_model(selected_model_path)
                processed_img = preprocess_image(img_path)
                prediction = model.predict(processed_img)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)

                disease_info = class_labels.get(crop_type, [])[predicted_class]
                disease_name = disease_info["label"]
                description = disease_info["description"]
                treatment = disease_info["treatment"]

                return render(request, "upload.html", {
                    "form": form,
                    "image_url": img_obj.image.url,
                    "prediction": {
                        "crop": crop_type.title(),
                        "disease": disease_name,
                        "confidence": f"{confidence * 100:.2f}%",
                        "description": description,
                        "treatment": treatment
                    },
                    "show_sell_button": disease_name.endswith("Healthy")
                })
            except Exception as e:
                return render(request, "upload.html", {"form": form, "error": f"Prediction error: {str(e)}"})
    else:
        form = ImageUploadForm()
    return render(request, "upload.html", {"form": form})

# Preprocessing function
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Farmer registration view
def register_farmer(request):
    if request.method == "POST":
        name = request.POST.get("name")
        # Process the form data (e.g., save to database)
        return redirect("welcome_page", name=name)
    return render(request, "farmer_registration.html")

# Welcome page view
def welcome_page(request, name):
    return render(request, "welcome.html", {"name": name})
# filepath: d:\Major project\myproject\myapp\views.py
from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

@login_required
def prediction_view(request):
    return render(request, 'upload.html')  # Render the upload page for crop status prediction

# filepath: d:\Major project\myproject\myapp\templatetags\custom_filters.py
from django import template

register = template.Library()

@register.filter(name='add_class')
def add_class(field, css_class):
    return field.as_widget(attrs={"class": css_class})

from django.shortcuts import render

def about(request):
    return render(request, 'about.html')



from django.core.mail import send_mail
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Feedback
import socket

def feedback(request):
    if request.method == 'POST':
        feedback_text = request.POST.get('feedback')
        user_email = request.POST.get('email')  # Add an email field in the form

        # Save feedback to the database
        feedback_entry = Feedback.objects.create(text=feedback_text, email=user_email)

        try:
            # Send a thank-you email
            send_mail(
                'Thank You for Your Feedback',
                'Thank you for using AgriNova AI. We appreciate your feedback and will use it to improve our services.',
                'amirbackup2023@gmail.com',  # Replace with your email
                [user_email],
                fail_silently=False,
            )
            # Update email_sent status
            feedback_entry.email_sent = True
            feedback_entry.save()

            messages.success(request, 'Thank you for your feedback! An email has been sent to you.')
        except socket.error as e:
            # Handle connection errors gracefully
            return JsonResponse({'error': f'Connection error: {e}'}, status=500)
        except Exception as e:
            # Handle other exceptions
            return JsonResponse({'error': f'An error occurred: {e}'}, status=500)

        return redirect('home')  # Redirect to the home page
    return redirect('home')

from transformers import AutoTokenizer, AutoModelForMaskedLM,AutoModelForCausalLM
import torch

def fill_mask_view(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text', '[MASK] is the practice of cultivating plants and livestock.')
        
        # Load the model and tokenizer
        # Load model directly


        tokenizer = AutoTokenizer.from_pretrained("Kobi-01/gemma_2b_agriculture")
        model = AutoModelForCausalLM.from_pretrained("Kobi-01/gemma_2b_agriculture")
        # tokenizer = AutoTokenizer.from_pretrained("recobo/agriculture-bert-uncased")
        # model = AutoModelForMaskedLM.from_pretrained("recobo/agriculture-bert-uncased")
        
        # Tokenize the input
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # Perform the prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.topk(outputs.logits[0, inputs["input_ids"][0] == tokenizer.mask_token_id], k=5).indices
        
        # Decode the predictions
        results = [tokenizer.decode(pred.item()).strip() for pred in predictions]
        
        return render(request, 'fill_mask.html', {'results': results, 'input_text': input_text})
    
    return render(request, 'fill_mask.html')




from django.shortcuts import render
from .models import Feedback

def feedback_stats(request):
    total_feedback = Feedback.objects.count()
    emails_sent = Feedback.objects.filter(email_sent=True).count()
    return render(request, 'feedback_stats.html', {
        'total_feedback': total_feedback,
        'emails_sent': emails_sent,
    })

from django.shortcuts import render

def feedback_page(request):
    return render(request, 'feedback_page.html')

import os
import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
@csrf_exempt
def chatbot_view(request):
    if request.method == 'POST':
        user_message = json.loads(request.body).get('message', '')
        if not user_message:
            return JsonResponse({'error': 'No message provided'}, status=400)

        # Correct Hugging Face API endpoint
        api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        headers = {
            "Authorization": f"Bearer {os.getenv('HF_API_KEY')}",  # Ensure this is set
            "Content-Type": "application/json"
        }
        payload = {"inputs": user_message}

        try:
            response = requests.post(api_url, headers=headers, json=payload)
            print(f"Response Status Code: {response.status_code}")
            print(f"Response Text: {response.text}")  # Log the response for debugging

            if response.status_code == 200:
                data = response.json()
                bot_response = data[0].get('summary_text', 'Sorry, I could not process your request.')
                return JsonResponse({'response': bot_response})
            else:
                return JsonResponse({'error': f"Error from Hugging Face API: {response.text}"}, status=response.status_code)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=405)