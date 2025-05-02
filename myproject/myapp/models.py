from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    crop_type = models.CharField(max_length=200)  # Optional if you just need it for prediction

from django.db import models

from django.db import models

class Feedback(models.Model):
    text = models.TextField()
    email = models.EmailField()
    email_sent = models.BooleanField(default=False)  # Track email status
    created_at = models.DateTimeField(auto_now_add=True)  # Track submission time

    def __str__(self):
        return f"Feedback from {self.email}"