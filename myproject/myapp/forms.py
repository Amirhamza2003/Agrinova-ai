from django import forms
from .models import UploadedImage

CROP_CHOICES = [
    ('corn', 'Corn'),
    ('potato', 'Potato'),
    ('rice', 'Rice'),
    ('wheat', 'Wheat'),
    ('sugarcane', 'Sugarcane'),
]

class ImageUploadForm(forms.ModelForm):
    crop_type = forms.ChoiceField(
        choices=CROP_CHOICES,
        required=True,
        widget=forms.Select(attrs={
            'class': 'form-select',
            'placeholder': 'Select a crop type'
        })
    )
    image = forms.ImageField(
        required=True,
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'image/*'
        })
    )

    class Meta:
        model = UploadedImage
        fields = ['image', 'crop_type']

    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            if image.size > 5 * 1024 * 1024:  # 5 MB limit
                raise forms.ValidationError("Image size should not exceed 5 MB.")
            if not image.content_type.startswith('image/'):
                raise forms.ValidationError("Uploaded file must be an image.")
        return image