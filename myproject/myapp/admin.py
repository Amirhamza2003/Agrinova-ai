from django.contrib import admin
from .models import Feedback

@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ('email', 'text', 'email_sent', 'created_at')
    list_filter = ('email_sent', 'created_at')
    search_fields = ('email', 'text')