from django.contrib import admin
from .models import UploadedPDF


@admin.register(UploadedPDF)
class UploadedPDFAdmin(admin.ModelAdmin):
    list_display = ('name', 'file_name', 'file_size', 'uploaded_at')
    readonly_fields = ('uploaded_at',)
    ordering = ('-uploaded_at',)
