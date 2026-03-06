from django.db import models


class UploadedPDF(models.Model):
    """Tracks every PDF ingested into the system."""
    name = models.CharField(max_length=255, help_text="Original filename")
    file_name = models.CharField(max_length=255, unique=True, help_text="Stored filename in media/")
    file_size = models.BigIntegerField(null=True, blank=True, help_text="File size in bytes")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return self.name


class ChatMessageStored(models.Model):
    """Stores chat messages for context and history."""
    role = models.CharField(max_length=10, help_text="'human' or 'ai'")
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."
