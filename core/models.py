from django.db import models

# Create your models here.

class Document(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("approved", "Approved"),
        ("flagged", "Flagged"),
    ]

    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='docs/')
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    reason = models.TextField(null=True, blank=True)

class Chunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    content = models.TextField()
    embedding_id = models.CharField(max_length=255)  # ID from vector DB


class AgentRun(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)

    state = models.JSONField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"AgentRun {self.id} - Doc {self.document.id}"