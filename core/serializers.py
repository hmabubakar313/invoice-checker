# core/serializers.py

from django.core.validators import FileExtensionValidator
from rest_framework import serializers
from core.models import Document

class DocumentSerializer(serializers.ModelSerializer):
    file = serializers.FileField(
        validators=[FileExtensionValidator(allowed_extensions=['pdf'])]
    )

    class Meta:
        model = Document
        fields = ['id', 'title', 'file']