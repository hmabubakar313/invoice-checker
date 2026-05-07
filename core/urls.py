from django.urls import path
from core.views import upload_document, ask_question

urlpatterns = [
    path('upload/', upload_document),
    path('ask/', ask_question),
]