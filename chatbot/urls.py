from django.urls import path
from . import views

app_name = "chatbot"

urlpatterns = [
    path("", views.demo_page, name="demo_page"),
    path("api/chat/", views.chat_api, name="chat_api"),
]