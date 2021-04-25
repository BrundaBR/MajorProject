from django.urls import path,include
from MlModel import views as v
urlpatterns = [
    path('',v.homepage,name="homepage"),
  
]
