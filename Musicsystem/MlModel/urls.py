from django.urls import path,include
from MlModel import views as v

urlpatterns = [
    path('',v.homepage,name="homepage"),
    path('getaudio/',v.get_audio_input,name="getaudio"),
    # path('personalized-playlist',v.get_playlist_display,name="personalized-playlist"),
]
