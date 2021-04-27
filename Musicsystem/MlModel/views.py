from django.shortcuts import render
import spotipy
import requests
import datetime
import base64
from urllib.parse import urlencode
# Create your views here.
   
def model():
    import librosa
    import soundfile
    import os, glob, pickle
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    from datetime import datetime
    print("start----",datetime.now())
    def extract_feature(file_name, mfcc, chroma, mel):
        X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
        return result
    # Emotions in the RAVDESS & TESS dataset
    emotions={
    '01':'neutral',
    '02':'calm',
    '03':'happy',
    '04':'sad',
    '05':'angry',
    '06':'fearful',
    '07':'disgust',
    '08':'surprised'
    }
    # Emotions to observe
    observed_emotions=['neutral','calm','happy','sad','angry','fearful', 'disgust','surprised']
    def load_data(test_size=0.2):
        x,y=[],[]
        for file in glob.glob('Audio_Speech/Actor_*/*.wav'):
            file_name=os.path.basename(file)
            emotion=emotions[file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
        return train_test_split(np.array(x), y, test_size=test_size, train_size= 0.75,random_state=9)

    file="output.wav"
    feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
    x_train,x_test,y_train,y_test=load_data(test_size=0.25)
    model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
    # Train the model
    model.fit(x_train,y_train)
    # Predict for the test set
    y_pred=model.predict(x_test)
    y_pre=model.predict([feature])
    accuracy=accuracy_score(y_true=y_test,y_pred=y_pred)
    print(accuracy)
    print(y_pre)
    emotion="".join(y_pre)
    track=render_playlist(emotion) # spotify call 
    print("end---",datetime.now())
    return track


def homepage(request):

    return render(request,'index.html')


def get_audio_input(request):
    import pyaudio
    import wave
    from array import array
    from struct import pack
    def record(outputFile):
        CHUNK=1024
        FORMAT=pyaudio.paInt16
        CHANNELS=2
        RECORD_SECONDS=5
        RATE=44100
        p=pyaudio.PyAudio()
        print("Listening..............")
        stream=p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK)
        frames=[]
        for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
            data=stream.read(CHUNK)
            frames.append(data)
        stream.stop_stream()
        print("Record ----- stop")
        stream.close()
        p.terminate()
        wf=wave.open(outputFile,'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    record('output.wav')
    track_return=model()
    return render(request,'recorder.html',{"tracks":track_return})


def render_playlist(emotion):
    if emotion=='happy' or emotion=='surprised' or emotion=='neutral':
        print("HAPPY EMOTION DETECTED!")
        print("GETTING PLAYLIST>>>>>>>>>>>>>>>>")
        client_id='2358ddbb408c47cdadf5da7bbfacf714'
        client_secret='567cfa73a1f14913b72e5ae1eece26ae'
        client_creds=f"{client_id}:{client_secret}"
        client_creds_b64=base64.b64encode(client_creds.encode())
        token_url='https://accounts.spotify.com/api/token'
        method="POST"
        token_data={
            "grant_type":"client_credentials"
        }
        token_headers={
            "Authorization":f"Basic {client_creds_b64.decode()}"
        }
        r=requests.post(token_url,data=token_data,headers=token_headers)
        token_response_data=r.json()
        valid_request=r.status_code in range(200,299)
        if valid_request:
            now=datetime.datetime.now()
            access_token=token_response_data['access_token']
            expires_in=token_response_data['expires_in']
            expires=now + datetime.timedelta(seconds=expires_in)
            did_expires=expires < now
            print("SPOTIFY TOKEN VALIDATION DONE")
            headers={
        "Authorization":f"Bearer {access_token}"
        }
        # this is for searching in spotify
            # endpoint="https://api.spotify.com/v1/search"
            # data=urlencode({"q":"happy","type":"tracks"})
            # lookup_url=f"{endpoint}?{data}"
            # #https://ofy.com/playlistpen.spoti/37i9dQZF1DXdPec7aLTmlC?si=8d3a9fe373894353
            # r=requests.get(lookup_url,headers=headers)
            # print(r.json())
            #https://open.spotify.com/embed/playlist/37i9dQZF1DXdPec7aLTmlC
            #https://open.spotify.com/genre/happy_mood-page
            id='37i9dQZF1DXdPec7aLTmlC'
            endpoint_p="https://open.spotify.com/embed/playlist/"
            data_p=id
            lookup_playlist=f"{endpoint_p}{data_p}"
            # # print(lookup_playlist)
            # r_p=requests.get(lookup_playlist,headers=headers)
            # # print(r_p.json())

            
            return lookup_playlist
    elif emotion=='angry' or emotion=='disgust' or emotion=='fearful':
        print("EMOTION DETECTED NO SO GOOD")
        id_rage='37i9dQZF1DX3ND264N08pv'
        id='37i9dQZF1DX889U0CL85jj'
        endpoint_p="https://open.spotify.com/embed/playlist/"
        data_p=id
        lookup_playlist=f"{endpoint_p}{data_p}"
        return lookup_playlist


    


