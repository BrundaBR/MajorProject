from django.shortcuts import render

# Create your views here.
   
def model():
    import librosa
    import soundfile
    import os, glob, pickle
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score


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
    print(y_pre)
def homepage(request):
    import pyaudio
    import wave
    from array import array
    from struct import pack
    def record(outputFile):
        CHUNK=1024
        FORMAT=pyaudio.paInt16
        CHANNELS=2
        RECORD_SECONDS=7
        RATE=44100
        p=pyaudio.PyAudio()
        stream=p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK)
        frames=[]
        for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
            data=stream.read(CHUNK)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf=wave.open(outputFile,'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    record('output.wav')
    model()

    return render(request,'index.html')
 