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
    for file in glob.glob('Actor_*/*.wav'):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, train_size= 0.75,random_state=9)
import time
x_train,x_test,y_train,y_test=load_data(test_size=0.25)
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
# Train the model
model.fit(x_train,y_train)
# Predict for the test set
y_pred=model.predict(x_test)














# import librosa
# import os,glob,pickle
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score
# import soundfile

# def extract_feature(file_name,mfcc,chroma,mel):
#   with soundfile.SoundFile(file_name) as sound_file:
#       X=sound_file.read(dtype="float32")
#       sample_rate=sound_file.samplerate
#       if chroma:
#         stfts=np.abs(librosa.stft(X))
#       result=np.array([])
#       if mfcc:
#         mfccs=np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
#       result=np.hstack((result,mfccs))
#       if chroma:
#         chromas=np.mean(librosa.feature.chroma_stft(S=stfts,sr=sample_rate).T,axis=0)
#       result=np.hstack((result,chromas))
#       if mel:
#         mels=np.mean(librosa.feature.melspectrogram(X,sr=sample_rate).T,axis=0)
#       result=np.hstack((result,mels))
#   return result

# emotions={
#     '01':'neutral',
#     '02':'calm',
#     '03':'happy',
#     '04':'sad',
#     '05':'angry',
#     '06':'fearful',
#     '07':'disgust',
#     '08':'surprised',
# }
# observerd_emotions=['calm','happy','fearful','disgust']
# def load_data(test_size=0.2):
#   x,y=[],[]
#   for file in glob.glob("Actor_*\*.wav"):
#     file_name=os.path.basename(file)
#     emotion=emotions[file_name.split("-")[2]]
#     if emotion not in observerd_emotions:
#       continue
#     feature=extract_feature(file,mfcc=True,chroma=True,mel=True)
#     x.append(feature)
#     y.append(emotion)
#   return train_test_split(np.array(x),y,test_size=test_size,random_state=9)

# x_train,x_test,y_train,y_test=load_data(test_size=0.25)
# print((x_train.shape[0],x_test.shape[0]))
# print(f'Features extracted:{x_train.shape[1]}')
# model=MLPClassifier(alpha=0.01,batch_size=256,epsilon=1e-08,hidden_layer_size=(300,),learning_rate='adaptive',max_iter=500)
# model.fit(x_train,y_train)
# y_pred=model.predict(x_test)
# accuracy=accuracy_score(y_true=y_test,y_pred=y_pred)
# print("accuracy:")
# print(y_pred)