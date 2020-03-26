import librosa
import scipy.io.wavfile as wav
import librosa.display
import matplotlib.pyplot as plt
import pyaudio
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras import backend as K
# from pydub import AudioSegment
from pydub.playback import play
# import glob
import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D, BatchNormalization,LSTM
# import sounddevice as sd
from keras.models import Sequential,load_model
import librosa.util
import numpy as np
# import random
import soundfile as sf
from ann_visualizer.visualize import ann_viz
from sklearn.model_selection import train_test_split
np.random.seed(2)

output_res = ['Alarm','Noise']
RATE = 22050
RECORD_SECONDS = 3.97
CHUNKSIZE = 1024
my_model = load_model('cnn_model.h5')
p = pyaudio.PyAudio()

#Loop
timeesz = 0
while True:
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

    frames = [] # A python-list of chunks(numpy.ndarray)

    for _ in range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS)):
        data = stream.read(CHUNKSIZE)
        frames.append(np.fromstring(data, dtype=np.float32))

    #Convert the list of numpy-arrays into a 1D array (column-wise)
    numpydata = np.hstack(frames)

    # close stream
    stream.stop_stream()
    wav.write('p'+str(1000)+'.wav',RATE,numpydata)
    
#     t = time.time()
    y,sr = librosa.load('p'+str(1000)+'.wav', duration=2.97)
    ps = librosa.feature.melspectrogram(y=y, sr=sr)
    ps = ps.reshape(128,128,1)
    predictions = my_model.predict(np.array([ps,]))
#     print(time.time()-t,'secs')
    print("Predictions:",predictions)
#     print(output_res[np.argmax(predictions)])
    confidence = predictions[0][0]*100
    if(confidence>=85):
#         print('Beep Sounds')
        timeesz +=1
        onset_env = librosa.onset.onset_strength(y=y, sr=22050,
                                                         hop_length=512,
                                                         aggregate=np.median)

        peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 35)
#         print(peaks,end="")
        times = librosa.frames_to_time(np.arange(len(onset_env)),
                                   sr=sr, hop_length=512)
    #     #     wait = 100
        if(len(peaks)==0 or len(peaks)==1):
            print('\nAbnormal as number of peaks detected is ',format(len(peaks)))
        else:
            val = times[peaks]
#             print(val)
            if(val[1]-val[0]<=1 or len(peaks)>=4):
#                 engine.say('Alarm Sound Detected')
#                 engine.runAndWait()
                print('Alarm Sound')
                time.sleep(0.6)
            else:
                print('Normal Beeps')
        #     plt.figure()
        #     ax = plt.subplot(2, 1, 2)
        #     D = librosa.stft(numpydata)
        #     librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
        #                              y_axis='log', x_axis='time')
        #     plt.subplot(2, 1, 1, sharex=ax)
        #     plt.plot(times, onset_env, alpha=0.8, label='Onset strength')
        #     plt.vlines(times[peaks], 0,
        #                onset_env.max(), color='r', alpha=0.8,
        #                label='Selected peaks')
        #     plt.legend(frameon=True, framealpha=0.8)
        #     plt.axis('tight')
        #     plt.tight_layout()
        #         break
    else:
        print('Noise')
    if(timeesz==10):
        break
        

stream.close()
p.terminate()
