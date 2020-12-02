import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
import scipy
import imageio
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
import warnings
# tf.compat.v1.disable_eager_execution()

import tensorflow as tf
from tensorflow import keras

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import librosa
import numpy as np

import time
import math
import argparse

import sys
from moviepy.editor import VideoFileClip

random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# coefficient of content and style
style_param = 2
content_param = 1e-3

num_epochs = 20

N_FFT = 512
N_CHANNELS = round(1 + N_FFT/2)
OUT_CHANNELS = 32
N_FILTERS  = 4096
N_SAMPLES = N_FILTERS/2

content_video = "drive/MyDrive/NIP Final Project Audio/Chess club.mp4"
style_video   = "drive/MyDrive/NIP Final Project Audio/Musk.mp3" 

duration = 0
#=============================<Helper Fuctions>=================================

def mp4_to_wav(file_path, duration_limit):
    video = VideoFileClip(file_path)
    audio = video.audio
    if duration_limit != 0:
      audio.duration = duration_limit
    audio.write_audiofile(os.path.splitext(file_path)[0]+".wav")
    return video.duration 

def preprocessing(content, style):
    global duration, N_SAMPLES
    duration = mp4_to_wav(content, duration)
    
    y, sr = librosa.load(content)
    y_shifted = librosa.effects.pitch_shift(y, sr, n_steps=-3)
    librosa.output.write_wav(os.path.splitext(content)[0]+"Pitch.wav", y_shifted, sr)

    a_content, sr = wav2spectrum(os.path.splitext(content)[0]+"Pitch.wav")
    if os.path.splitext(style)[1] == ".mp4":
      mp4_to_wav(style, duration)
      a_style, sr = wav2spectrum(os.path.splitext(style)[0]+".wav")
    else:
      a_style, sr = wav2spectrum(style)
    N_SAMPLES = min(a_content.shape[1],a_style.shape[1])
    a_style = a_style[:N_CHANNELS, :N_SAMPLES]
    return a_content, a_style, N_SAMPLES, sr

def wav2spectrum(file):
    filename = os.path.splitext(file)[0]
    print("filename: "+filename)
    try:
      S = np.load(filename+'.npy')
      srFile = open(filename+'.txt', 'r')
      sr = int(srFile.readline())
      srFile.close()
      STYLE_CSV_EXIST = True
      print(filename+" data exist")
    except IOError:
      x, sr = librosa.load(file)
      S = librosa.stft(x, N_FFT)
      p = np.angle(S)
      S = np.log1p(np.abs(S))
      print(filename+" wav2spectrum calculation")
      np.save(filename+'.npy',S)
      srFile = open(filename+'.txt', 'w')
      srFile.write('{}'.format(sr))
      srFile.close()
      print("wav2spectrum result saved for "+filename)
    return S, sr


def spectrum2wav(spectrum, sr, outfile):
    a = np.exp(spectrum) - 1
    p = 2 * np.pi * np.random.random_sample(spectrum.shape) - np.pi
    for i in range(50):
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, N_FFT))
    librosa.output.write_wav(outfile, x, sr)

def plt_spectrum(content, name):
    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.imsave( name+'.png', content[:400, :])

#========================<Loss Function Builder Functions>======================
def gramMatrix(x):
    m, n_C, n_H, n_W = x.shape
    x_unrolled = tf.reshape(x,(m * n_C * n_H, n_W))
    g = tf.matmul(x_unrolled,tf.transpose(x_unrolled)) / 244
    return g
    
def compute_content_loss(a_C, a_G):
    content_loss = tf.nn.l2_loss(a_C - a_G)
    return content_loss

def compute_layer_style_loss(a_S, a_G):
    style_gram = gramMatrix(a_S)
    gram = gramMatrix(a_G)
    style_loss = 4 * tf.nn.l2_loss(gram - style_gram)
    return style_loss

def total_loss(a_C, a_S, a_G):
    content_loss = content_param * compute_content_loss(a_C, a_G)
    style_loss = style_param * compute_layer_style_loss(a_S, a_G)
    return content_loss + style_loss


#=========================<Pipeline Functions>==================================
class Evaluator(object):
    def __init__(self, kFunction):
        self.kFunction = kFunction
        
    def loss(self, x):
        self.loss_, self.grads_ = self.kFunction([x.reshape((1,1, N_CHANNELS, round(x.shape[0]/N_CHANNELS)))])
        return self.loss_.astype(np.float64)

    def grads(self, x):
        return self.grads_.flatten().astype(np.float64)

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.conv2D = tf.keras.layers.Conv2D(filters=N_SAMPLES, kernel_size=1, strides=(1, 1), padding='VALID', activation='selu')
    self.relu = tf.keras.layers.LeakyReLU(alpha=0.2)

  def call(self, inputs):
    x1 = self.conv2D(inputs)
    x1 = self.relu(x1)
    return x1 + inputs

def CNN():
    model = tf.keras.models.Sequential()
    model.add(MyModel())
    return model

def styleTransfer(a_content, a_style, N_SAMPLES):
    print("   Building transfer model.")
    a_C = tf.convert_to_tensor(a_content)[None, None, :, :]
    a_S = tf.convert_to_tensor(a_style)[None, None, :, :]
    a_G = K.placeholder(a_C.shape)


    model = CNN()
    
    content_features = model(K.variable(a_C))
    style_features = model(K.variable(a_S))
    gen_features = model(a_G)
    
    # Get the loss
    loss = total_loss(content_features, style_features, gen_features)

    # Setup gradients or use K.gradients().
    grads = K.gradients(loss, a_G)
    kFunction = K.function([a_G], [loss] + grads)
    evaluator = Evaluator(kFunction)
    x = np.zeros((1,1, N_CHANNELS, N_SAMPLES))
    print("   Beginning transfer.")
    # Train the Model
    for epoch in range(1, num_epochs + 1):
        if epoch % 1 == 0:
          print(str(epoch)+"/"+str(num_epochs))
        res = scipy.optimize.fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=500)
        x = res[0].reshape((1,1, N_CHANNELS, N_SAMPLES))
    return x
    

#=========================<Main>================================================

def main():
    print("Starting audio style transfer program.")
    a_content, a_style, N_SAMPLES, sr = preprocessing(content_video, style_video)
    x = styleTransfer(a_content, a_style, N_SAMPLES)
    
    OUTPUT_FILENAME='orginal.wav'
    spectrum2wav(x[0,0], sr, OUTPUT_FILENAME)
    print("audio conversion complete")
    plt_spectrum(a_content, 'Content_spectrum')
    plt_spectrum(a_style, 'Style_spectrum')
    a = np.zeros_like(a_content)
    a[:N_CHANNELS,:] = np.exp(x[0,0]) - 1
    plt_spectrum(a, 'Gen_spectrum')

if __name__ == "__main__":
    main()