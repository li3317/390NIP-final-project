import os
import numpy as np
from numpy import savetxt
from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import librosa
import librosa.display
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import random
# from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
import warnings

random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PROJECT_PATH = "drive/My Drive/NIP Final Project Audio/"
CONTENT_FILE_NAME = "AliceInWonderLandShort.wav"             #Reference: https://www.youtube.com/watch?v=uUcJSTpMavA
CONTENT_AUD_PATH  = PROJECT_PATH+CONTENT_FILE_NAME           #DONE: Add Content Path. 
STYLE_FILE_NAME   = "TheLittlePrinceShort.wav"               #Reference: https://www.youtube.com/watch?v=yWQo_AAHDUA
STYLE_AUD_PATH    = PROJECT_PATH+STYLE_FILE_NAME             #DONE: Add Style Path. 

CONTENT_CSV_EXIST = False
STYLE_CSV_EXIST = False

NNFT = 512        #Default Value Currently
WIN_LENGTH = NNFT #Default Value Currently
HOP_LENGTH = WIN_LENGTH // 4
N_FILTERS = 4096

CONTENT_WEIGHT = 1e-6    # Alpha weight.
STYLE_WEIGHT = 3.5e-5      # Beta weight.

tf.compat.v1.disable_eager_execution()

#=============================<Helper Fuctions>=================================

def gramMatrix(x):
    # features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = np.matmul(x.T, x) / N_SAMPLES
    return gram

def displayAudioSpectrum(frequencyMagnitude, samplingRate, fileName):
    fig, ax = plt.subplots()
    librosa.display.specshow(librosa.power_to_db(frequencyMagnitude, ref=np.max), sr=samplingRate, hop_length = HOP_LENGTH, y_axis='mel', x_axis='time', cmap = cm.jet)
    ax.set(title = "Content: "+fileName)
    fig.savefig("drive/My Drive/NIP Final Project Audio/"+fileName+".jpg")
    print("Spectrum of "+fileName+" saved")

#========================<Loss Function Builder Functions>======================

def styleLoss(style_gram, g_gram, N_CHANNELS, N_SAMPLES):
    return K.sum(K.square(tf.nn.l2_loss(g_gram - style_gram)) / (4. * (N_FILTERS ** 2) * (N_CHANNELS * N_SAMPLES) ** 2))   #DONE: implement styleLoss, change numFilter to correct variable.
    # Reference: Slide 8

def contentLoss(content, gen):
    return K.sum(K.square(tf.nn.l2_loss(gen - content)))


# def totalLoss(x): # designed to keep the generated image locally coherent. Reference: https://keras.io/examples/generative/neural_style_transfer/
#     a = K.square(x[:, : CONTENT_IMG_H - 1, : CONTENT_IMG_W - 1, :] - x[:, 1 : , : CONTENT_IMG_W - 1, :])
#     b = K.square(x[:, : CONTENT_IMG_H - 1, : CONTENT_IMG_W - 1, :] - x[:, : CONTENT_IMG_W - 1, 1 : , :])
#     return K.sum(K.pow(a+b,1.25))   #DONE: implement total varient loss.




#=========================<Pipeline Functions>==================================



def getRawData():
    global CONTENT_CSV_EXIST
    global STYLE_CSV_EXIST
    print("   Loading Audios.")
    print("      Content audio URL:  \"%s\"." % CONTENT_AUD_PATH)
    print("      Style audio URL:    \"%s\"." % STYLE_AUD_PATH)
    # librosa libray loads amplitude and sampling rate (HZ) of the audio file (amplitude vs time)
    try:
        cFrequencyMagnitude = np.load(CONTENT_AUD_PATH+'FrequencyMagnitude.npy')
        cSrFile = open(CONTENT_AUD_PATH+'SamplingRate.txt', 'r')
        cSamplingRate = int(cSrFile.readline())
        cSrFile.close()
        CONTENT_CSV_EXIST = True
        print("Content Raw Data Exist")
    except IOError:
        cAmplitude, cSamplingRate = librosa.load(CONTENT_AUD_PATH) 

    try:
        sFrequencyMagnitude = np.load(STYLE_AUD_PATH+'FrequencyMagnitude.npy')
        sSrFile = open(STYLE_AUD_PATH+'SamplingRate.txt', 'r')
        sSamplingRate = int(sSrFile.readline())
        sSrFile.close()
        STYLE_CSV_EXIST = True
        print("Style Raw Data Exist")
    except IOError:
        sAmplitude, sSamplingRate = librosa.load(STYLE_AUD_PATH)
        print("      Audios have been loaded.")

    """ Reference: http://man.hubwiz.com/docset/LibROSA.docset/Contents/Resources/Documents/generated/librosa.core.stft.html
    frequency conversion from (amplitude vs time) to (power vs frequency)

    function stft returns a complex-valued matrix D such that:
        np.abs(D[f, t]) is the magnitude of frequency bin f at frame t
        np.angle(D[f, t]) is the phase of frequency bin f at frame t

    Parameters: 
        n_fft:
            recommended value is 512, 
            replaces all the period larger than win_length to zero paddings to improve frequency resolution by TTF 
        hop_length:
            defaultly equal to win_length // 4,
            actual length without overlaping in the window
        win_length:
            defaultly equal to n_fft,
            the length of window each ttf will be done
    """
    if not (CONTENT_CSV_EXIST):
        cSTFT = librosa.stft(cAmplitude, NNFT, HOP_LENGTH, WIN_LENGTH)
        print("      STFT Conversion for Cotent Completed.")
        cFrequencyMagnitude = np.abs(cSTFT)
        np.save(CONTENT_AUD_PATH+'FrequencyMagnitude.npy', cFrequencyMagnitude)
        cSrFile = open(CONTENT_AUD_PATH+'SamplingRate.txt','w')  # w : writing mode  /  r : reading mode  /  a  :  appending mode 
        cSrFile.write('{}'.format(cSamplingRate))   
        cSrFile.close() #Reference: https://stackoverflow.com/questions/36900443/how-to-format-the-file-output-python-3
        print("Raw Data files for Cotent saved")
    if not (STYLE_CSV_EXIST): 
        sSTFT = librosa.stft(sAmplitude, NNFT, HOP_LENGTH, WIN_LENGTH)
        print("      STFT Conversion for Style Completed.")
        sFrequencyMagnitude = np.abs(sSTFT)
        np.save(STYLE_AUD_PATH+'FrequencyMagnitude.npy',sFrequencyMagnitude)
        sSrFile = open(STYLE_AUD_PATH+'SamplingRate.txt','w')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
        sSrFile.write('{}'.format(sSamplingRate))
        sSrFile.close()
        print("Raw Data files for Style saved")

    """ reference: https://librosa.org/doc/main/auto_examples/plot_display.html
        Display audio spectrum
    """
    if not (CONTENT_CSV_EXIST):
        displayAudioSpectrum(cFrequencyMagnitude, cSamplingRate, CONTENT_FILE_NAME)
    if not (STYLE_CSV_EXIST): 
        displayAudioSpectrum(sFrequencyMagnitude, sSamplingRate, STYLE_FILE_NAME)

    """
        check if data is loaded correctly
    """
    # print("Content Sampling Rate: " + str(cSamplingRate))
    # print("Content Frequncy Magnitude: ")
    # print(cFrequencyMagnitude)
    # print("Style Sampling Rate: " + str(sSamplingRate))
    # print("Style Frequncy Magnitude: ")
    # print(sFrequencyMagnitude)

    return ((cFrequencyMagnitude, cSamplingRate), (sFrequencyMagnitude, sSamplingRate))




def preprocessData(raw):
    frequencyMagnitude, samplingRate = raw
    return np.log1p(frequencyMagnitude) # use log recieve numerical stability
    


def styleTransfer(cData, sData, cSamplingRate, sSamplingRate):
    cTensor = K.variable(cData)
    sTensor = K.variable(sData)
    gTensor = K.placeholder(shape=cTensor.shape)
    N_CHANNELS = cTensor.shape[0]
    N_SAMPLES  = cTensor.shape[1]
    sTensor = sTensor[:N_CHANNELS, :N_SAMPLES] # cut the size of style to fit the size of content
    print("Shape of Content Tensor: %s." % str(cTensor.shape))
    print("Shape of Style Tensor: %s." % str(sTensor.shape))
    print("Shape of Generating Tensor: %s." % str(gTensor.shape))
    print("N_CHANNELS: "+str(N_CHANNELS))
    print("N_SAMPLES: "+str(N_SAMPLES))
    
    """ 
    Reference: https://www.tensorflow.org/guide/effective_tf2
    tf.nn network creation:
        all previous tf.nn layer can be inserted into next layer as input

    Reference: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    tf.nn.conv2d layer:
        Parameters: 
            input: A Tensor of type (half, bfloat16, float32, float64)
            filters: A Tensor of same type as input, shape (filter_height, filter_width, in_channels, out_channels)
            strides: list of ints, the stride of the sliding window for each dimension of input
            padding: "SAME" or "VALID"
        Example:
            x_in = np.array([[
                [[2], [1], [2], [0], [1]],
                [[1], [3], [2], [2], [3]],
                [[1], [1], [3], [3], [0]],
                [[2], [2], [0], [1], [1]],
                [[0], [0], [3], [1], [2]], ]])
            kernel_in = np.array([
                [ [[2, 0.1]], [[3, 0.2]] ],
                [ [[0, 0.3]],[[1, 0.4]] ], ])
            x = tf.constant(x_in, dtype=tf.float32)
            kernel = tf.constant(kernel_in, dtype=tf.float32)
            tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
    """

    kernel_in = np.random.randn(1, 11, N_CHANNELS, N_FILTERS)
    kernel = tf.constant(kernel_in, dtype='float32')
    sess = tf.compat.v1.Session()
    with sess.as_default():
      cNet = tf.nn.conv2d(cData.T[None,None,:,:].astype(np.float32), kernel, strides=[1, 1, 1, 1], padding="VALID")
      cNet = tf.nn.relu(cNet)
      cEval = cNet.eval()

      sNet = tf.nn.conv2d(sData.T[None,None,:,:].astype(np.float32), kernel, strides=[1, 1, 1, 1], padding="VALID")
      sNet = tf.nn.relu(sNet)
      sEval = sNet.eval()

      gNet = tf.nn.conv2d(np.random.randn(1,1,N_SAMPLES,N_CHANNELS).astype(np.float32), kernel, strides=[1, 1, 1, 1], padding="VALID")
      gNet = tf.nn.relu(gNet)

      style_features = np.reshape(sEval, (-1, N_FILTERS))
      g_features = np.reshape(gNet.eval(), (-1, N_FILTERS))
      
      opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
      x = tf.Variable(np.zeros((1,1, N_SAMPLES, N_CHANNELS)))

      print("gram calculation complete")
      content_loss = CONTENT_WEIGHT * contentLoss(cEval,gNet.eval())
      print("content loss calculation complete")
      style_loss = STYLE_WEIGHT * styleLoss(gramMatrix(style_features),gramMatrix(g_features),N_CHANNELS,N_SAMPLES)
      print("style loss calculation complete")
      total_loss =  tf.cast(style_loss,content_loss.dtype) + content_loss # Reference: https://stackoverflow.com/questions/35725513/tensorflow-cast-a-float64-tensor-to-float32
      print("loss calculation complete")
      print(total_loss.eval())
           



      opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    
      sess.run(tf.compat.v1.initialize_all_variables())
      print('Started optimization.')
      opt.minimize(sess, x)
      print('Final loss:')
      print(total_loss.eval())
      result = x.eval()


#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()                                  #TODO: Get raw Data of two audio
    cData = preprocessData(raw[0])   # Content image.   #TODO: Preprocess Data of Content Audio
    sData = preprocessData(raw[1])   # Style image.     #TODO: Preprocess Data of Style Audio 
    styleTransfer(cData, sData, raw[0][1], raw[1][1])
    print("Done. Goodbye.")
    


if __name__ == "__main__":
    main()
