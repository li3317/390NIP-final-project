import os
import numpy as np
from numpy import savetxt
from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import librosa
import librosa.display
import matplotlib.cm as cm
import matplotlib.pyplot as plt
# import random
# from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
# from tensorflow.keras.applications import vgg19
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
#=========================<Pipeline Functions>==================================

def displayAudioSpectrum(frequencyMagnitude, samplingRate, fileName):
    fig, ax = plt.subplots()
    librosa.display.specshow(librosa.power_to_db(frequencyMagnitude, ref=np.max), sr=samplingRate, hop_length = HOP_LENGTH, y_axis='mel', x_axis='time', cmap = cm.jet)
    ax.set(title = "Content: "+fileName)
    fig.savefig("drive/My Drive/NIP Final Project Audio/"+fileName+".jpg")
    print("Spectrum of "+fileName+" saved")

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
    print("Shape of Content Tensor: %s." % str(cTensor.shape))
    print("Shape of Style Tensor: %s." % str(sTensor.shape))
    print("Shape of Generating Tensor: %s." % str(gTensor.shape))



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
