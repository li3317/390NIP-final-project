
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from keras.preprocessing.image import save_img
from scipy.misc import imresize
# from PIL import Image
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings

random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_IMG_PATH = "img/content.jpg"           #DONE: Add Content Path.
STYLE_IMG_PATH = "img/style.jpg"             #DONE: Add Style Path.


CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 1e-6    # Alpha weight.
STYLE_WEIGHT = 3.5e-5      # Beta weight.
TOTAL_WEIGHT = 2.5e-8

TRANSFER_ROUNDS = 120

numFilter = 7

#=============================<Helper Fuctions>=================================
'''
DONE: implement deprocessImage.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(img): # Reference: https://keras.io/examples/generative/neural_style_transfer/
    img = img.reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1] # flip order of 3rd dimension, BGR -> RGB. Reference: https://stackoverflow.com/questions/31633635/what-is-the-meaning-of-inta-1-in-python
    return np.clip(img, 0, 255).astype("uint8")


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram



#========================<Loss Function Builder Functions>======================

def styleLoss(style, gen):
    return K.sum(K.square(gramMatrix(style) - gramMatrix(gen)) / (4. * (numFilter ** 2) * (STYLE_IMG_H * STYLE_IMG_W) ** 2))   #DONE: implement styleLoss, change numFilter to correct variable.
    # Reference: Slide 8

def contentLoss(content, gen):
    return K.sum(K.square(gen - content))


def totalLoss(x): # designed to keep the generated image locally coherent. Reference: https://keras.io/examples/generative/neural_style_transfer/
    a = K.square(x[:, : CONTENT_IMG_H - 1, : CONTENT_IMG_W - 1, :] - x[:, 1 : , : CONTENT_IMG_W - 1, :])
    b = K.square(x[:, : CONTENT_IMG_H - 1, : CONTENT_IMG_W - 1, :] - x[:, : CONTENT_IMG_W - 1, 1 : , :])
    return K.sum(K.pow(a+b,1.25))   #DONE: implement total varient loss.





#=========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))



def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = imresize(img, (ih, iw, 3))
        
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


'''
DONE: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    model = vgg19.VGG19(include_top=False, weights="imagenet", input_tensor=inputTensor)   #DONE: import vgg model. Reference: Slide 8
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")
    loss = 0.0
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"
    
    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    contentGenOutput = contentLayer[2, :, :, :]
    loss += CONTENT_WEIGHT * contentLoss(contentOutput,contentGenOutput)   #DONE: implement content_loss * alpha part.

    print("   Calculating style loss.")
    for layerName in styleLayerNames:
        styleLayer = outputDict[layerName] # Reference: For code above for content loss
        styleOutput = styleLayer[0, :, :, :]
        styleGenOutput = styleLayer[2, :, :, :]
        loss += (STYLE_WEIGHT / len(styleLayerNames)) * styleLoss(styleOutput,styleGenOutput)   #DONE: implement + style_loss * beta part. Reference: https://keras.io/examples/generative/neural_style_transfer/
    
    loss += TOTAL_WEIGHT * totalLoss(genTensor)    #DONE: implement total loss. Reference: https://keras.io/examples/generative/neural_style_transfer/
    
    # DONE: Setup gradients or use K.gradients().
    
    grads = K.gradients(loss, genTensor)    # Reference: https://stackoverflow.com/questions/49834380/k-gradientsloss-input-img0-return-none-keras-cnn-visualization-with-ten
    outputs = [loss]
    outputs += grads

    #=========================<Evaluator>===========================================
    # Reference: https://notebook.community/aidiary/notebooks/keras/170818-neural-style-transfer-examples
    def eval_loss_and_grads(x):
        x = x.reshape((1,CONTENT_IMG_H,CONTENT_IMG_W,3))
        outs = K.function([genTensor], outputs)([x])
        loss = outs[0]
        grad = outs[1].flatten().astype('float64')
        return loss, grad 

    class Evaluator:
        
        def loss(self, x):
            loss, grad = eval_loss_and_grads(x)
            self._grad = grad
            return loss
        
        def gradients(self, x):
            return self._grad

    evaluator = Evaluator()
    x = np.random.uniform(0, 255, (1, CONTENT_IMG_H, CONTENT_IMG_W, 3)) - 128.0 # Reference: https://neurowhai.tistory.com/169 
    print("   Beginning transfer.")
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        #DONE: perform gradient descent using fmin_l_bfgs_b. Reference: https://notebook.community/aidiary/notebooks/keras/170818-neural-style-transfer-examples
        x, tLoss, info_dic =  fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.gradients, maxfun=20)
        print("      Loss: %f." % tLoss)
    img = deprocessImage(x)
    saveFile = "img/output_final.jpg"   #DONE: set saveFile path.
    save_img(saveFile, img)   #Uncomment when everything is working right.
    print("      Image saved to \"%s\"." % saveFile)
    print("   Transfer complete.")



#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()
