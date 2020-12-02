# cd to directory where the Frames folder is (cannot be a shortcut ), 
# and create a folder called transferredFrames in the same directory Frames is in

%cd /content/drive/My\ Drive/Colab\ Notebooks/Frames

import tensorflow as tf
import regex as re
from pathlib import Path

tf.compat.v1.disable_eager_execution()

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import warnings
import time



random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

STYLE_IMG_PATH = "../style.jpg"     #TODO: Add this.
TRANSFER_ROUNDS = 20 # 100

TOTAL_WEIGHT = 1e-6 #1.0
STYLE_WEIGHT = 1e-6 #1.0
CONTENT_WEIGHT = 2.5e-8 #0.3


CONTENT_IMG_H = 256 #500
CONTENT_IMG_W = 256 #500

pathlist = Path('').rglob('*.jpg')
OUTPUT_DIR = '../transferredFrames'


#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(x):
    x = x.reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x



def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


#========================<Loss Function Builder Functions>======================

def styleLoss(style, combination):
    # return None   #TODO: implement.
    S = gramMatrix(style)
    C = gramMatrix(combination)
    channels = 3
    size = CONTENT_IMG_H * CONTENT_IMG_W
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


def contentLoss(base, combination):
    return K.sum(K.square(combination - base))


def totalLoss(x):
    # return None   #TODO: implement.
    a = K.square(x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1, :] - x[:, 1:, :CONTENT_IMG_W - 1, :])
    b = K.square(x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1, :] - x[:, :CONTENT_IMG_H - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

#=========================<Pipeline Functions>==================================


def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resize(img, (ih, iw, 3))
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

# using this instead 
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(CONTENT_IMG_H, CONTENT_IMG_W))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def style_transfer():
    contentTensor = K.variable(preprocess_image(currentPath)) 
    styleTensor = K.variable(preprocess_image(STYLE_IMG_PATH))
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    # model = None   #TODO: implement.


    model = vgg19.VGG19(input_tensor=inputTensor,
                        weights='imagenet', include_top=False)
    

    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")


    loss = K.variable(0.0)
    styleLayerNames = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    contentLayerName = outputDict['block5_conv2']
    print("   Calculating content loss.")
    contentOutput = contentLayerName[0, :, :, :]
    genOutput = contentLayerName[2, :, :, :]
    # loss += None   #TODO: implement.
    loss = loss + CONTENT_WEIGHT * contentLoss(contentOutput, genOutput)

    for layerName in styleLayerNames:
        layer_output = outputDict[layerName]
        style_reference_output = layer_output[1, :, :, :]
        gen_style_output = layer_output[2, :, :, :]
        sl = styleLoss(style_reference_output, gen_style_output)
        loss = loss + (STYLE_WEIGHT / len(styleLayerNames)) * sl
    # loss += None   #TODO: implement.
    loss = loss + TOTAL_WEIGHT * totalLoss(genTensor)

    # TODO: Setup gradients or use K.gradients().
    grads = K.gradients(loss, genTensor)

    outputs = [loss]
    outputs += grads

    f_outputs = K.function([genTensor], outputs)
    def eval_loss_and_grads(x):
        x = x.reshape((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
        outs = f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values

    # fmin_l_bfgs_b needs function for loss and gradients
    # can't call and save value from compute_loss_and_grads
    # calling compute_loss_and_grads twice is inefficient
    # create class for this
    class LossAndGrads(object):
        def loss(self, x):
            lossVal, gradVal = eval_loss_and_grads(x)
            self.lossVal = lossVal
            self.gradVal = gradVal
            return self.lossVal

        def grads(self, x):
            return self.gradVal


    lossAndGrads = LossAndGrads()

    x = preprocess_image(currentPath)




    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        #TODO: perform gradient descent using fmin_l_bfgs_b.
        start_time = time.time()
        x, tLoss, info = fmin_l_bfgs_b(lossAndGrads.loss, x.flatten(), fprime=lossAndGrads.grads, maxfun=20, maxiter=20)
        # print("      Loss: %f." % tLoss)
        # save current generated image
        img = deprocessImage(x.copy())
        # img = deprocessImage(x)

        if i == 19:
          saveFile = OUTPUT_DIR + '/' + currentPath  #+ 'round_%d.png' % i
          save_img(saveFile, img)
          print("      Image saved to \"%s\"." % saveFile)

        end_time = time.time()
        
        # print('Step %d completed in %ds' % (i, end_time - start_time))
    print("   Transfer complete.")
#=========================<Main>================================================


# code for smoothing out the style image

# Reference: https://github.com/znxlwm/pytorch-CartoonGAN
def smoothout(dataset_name, img_size):
    check_folder('./dataset/{}/{}'.format(dataset_name, 'trainB_smooth'))
    file_list = glob('./dataset/{}/{}/*.*'.format(dataset_name, 'trainB'))
    save_dir = './dataset/{}/trainB_smooth'.format(dataset_name)

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)

    for f in tqdm(file_list):
        file_name = os.path.basename(f)

        bgr_img = cv2.imread(f)
        gray_img = cv2.imread(f, 0)

        bgr_img = cv2.resize(bgr_img, (img_size, img_size))
        pad_img = np.pad(bgr_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
        gray_img = cv2.resize(gray_img, (img_size, img_size))

        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, kernel)

        gauss_img = np.copy(bgr_img)
        idx = np.where(dilation != 0)
        for i in range(np.sum(dilation != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

        cv2.imwrite(os.path.join(save_dir, file_name), gauss_img)




def main():
    K.clear_session()
    print("Starting style transfer program.")

    # for testing 1 photo
    # global currentPath
    # currentPath = 'frame629.jpg'
    # style_transfer()
    
    smoothout(STYLE_IMG_PATH, 256)

    global currentPath
    for path in pathlist:
      filename = str(path)
      regex = re.compile(r'\d+')
      num = int(regex.findall(filename)[0])
      if num >= 446 and num <= 520: # can modify this to get number of images to transfer
        currentPath = filename
        print(filename)
        style_transfer() # comment this to check all images to be processed 


    print("Done. Goodbye.")



if __name__ == "__main__":
    main()
