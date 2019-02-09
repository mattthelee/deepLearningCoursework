import time
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import tensorflow as tf
import scipy
import pdb
import skimage

def exercise1Preprocess(imgpath, scale = 3):
    image = scipy.misc.imread(imgpath, True, 'RGB')
    (height,width) = image.shape
    print(f"\nSize of image: {image.shape}")

    label_ = modcrop(image, scale)
    newHeight = int(height/scale)
    newWidth = int(width/scale)
    newImage = tf.image.resize_bicubic(label_,[newHeight,newWidth])
    newImage = tf.image.resize_bicubic(newImage,[height,width])
    return newImage, label_


def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def preprocess(path, scale=3):
  """
  Preprocess single image file
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation
  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
  image = imread(path, is_grayscale=True)
  label_ = modcrop(image, scale)

  # Must be normalized
  image = image / 255.
  label_ = label_ / 255.

  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

  return input_, label_

"""Set the image hyper parameters
"""
c_dim = 1
input_size = 255

"""Define the model weights and biases
"""

# define the placeholders for inputs and outputs
inputs = tf.placeholder(tf.float32, [None, input_size, input_size, c_dim], name='inputs')

## ------ Add your code here: set the weight of three conv layers
# replace '0' with your hyper parameter numbers
# conv1 layer with biases: 64 filters with size 9 x 9
# conv2 layer with biases and relu: 32 filters with size 1 x 1
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5



def convWithRelu(input, biases, weights):
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID')
    conv = tf.nn.bias_add(conv, biases)
    conv = tf.nn.relu(conv)
    return conv

def convOnly(input, biases, weights):
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID')
    conv = tf.nn.bias_add(conv, biases)
    return conv


weights = {
    'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
    'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
    'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
    }

biases = {
      'b1': tf.Variable(tf.zeros([64]), name='b1'),
      'b2': tf.Variable(tf.zeros([32]), name='b2'),
      'b3': tf.Variable(tf.zeros([1]), name='b3')
    }

"""Define the model layers with three convolutional layers
"""
## ------ Add your code here: to compute feature maps of input low-resolution images
# replace 'None' with your layers: use the tf.nn.conv2d() and tf.nn.relu()
# conv1 layer with biases and relu : 64 filters with size 9 x 9

conv1 = convWithRelu(inputs,biases['b1'], weights['w1'])
##------ Add your code here: to compute non-linear mapping
# conv2 layer with biases and relu: 32 filters with size 1 x 1

conv2 = convWithRelu(conv1,biases['b2'], weights['w2'])
##------ Add your code here: compute the reconstruction of high-resolution image
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
conv3 = convOnly(conv2,biases['b3'], weights['w3'])


"""Load the pre-trained model file
"""
model_path='./model/model.npy'
model = np.load(model_path, encoding='latin1').item()
##------ Add your code here: show the weights of model and try to visualisa
# variabiles (w1, w2, w3)

# For each of the layers get its weights
for key in ['w1','w2','w3']:
    weight = model[key]
    # Bizarrely w3 weights are in a different shape
    if key == 'w3':
        (height,width,filters,_) = weight.shape
    else:
        (height,width,_,filters) = weight.shape

    # The 2nd layers are much more clearly visualised if normalised
    if key == 'w2':
        normalisation = True
    else:
        normalisation = False

    # Set up subplot grid
    columnMax = 12
    rowMax = int(filters / columnMax) + 1
    fig, subplots = plt.subplots(rowMax, columnMax,)

    # Remove axes for all subplots even those we have no weights for
    for subRow in subplots:
        for sub in subRow:
            sub.axis('off')

    if normalisation:
        # Get min and max values for normalisation of each layer
        vmin = 10000000
        vmax = 0
        for i in range(filters):
            if key == 'w3':
                filter = weight[:,:,i,0]
            else:
                filter = weight[:,:,0,i]
            filMax = np.amax(filter)
            filMin = np.amin(filter)
            if filMax > vmax:
                vmax = filMax
            if filMin < vmin:
                vmin = filMin

    # For each filter add a subplot
    for filterNo in range(filters):
        if key == 'w3':
            filter = weight[:,:,filterNo,0]
        else:
            filter = weight[:,:,0,filterNo]
        # Find subplot indices
        column = filterNo % columnMax
        row = int(filterNo / columnMax)
        print(f"Weight: {key}, filter: {filterNo}")
        print(filter)
        if normalisation:
            subplots[row ,column].imshow(filter, shape=filter.shape, cmap=cm.Greys_r, vmin=vmin, vmax=vmax, interpolation='none')
        else:
            subplots[row ,column].imshow(filter, shape=filter.shape, cmap=cm.Greys_r,interpolation='none')
        subplots[row ,column].set_title(filterNo)
    plt.show()



"""Initialize the model variabiles (w1, w2, w3, b1, b2, b3) with the pre-trained model file
"""
# launch a session
sess = tf.Session()

for key in weights.keys():
  sess.run(weights[key].assign(model[key]))

for key in biases.keys():
  sess.run(biases[key].assign(model[key]))



"""Read the test image
"""
blurred_image, groundtruth_image = preprocess('./image/butterfly_GT.bmp')

"""Run the model and get the SR image
"""
# transform the input to 4-D tensor
input_ = np.expand_dims(np.expand_dims(blurred_image, axis =0), axis=-1)

# run the session
# here you can also run to get feature map like 'conv1' and 'conv2'
output_ = sess.run(conv3, feed_dict={inputs: input_})

# Crop the groundtruth and blurred images to be same size as the output
(_, output_height, output_width, _) = output_.shape
(groundtruth_height, groundtruth_width) = groundtruth_image.shape

# Find the amount we need to crop
heightTrim = int((groundtruth_height - output_height)/2)
widthTrim = int((groundtruth_width - output_width)/2)

# Perform the crop
groundtruth_image = groundtruth_image[heightTrim:-heightTrim,widthTrim:-widthTrim]
blurred_image = blurred_image[heightTrim:-heightTrim,widthTrim:-widthTrim]
output_ = output_.reshape(groundtruth_image.shape)

##------ Add your code here: save the blurred and SR images and compute the psnr
# hints: use the 'scipy.misc.imsave()'  and ' skimage.meause.compare_psnr()'
scipy.misc.imsave('blurred_image.png',blurred_image)
scipy.misc.imsave('output.png',output_)
scipy.misc.imsave('groundtruth_image.png',groundtruth_image)

print(f"\nCNN PSNR score: {skimage.measure.compare_psnr(groundtruth_image,output_)}")
print(f"Baseline PSNR score: {skimage.measure.compare_psnr(groundtruth_image,blurred_image)}")
