

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import os, contextlib2

from skimage.segmentation import find_boundaries

import tensorflow.keras.layers as L
import efficientnet.tfkeras as efn
import re

import streamlit as st

IMAGE_SIZE = (256, 1600)
N_CHANNELS = 3
N_CLASSES = 4
classes = ['No Defects', 'Defects']

# Read tfrec

def read_tfrecord(example, vars = ('image', 'rle')):
    features = {
        'img_id': tf.io.FixedLenFeature([], tf.string), 
        'image': tf.io.FixedLenFeature([], tf.string), 
        'rle': tf.io.FixedLenFeature([4], tf.string),
        }
    features = {k: features[k] for k in vars}
    example = tf.io.parse_single_example(example, features)
    return [example[var] for var in features]

def load_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    return dataset.map(read_tfrecord)

def rle2mask(rle, mask_shape):
    '''
    Converts a run lenght encoding (RLE) into a mask of shape mask_shape
    Args:
        rle: (str or bytestring) run lenght encoding. A series of space
            separated start-pixel run pairs.
        mask_shape: (tuple of 2 ints) the 2D expected shape of the mask
    Returns: mask of shape mask_shape
    '''
    size = tf.math.reduce_prod(mask_shape)

    s = tf.strings.split(rle)
    s = tf.strings.to_number(s, tf.int32)

    starts = s[0::2] - 1
    lens = s[1::2]

    total_ones = tf.reduce_sum(lens)
    ones = tf.ones([total_ones], tf.int32)

    r = tf.range(total_ones)
    lens_cum = tf.math.cumsum(lens)
    s = tf.searchsorted(lens_cum, r, 'right')
    idx = r + tf.gather(starts - tf.pad(lens_cum[:-1], [(1, 0)]), s)

    mask_flat = tf.scatter_nd(tf.expand_dims(idx, 1), ones, [size])
    mask = tf.reshape(mask_flat, (mask_shape[1], mask_shape[0]))
    return tf.transpose(mask)

def build_mask_array(rle, mask_size, n_classes=1):
    '''
    Converts a RLE or a list of RLEs, into an array of
    shape (*mask_size, n_classes)
    '''
    if n_classes == 1:
        mask = rle2mask(rle, mask_size)
        mask = tf.expand_dims(mask, axis=2)
    else:
        mask = [rle2mask(rle[i], mask_size) for i in range(n_classes)]
        mask = tf.stack(mask, axis = -1)
    mask = tf.reshape(mask, (*mask_size, n_classes))
    return mask

def decode_resize_inputs(inputs, target_size, image_size = IMAGE_SIZE,
                         n_channels = N_CHANNELS, n_classes = N_CLASSES):
    (image_data, rle) = inputs[:2]
    image = tf.image.decode_jpeg(image_data, n_channels)
    image = tf.cast(image, tf.float32) / 255.0  
    mask = build_mask_array(rle, image_size, n_classes) 

    if target_size != image_size:
        image = tf.image.resize(image, target_size)
        mask = tf.image.resize(mask, target_size)

    image = tf.reshape(image, [*target_size, n_channels]) 
    mask = tf.reshape(mask, [*target_size, n_classes])
    return (image, mask)

def final_reshape(inputs, target_size, n_channels = N_CHANNELS, 
                  n_classes = N_CLASSES):
 
    image, mask = inputs
    image = tf.reshape(image, [*target_size, n_channels]) 

    return image

def get_dataset(filenames, target_size, batch_size,
                cache   = False,
                repeat  = False, 
                shuffle = False,
                drop_remainder = False
                ):

      dataset = load_dataset(filenames)

      dataset = dataset.map(lambda *data: decode_resize_inputs(data, target_size))
    
      dataset = dataset.map(lambda *data: final_reshape(data, target_size))

      if cache: dataset = dataset.cache()  
      if repeat: dataset = dataset.repeat() 
      if shuffle: dataset = dataset.shuffle(shuffle, reshuffle_each_iteration=True) 

      dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
      return dataset

cnn_dict = {
    'efficientnetb0': (efn.EfficientNetB0),
    'efficientnetb1': (efn.EfficientNetB1),
    'efficientnetb2': (efn.EfficientNetB2),
    'efficientnetb3': (efn.EfficientNetB3),
    'efficientnetb4': (efn.EfficientNetB4),
    'efficientnetb5': (efn.EfficientNetB5),
    'efficientnetb6': (efn.EfficientNetB6),
    'efficientnetb7': (efn.EfficientNetB7),
    # 'efficientnetL2': (efn.EfficientNetL2),
    'resnet50':       (tf.keras.applications.ResNet50),
    'resnet101':      (tf.keras.applications.ResNet101),
    'resnet152':      (tf.keras.applications.ResNet152)
    }

def get_cnn_func(cnn_name):
    if isinstance(cnn_name, int):
        if cnn_name < 8:
            cnn_key = 'efficientnetb{}'.format(cnn_name)
        else: cnn_key = 'resnet{}'.format(cnn_name)

    else: cnn_key = re.sub('[\W_]', '', cnn_name).lower()
    return cnn_dict[cnn_key]


def build_classifier(base_name, n_classes, input_shape = (None, None, 3),
                     weights ='imagenet', head_dropout=None, name_suffix=''):
    '''
    Assembles an CNN classifier
    Args:
        base_name: (str) one of 'efficientnetb0' to 'efficientnetb7' or
            'resnet50' to 'resnet152'.
        n_classes: (int) number of classes. i.e. number of filters in the output layer.
        weights: pretrained weights. one of None, 'imagenet',
            'noisy-student' or the path to the weights file to be loaded.
            Note: 'noisy-student' is only available for efficientnet classifiers.
        head_dropout: None or float. If float, the dropout rate before the last
            dense layer.
        name_suffix: (str) string to add to the model name
    '''
    CNN = get_cnn_func(base_name)
    base = CNN(input_shape=input_shape, weights=weights, include_top=False)

    x = L.GlobalAveragePooling2D()(base.output)
    if head_dropout:
        x = L.Dropout(head_dropout)(x)
    x = L.Dense(n_classes, activation='sigmoid', name='output_layer')(x)
    model = tf.keras.Model(inputs=base.input, outputs=x,
                           name='{}{}'.format(base.name.title(), name_suffix))
    return model

def block_names(stage):
    conv_name = 'decoder_stage{}_conv'.format(stage)
    bn_name = 'decoder_stage{}_bn'.format(stage)
    relu_name = 'decoder_stage{}_relu'.format(stage)
    up_name = 'decoder_stage{}_upsample'.format(stage)
    return conv_name, bn_name, relu_name, up_name

def to_tuple(x):
    if isinstance(x, tuple):
        if len(x) == 2:
            return x
    elif np.isscalar(x):
        return (x, x)
    raise ValueError('Value should be tuple of length 2 or int value, got "{}"'.format(x))

def get_layer_number(model, layer_name):
    """
    Finds layer in tf.keras model by name
    Args: model: tf.keras `Model`
          layer_name: str, name of layer
    Returns: index of layer
    Raises: ValueError: if model does not contain layer with such name
    """
    for i, l in enumerate(model.layers):
        if l.name == layer_name:
            return i
    raise ValueError('No layer with name {} in  model {}.'.format(layer_name, model.name))

unet_skip_connections_dict = {
    'efficientnet': ('block6a_expand_activation', 'block4a_expand_activation',
                     'block3a_expand_activation', 'block2a_expand_activation'),
    'resnet':       ('conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out',
                     'conv1_relu')
    }

# Unet Blocks
def ConvNRelu(filters, kernel_size, use_batchnorm=False, conv_name='conv', bn_name='bn', relu_name='relu'):
    def layer(x):
        x = L.Conv2D(filters, kernel_size, padding="same", name=conv_name, use_bias=not(use_batchnorm))(x)
        if use_batchnorm:
            x = L.BatchNormalization(name=bn_name)(x)
        x = L.Activation('relu', name=relu_name)(x)
        #x = L.LeakyReLU(alpha=0.1,  name=relu_name)(x)
        return x
    return layer

def DecoderUpsample2DBlock(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),
                     use_batchnorm=False, skip=None):
    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name = block_names(stage)
        x = L.UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)
        if skip is not None:
            x = L.Concatenate()([x, skip])

        x = ConvNRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1', relu_name=relu_name + '1')(x)
        x = ConvNRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)
        return x
    return layer

def DecoderTranspose2DBlock(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),
                      transpose_kernel_size=(4,4), use_batchnorm=False, skip=None):
  
    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name = block_names(stage)
        x = L.Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name=up_name, use_bias=not(use_batchnorm))(input_tensor)
  
        if use_batchnorm:
            x = L.BatchNormalization(name=bn_name+'1')(x)
  
        x = L.Activation('relu', name=relu_name+'1')(x)

        if skip is not None:
            x = L.Concatenate()([x, skip])

        x = ConvNRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)
        return x
    return layer

# Unet Model
def unet_model(backbone, num_classes, skip_connection_layers,
               decoder_filters=(256,128,64,32,16),
               upsample_rates=(2,2,2,2,2),
               n_upsample_blocks=5,
               block_type='transpose',
               activation='sigmoid',
               use_batchnorm=True):

    input = backbone.input
    x = backbone.output

    if block_type == 'transpose':
        up_block = DecoderTranspose2DBlock
    elif block_type == 'upsampling':
        up_block = DecoderUpsample2DBlock
    else: raise Exception('block_type must be one of \'transpose\' or \'upsampling\'')

    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                               for l in skip_connection_layers])

    for i in range(n_upsample_blocks):   
        skip_connection = None    

        if i < len(skip_connection_idx):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        upsample_rate = to_tuple(upsample_rates[i])

        x = up_block(decoder_filters[i], i, upsample_rate=upsample_rate,
                     skip=skip_connection, use_batchnorm=use_batchnorm)(x)

    x = L.Conv2D(num_classes, (3,3), padding='same', name='final_conv')(x)
    x = L.Activation(activation, name=activation)(x)

    model = tf.keras.Model(input, x, name='{}-Unet'.format(backbone.name.title()))
    return model

# Unet Builder (for matching and unmatching weights)
def unet(backbone_name, num_classes, input_shape=(None, None, 3),
         skip_connection_layers='default',
         decoder_filters=(256,128,64,32,16),
         upsample_rates=(2,2,2,2,2),
         n_upsample_blocks=5,
         block_type='transpose',
         activation='sigmoid',
         use_batchnorm=True,
         freeze_backbone = False,
         weights='imagenet',
         weights_by_name = False):

    if not weights_by_name: 
        cnn_weights = weights
    else: 
        cnn_weights = None

    CNN = get_cnn_func(backbone_name)
    backbone = CNN(include_top=False, weights=cnn_weights, input_shape=input_shape)
    
    if weights_by_name:
        backbone.load_weights(weights, by_name=True)

    backbone.trainable = not freeze_backbone

    if skip_connection_layers == 'default':
        if 'efficientnet' in backbone.name.lower():
            skips_dict_key = 'efficientnet'
        elif 'resnet' in backbone.name.lower():
            skips_dict_key = 'resnet'
        skip_connection_layers = unet_skip_connections_dict[skips_dict_key]

    model = unet_model(backbone, num_classes, skip_connection_layers,
                       decoder_filters, upsample_rates, n_upsample_blocks,
                       block_type, activation, use_batchnorm)
    return model

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    den = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth
    dice = (2. * intersection + smooth)/den
    return dice

def dice_avg(y_true, y_pred):
    '''
    Calculates the image-wise average dice coefficient in a batch of masks.
    args:
        y_true: tensor of shape (N, H, W, C), where N is the batch size and C
                is the number of classes (i.e. num of channels of the mask)
        y_pred: tensor of shape y_true.shape
    Returns the average dice coefficient accross images in the batch.
    '''
    img_dice = tf.vectorized_map(lambda args: dice_coef(*args), [y_true, y_pred])
    img_dice_avg = tf.math.reduce_mean(img_dice)
    return img_dice_avg

def channel_avg_dice(y_true, y_pred, smooth = 1.):
    '''
    Calculates the channel-wise average dice coefficient in a batch of masks.
    args:
        y_true: tensor of shape (N, H, W, C), where N is the batch size and C
                is the number of classes (i.e. num of channels of the mask)
        y_pred: tensor of shape y_true.shape
    Returns the average dice coefficient accross all channels.
    '''
    # Stacking the channels along the 1st dimension.
    # They will be ordered by channel_num then by image i.e. y_true[2] will be
    # the channel 0 of image 2
    y_true = tf.concat(tf.unstack(y_true, axis=-1), axis=0)
    y_pred = tf.concat(tf.unstack(y_pred, axis=-1), axis=0)

    channel_dice = tf.vectorized_map(lambda args: dice_coef(*args, smooth),
                                     [y_true, y_pred])
    channel_dice_avg = tf.math.reduce_mean(channel_dice)
    return channel_dice_avg

def dice_loss(y_true, y_pred, smooth = 1.):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = y_true_flat * y_pred_flat
    den = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth
    score = (2. * tf.reduce_sum(intersection) + smooth)/den
    return 1. - score

def dice_coefficient(y_true, y_pred, smooth=1.):
    # compatible with evaluation in prediction loop
    smooth = tf.constant(smooth)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    denom = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth
    dice = (tf.constant(2.) * intersection + smooth)/denom
    return dice

def bce_dice_loss(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def dice(y, p): 
    return dice_coef(y, p)

def adice(y, p): 
    return dice_avg(y, p)

def chdice(y, p): 
    return channel_avg_dice(y, p)

mask_rgb = [(255, 48, 48), (24, 116, 205), (208, 32, 144), (34, 139, 34)]

def plot_gt_vs_pred(pred_demo):
    rgb_colors = mask_rgb

    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12, 4))
    ax1.imshow(pred_demo[0])
    ax1.axis('off')
    ax1.set_title('Test Image')

    ax2.imshow(pred_demo[0])
    mask = pred_demo[1]
    masks = [(contoured_mask(mask[..., c], rgb_color = rgb_colors[c], 
                        alpha=0.3)) for c in range(mask.shape[-1])]
    [ax2.imshow(m, alpha = 1) for m in masks]
    ax2.axis('off')
    ax2.set_title('Prediction')
    plt.tight_layout()
    plt.show()
    return fig

def contoured_mask(mask, rgb_color = (0, 0, 255), alpha = 0.2):
    rgb_color = np.array(rgb_color)/255
    epsilon = 10e-6
    mask = np.squeeze(mask+epsilon).round().astype(int)
    boundary = find_boundaries(mask).astype(float)
    mask_ = np.zeros((*mask.shape, 4))
    for i, c in enumerate(rgb_color):
        mask_[..., i] = c
    mask_[..., 3] = np.maximum(alpha*mask, boundary)
    return mask_

def class_norm():
    mask_rgb = [(230, 184, 0), (0, 128, 0), (102, 0, 204), (204, 0, 102)]
    fig, ax = plt.subplots(1, 4, figsize=(4, 1))
    for i in range(4):
        ax[i].axis('off')
        ax[i].imshow(np.ones((50, 50, 3), dtype=np.uint8) * mask_rgb[i])
        ax[i].set_title("class {}".format(i+1))
    fig.suptitle("Defect Classes")
    plt.show()
    return fig