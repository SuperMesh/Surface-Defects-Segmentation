

import os, contextlib2
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from IPython.display import display, HTML
from PIL import Image
import time

from tensorflow import keras
import cv2
import tensorflow as tf

from utils import *

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

import streamlit.components.v1 as components

import warnings
warnings.filterwarnings("ignore")

IMAGE_SIZE = (256, 1600)
N_CHANNELS = 3
N_CLASSES = 4
target_size = (128, 800)
INPUT_SHAPE = (*target_size, N_CHANNELS)
classes = ['No Defects', 'Defects']

def _bytestring_feature(list_of_bytestrings):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints): # int64
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def image_bits_from_id(image_id):
    image = Image.open(paths_dict[image_id])
    image = tf.constant(image)
    image_bits = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)
    image_bits = image_bits.numpy()
    return image_bits

def create_tfrec_example(image_id, defect = 0):
    image = image_bits_from_id(image_id) 
    rle_1 = '1 0'.encode()
    rle_2 = '1 0'.encode()
    rle_3 = '1 0'.encode()
    rle_4 = '1 0'.encode()
    
    feature = {
        'image': _bytestring_feature([image]),
        'img_id': _bytestring_feature([image_id.encode()]),
        'width': _int_feature([1600]), 
        'height': _int_feature([256]),
        'rle': _bytestring_feature([rle_1, rle_2, rle_3, rle_4])
        }
    
    tfrec_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tfrec_example

def get_tfrec_fnames(id_list, description, NUM_TFRECS):
    shard_object_count = [int(len(id_list) / NUM_TFRECS)] * NUM_TFRECS

    for i in range(len(id_list) % NUM_TFRECS):  
        shard_object_count[i] = shard_object_count[i] + 1
    
    tf_record_output_filenames = ['severstal-256x1600-{}-{:02d}-{}.tfrec'.format(
        description, idx+1, shard_object_count[idx]) for idx in range(NUM_TFRECS)]
    return tf_record_output_filenames

def open_sharded_tfrecs(exit_stack, tfrec_names):
    return [exit_stack.enter_context(tf.io.TFRecordWriter(fname)) for fname in tfrec_names]

def data_meta(filenames, vars = ('img_id', 'rle')):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda x: read_tfrecord(x, vars = vars))
    return dataset

def get_eval_datasets(target_size = (128, 800), augment = False, repeat = False):
    tfrec_lists = TFRECS_TEST
    datasets = get_dataset(TFRECS_TEST, target_size, 4)
    meta_datasets = data_meta(tfrec_lists)
    return datasets, meta_datasets


st.title("Surface Defect Detection & Segmentation")

activities = ["About", "Upload Image"]
choice = st.sidebar.selectbox("Select Activty", activities)

if choice == 'Upload Image':
        st.header("Upload Image")
        image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        st.markdown("* * *")

        if image_file is not None:
            our_image = Image.open(image_file)
            st.image(our_image)
            im = our_image.save('./uploads/sample.jpg')

        if st.button('Process'):
            with st.spinner('Processing image ... '):

                
                # Create tfrec
                file_paths_test = tf.io.gfile.glob('uploads/*.jpg')
                file_ids_test = [x.split('\\')[-1] for x in file_paths_test]
                file_ids = file_ids_test
                file_paths = file_paths_test
                paths_dict = dict(zip(file_ids, file_paths))

                test_tfrec_names = get_tfrec_fnames(file_ids_test, 'single', NUM_TFRECS = 1)
                test_dict = {'ids': file_ids_test, 'tfrec_names': test_tfrec_names, 'defect': None, 'NUM_TFRECS': 1}

                for d in [test_dict]:
                    id_list = d['ids']
                    tfrec_names = d['tfrec_names']
                    defect = d['defect']
                    NUM_TFRECS = d['NUM_TFRECS']

                    with contextlib2.ExitStack() as tf_record_close_stack:
                        output_tfrecords = open_sharded_tfrecs(tf_record_close_stack, tfrec_names)    

                        for i, image_id in enumerate(id_list): 
                            tf_record = create_tfrec_example(image_id, defect)
                            output_tfrecords[i%NUM_TFRECS].write(tf_record.SerializeToString())

                TFRECS_TEST = tf.io.gfile.glob('*single*.tfrec')


                loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)
                metrics = ['accuracy', 'AUC']

                model = build_classifier('efficientnetb3', 1, INPUT_SHAPE, weights=None)
                model.load_weights('efficientnet-b3-bin-f3_128x800_acc_98328.h5')
                model.compile(optimizer='adam', loss=loss, metrics=metrics)

                dsets_bin, meta_dsets = get_eval_datasets(target_size)
                preds = model.predict(dsets_bin)

                meta_list = []

                for item in meta_dsets:
                    meta_list.append([i.numpy() for i in item])

                meta_df = pd.DataFrame(meta_list, columns = ['ImageId', 'rle'])
                meta_df['preds_prob'] = np.squeeze(preds)
                meta_df.drop(['rle'], axis = 1)

                adam = tf.keras.optimizers.Adam()
                seg_metrics = [dice, adice, chdice]

                seg_model = unet('efficientnetb1', 4, INPUT_SHAPE, weights=None)
                seg_model.load_weights('efficientnet-b1-unet-f3_128x800_adice_68377.h5')
                seg_model.compile(optimizer=adam, loss=dice_loss, metrics=seg_metrics)

                meta_dict = meta_df.to_dict()
                empty_mask = np.zeros(target_size, int) 
                n_demo = 1

                binary_thresh = 0.55
                thresh_upper = [0.7, 0.7, 0.7, 0.7]
                thresh_lower = [0.4, 0.5, 0.4, 0.5]
                min_area     = [180, 260, 200, 500]

                demo_preds = [];   # for plots  
                i = 0; 

                for p, ds_part in enumerate(dsets_bin):
                    
                    preds = seg_model.predict(ds_part, verbose = 1)    # shape 128, 800, 4
                
                    for pred in preds:
                        pred_mask = np.zeros(pred.shape, int)
                        
                        # Start validation from i=0
                        if meta_dict['preds_prob'][i] < binary_thresh:
                            st.header('**No Defect** found in the sample.')
                            prob = "{:.4f}".format(meta_dict['preds_prob'][0])
                            st.write("Defect Confidence: ", prob)
            
                            break

                        else:
                            st.header('**Defect** found in test sample.')
                            prob = "{:.4f}".format(meta_dict['preds_prob'][0])
                            st.write("Defect Confidence: ", prob)
                            for ch in range(N_CLASSES):  # 0, 1, 2, 3  
                                ch_probs = pred[..., ch]
                                ch_pred = (ch_probs > thresh_upper[ch]).astype(int) 
                                
                                if ch_pred.sum() < min_area[ch]:
                                    ch_pred = empty_mask
                                else:
                                    ch_pred = (ch_probs > thresh_lower[ch]).astype(int)
                                    pred_mask[..., ch] = ch_pred
                            
                            demo_preds.append([i, pred_mask])   

                        images_ds = dsets_bin.map(lambda x: x).unbatch()
                        masked_idxs = [p[0] for p in demo_preds]
                        
                        imgs = []
                        for i, img in enumerate(images_ds):
                            if i in masked_idxs: 
                                imgs.append(img)
                                if len(imgs) == n_demo: 
                                    break
                                    
                        demo_tuples = [(imgs[i], d[1]) for i, d in enumerate(demo_preds)]

                        seg_result = plot_gt_vs_pred(demo_tuples[0])
                        st.pyplot(seg_result)

                        mask_rgb = [(255, 48, 48), (24, 116, 205), (208, 32, 144), (34, 139, 34)]

                        base_html = '''
                        <style>
                        .dot {height: 18px; width: 25px; border-radius: 10%%; display: inline-block;
                            font-size: 15px; color: white; text-align:center;}
                        %s
                        </style>
                        <div> 
                        <span style="display: inline-block;font-size: 15px">Defect Class:</span>
                        %s</div> 
                        ''' 
                        css_str, html_str = '', ''
                        for i, c in enumerate(mask_rgb, 1):
                            css_str += '.dot%d {background-color: rgb%s}\n' % (i, c)
                            html_str += '<span class="dot dot%d">%d</span>\n' % (i, i)

                        components.html(base_html % (css_str, html_str))

                        del demo_tuples    

                    del preds, pred

                os.remove(tf.io.gfile.glob('*single*.tfrec')[0])
                os.remove(tf.io.gfile.glob('uploads/*.jpg')[0])
