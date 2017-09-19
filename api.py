#!/usr/bin/env python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#from PIL import Image
import cv2
import tensorflow as tf
import numpy as np
import sys
import select
from IPython import embed
from tensorflow.python.client import timeline

#import imagenet_input as data_input

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth=True

import resnet_mix as resnet

checkpoint= os.path.dirname(os.path.realpath(__file__)) + '/ckpt/model.ckpt-7799'

h0, w0 = 256, 256
crop_size = 224

batch_size = None
gpu_nums = 1

batch_indices = {}
batch_indices['use_split'] = False
batch_indices['t1_batch'] = batch_size
batch_indices['t2_batch'] = batch_size


h_max=h0-crop_size
w_max=w0-crop_size
h_max_half=int(0.5*h_max)
w_max_half=int(0.5*w_max)
x_off=w_max_half
y_off=h_max_half

gender_list=['Female', 'Male']
smile_list=['No Smile', 'Smile']
glass_list=['No Glass', 'Glass']

def resize_crop_image(img):
    assert(len(img.shape)==3)
    img = cv2.resize(img, (w0,h0), interpolation=cv2.INTER_LINEAR)
    img = img[y_off:y_off+crop_size, x_off:x_off+crop_size, :]
    return img

def preprocess_image(Image_data, input_rgb=True):
    assert(len(Image_data.shape)==4)
    if input_rgb:
        Image_data = Image_data[:, :, :, ::-1] # convert to bgr
    Image_data[:, :, :, 0] -= 103.939
    Image_data[:, :, :, 1] -= 116.779
    Image_data[:, :, :, 2] -= 123.68
    return Image_data

def de_preprocess_image(image_demeaned):
    return (image_demeaned+[103.939, 116.779, 123.68])[:,:,::-1].astype(np.uint8)


def visual_results(Image_data, preds, Labels=None, Top=0):
    from pylab import plt
    pred_age_value = preds['age']
    pred_gender_value = preds['gender']
    pred_smile_value = preds['smile']
    pred_glass_value = preds['glass']
    Num = Image_data.shape[0] if Top==0 else Top

    for k in xrange(Num):
        print k, Num
        plt.figure(1)
        plt.imshow(de_preprocess_image(Image_data[k]))
        title_str='Prediction: Age %0.1f, %s, %s, %s.'%(
                pred_age_value[k],
                gender_list[pred_gender_value[k]],
                glass_list[pred_glass_value[k]],
                smile_list[pred_smile_value[k]])
        x_label_str = 'GT: '
        try:
            x_label_str = x_label_str + 'Age %0.1f' % Labels['age'][k]
        except:
            pass
        try:
            x_label_str = x_label_str + '%s, %s, %s' %(
                gender_list[int(Labels['gender'][k])],
                glass_list[int(Labels['glass'][k])],
                smile_list[int(Labels['smile'][k])])
        except:
            pass

        plt.title(title_str)
        plt.xlabel(x_label_str)
        plt.show()

def load_model():
#with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False, name='global_step')
    hp = resnet.HParams(batch_size=batch_size,
                        batch_indices=batch_indices,
                        num_gpus=gpu_nums,
                        num_classes_age=101,
                        num_classes_gender=2,
                        num_classes_smile=2,
                        num_classes_glass=2,
                        input_size=crop_size,
                        weight_decay=0,
                        momentum=0,
                        finetune=0)
    network_val = resnet.ResNet(hp, global_step, name="val")
    network_val.build_model()
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session()
    sess.run(init)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
    print('Load checkpoint %s' % checkpoint)
    saver.restore(sess, checkpoint)
    return sess, network_val

def predict(sess, net, Image_data):
    [preds_age, preds_gender, preds_smile, preds_glass] = sess.run(
                        [net.preds_age, net.preds_gender, net.preds_smile, net.preds_glass], 
                                feed_dict={net.is_train:False,
                                           net.gpu0_images: Image_data})    

    return {'age': preds_age, 'gender':preds_gender, 'smile':preds_smile, 'glass': preds_glass}




if __name__ == '__main__':
    sess, network_val = load_model()

    img_add = '/home/lijianshu/Multitask/data/MTFL/crop3_lfw_5590/Aaron_Guiel_0001.jpg'
    #img=np.asarray(Image.open(img_add).convert('RGB'), dtype=np.float32)
    #img=resize_crop_image(img)
    #Image_data=preprocess_image(img[np.newaxis], input_rgb=True)

    img=cv2.imread(img_add, cv2.IMREAD_COLOR).astype(np.float32)
    img=resize_crop_image(img)
    Image_data=preprocess_image(img[np.newaxis], input_rgb=False)

    preds = predict(sess, network_val, Image_data)

    visual_results(Image_data, preds)
