# 2017-10-27
# Trung-Hieu Tran @ IPVS

import os
import tensorflow as tf
import numpy as np
import math
import time
from denoise_cnn import SDCNN
import pprint
import preprocess_images as pimages

#global flags
flags = tf.app.flags
flags.DEFINE_integer("epoch",10000,"numer of iteration of traning ")
flags.DEFINE_float("learningRate",10**-4,"learning rate for traning")
flags.DEFINE_integer("patchSize",32, "patch size for traning data")
flags.DEFINE_string("dataset","BSD68_poisson","name of training dataset")
flags.DEFINE_string("checkpointDir","checkpoint","name of folder for storing checkpoint")
flags.DEFINE_string("logDir","logs", "name of folder for storing logs")
flags.DEFINE_integer("batchSize",100,"setting the patch size for training")
flags.DEFINE_integer("overlapSize",8," overlaping between patch")
flags.DEFINE_string("dataDir","../../output","name of folder storing training data")

flags.DEFINE_boolean("isTrain",True,"True for training and False for testing")

def main(_):
    pp= pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)
    FLAGS = flags.FLAGS

    checkFolder()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        if FLAGS.isTrain:
            print(" ... TRAIN")
            DCNN = SDCNN(sess,FLAGS)
            DCNN.train(FLAGS)
        else:
            print(" ... TEST")
            DCNN = SDCNN(sess,FLAGS)
            outputImages = DCNN.test(FLAGS)
            pimages.write_outputs(FLAGS.dataDir, FLAGS.dataset, outputImages)
    #pp.

def checkFolder():
    F = flags.FLAGS
    if not os.path.exists(F.checkpointDir):
        os.makedirs(F.checkpointDir)
    if not os.path.exists(F.logDir):
        os.makedirs(F.logDir)
    date = time.strftime("%d%m")
    if not os.path.exists(os.path.join(F.logDir,date)):
        os.makedirs(os.path.join(F.logDir,date))
if __name__ == '__main__':
    tf.app.run()
    #main("")
