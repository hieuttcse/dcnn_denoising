# 2017-10-27
# Trung-Hieu Tran @ IPVS

import os
import time
import numpy as np
import tensorflow as tf
import scipy.io as sio
from utils import *
import preprocess_images as pimages




class SDCNN(object): # Simple Denosing CNN
    def __init__(self,sess,flags):
        self.sess = sess
        self.batchSize = flags.batchSize
        self.patchSize = flags.patchSize
        self.dataset = flags.dataset
        self.checkpointDir = flags.checkpointDir
        self.dataDir = flags.dataDir
        self.overlapSize = flags.overlapSize
        self.learningRate = flags.learningRate
        self.logDir = flags.logDir
        self.epoch = flags.epoch

        data = time.strftime('%d%m')
        self.date = data
        self.build_model()
        self.count = 0

    def build_model(self):
        # Input
        self.input = tf.placeholder(tf.float32,[self.batchSize,self.patchSize,self.patchSize,1],
                                    name='input_noisy')
        # GT
        self.gt   = tf.placeholder(tf.float32,[self.batchSize,self.patchSize,self.patchSize,1],
                                    name='output_gt')

        # Define CNN
        with tf.variable_scope("denoising_net"):
            self.output = self.denoising_net(self.input)


        # Define MSE for error back-propagation
        self.MSE = tf.reduce_mean(tf.square(tf.subtract(self.output,self.gt)))

        self.saver = tf.train.Saver(max_to_keep=1)
    def build_test_model(self,ny ,nx):
         # Input
        self.input_test = tf.placeholder(tf.float32,[1,ny,nx,1],
                                    name='input_noisy')
        # GT
        self.gt_test   = tf.placeholder(tf.float32,[1,ny,nx,1],
                                    name='output_gt')

        # Define CNN
        with tf.variable_scope("denoising_net"):
            self.output_test = self.denoising_net(self.input_test,reuse=True)


        # Define MSE for error back-propagation
        self.MSE_test = tf.reduce_mean(tf.square(tf.subtract(self.output_test,self.gt_test)))

        #self.saver = tf.train.Saver(max_to_keep=1)

    def test(self,flags):
        print("... Start Testing ...")
        #load testing data
        noisyImages,gtImages = pimages.get_testing_images(self.dataDir, self.dataset)
        [nSize,ny,nx] = noisyImages.shape

        self.build_test_model(ny,nx)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        #load trainging variables
        if self.load_net(self.checkpointDir, self.dataset):
            print(' Successful loading pretrained network')
        else:
            print(' Fail loading pretrained network')
            return
        outputImages = []
        for i in range(0,nSize):
            gtBatches = gtImages[i,:,:]
            noisyBatches = noisyImages[i,:,:]
            gtBatches = np.expand_dims(gtBatches,axis = -1)
            gtBatches = np.expand_dims(gtBatches,axis = 0)
            noisyBatches = np.expand_dims(noisyBatches,axis= -1)
            noisyBatches = np.expand_dims(noisyBatches,axis= 0)
            [test_MSE,outBatches] = self.sess.run([self.MSE_test,self.output_test],
                                          feed_dict={self.input_test:noisyBatches,self.gt_test:gtBatches})
            outputImages.append(np.squeeze(outBatches))
            print(' Test  MSE for img %d: %.4f \n'%(i,test_MSE))
        return outputImages

    def train(self,flags):
        print("... Start Traning ...")
        globalStep1 = tf.Variable(0, name='global_step_train1',trainable=False)

        optimizerDenoising = tf.train.AdamOptimizer(flags.learningRate)
        optimalDenoising = optimizerDenoising.minimize(self.MSE,global_step=globalStep1)

        t_vars = tf.trainable_variables()
        self.var_one = [var for var in t_vars if 'one' in var.name]
        self.var_two = [var for var in t_vars if 'two' in var.name]
        self.var_last= [var for var in t_vars if 'last' in var.name]

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)


        #load trainging variables
        needInit = False
        if self.load_net(self.checkpointDir, self.dataset):
            print(' Successful loading pretrained network')
            #self.guarantee_initialized_variables()
            #varList = []
            #varList.extend([optimizerDenoising._beta1_power,optimizerDenoising._beta2_power])
            #init_op = tf.variables_initializer(varList)
            #self.sess.run(init_op)
        else:
            print(' Fail loading pretrained network')
            #init_op = tf.global_variables_initializer()
            #self.sess.run(init_op)kk

        # load training data
        noisyPatches, gtPatches = pimages.get_training_patches(self.dataDir, self.dataset, self.patchSize, self.overlapSize)
        noPatches = gtPatches.shape[0]
        noEval = int(math.floor(noPatches*0.01))
        noTrain = noPatches - noEval
        print(" Using %d patches for Training and %d patches for Evaluation"%(noTrain,noEval))

        gtpTrain = gtPatches[0:noTrain,:,:]
        noisypTrain = noisyPatches[0:noTrain,:,:]

        gtpEval = gtPatches[noTrain:,:,:]
        noisypEval = noisyPatches[noTrain:,:,:]
        noTraningBatches = int(math.floor(noTrain/self.batchSize))
        noEvalBatches   = int(math.floor(noEval/self.batchSize))
        for epoch in range(self.epoch):
            # open log file

            if epoch ==0:
                logTrainFile = open(os.path.join(self.logDir,self.date,'train_epoch.log'),'w')
                logEvalFile = open(os.path.join(self.logDir,self.date,'eval_epoch.log'),'w')
            else:
                logTrainFile = open(os.path.join(self.logDir, self.date, 'train_epoch.log'), 'a+')
                logEvalFile = open(os.path.join(self.logDir,self.date,'eval_epoch.log'),'a+')

            # Training code
            #   -- random permute the training set.
            ranIndex = np.random.permutation(np.asarray(range(0,noTrain)))
            noisy_train_patches = noisypTrain[ranIndex,:,:]
            gt_train_patches = gtpTrain[ranIndex,:,:]
            for bIdx in range(0,noTraningBatches):
                idxStart = bIdx * self.batchSize
                idxEnd = idxStart + self.batchSize
                in1 = noisy_train_patches[idxStart:idxEnd,:,:]
                gt1 = gt_train_patches[idxStart:idxEnd,:,:]

                # last dim should be 1
                in1 = np.expand_dims(in1,axis=-1)
                gt1 = np.expand_dims(gt1,axis=-1)

                [v,val_MSE] = self.sess.run([optimalDenoising,self.MSE],
                                          feed_dict={self.input: in1, self.gt: gt1})
                self.count +=1

                print('Epoch train[%2d] total MSE: %.4f \n'%(epoch,val_MSE))
            # Evaluating code
            ranIndex = np.random.permutation(np.asarray(range(0,noEval)))
            noisy_eval_patches = noisypEval[ranIndex,:,:]
            gt_eval_patches = gtpEval[ranIndex,:,:]
            for bIdx in range(0,noEvalBatches):
                idxStart = bIdx * self.batchSize
                idxEnd = idxStart + self.batchSize
                in1 = noisy_eval_patches[idxStart:idxEnd,:,:]
                gt1 = gt_eval_patches[idxStart:idxEnd,:,:]
                # add last dim
                in1 = np.expand_dims(in1,axis=-1)
                gt1 = np.expand_dims(gt1,axis=-1)

                [val_eval_MSE] = self.sess.run([self.MSE],
                                          feed_dict={self.input:in1,self.gt:gt1})
                print('Epoch eval[%2d] total MSE: %.4f \n'%(epoch,val_eval_MSE))

            if np.mod(epoch,100) == 0:
                logTrainFile.write('epoch %06d MSE %.6f \n'%(epoch,val_MSE))
                logTrainFile.flush()
                logEvalFile.write('epoch %06d MSE %.6f \n'%(epoch,val_eval_MSE))
                self.save_net(self.checkpointDir,self.dataset,0)

            logTrainFile.close()
            logEvalFile.close()

    def guarantee_initialized_variables(self, list_of_variables=None):
        if list_of_variables is None:
            list_of_variables = tf.global_variables()
        notInit = self.sess.run(tf.report_uninitialized_variables(list_of_variables))
        uninitialized_variables = list(tf.get_variable(name) for name in
                                       self.sess.run(tf.report_uninitialized_variables(list_of_variables)))
        self.sess.run(tf.initialize_variables(uninitialized_variables))
        return uninitialized_variables

    def load_net(self, checkpointDir, dataset):
        savedDir = os.path.join(checkpointDir, dataset)

        checkpointState = tf.train.get_checkpoint_state(savedDir)
        if checkpointState and checkpointState.model_checkpoint_path:
            checkpointName = os.path.basename(checkpointState.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(savedDir, checkpointName))
            return True
        else:
            return False

    def save_net(self, checkpointDir, dataset, step):
        savedDir = os.path.join(checkpointDir, dataset)
        if not os.path.exists(savedDir):
            os.makedirs(savedDir)
        savedFile = os.path.join(savedDir,dataset)
        self.saver.save(self.sess, savedFile, global_step=step)

    def denoising_net(self,input_,reuse=False):
       with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
           print(" variable_scope_is %s"%tf.get_variable_scope()._name)
           h1 = self.conv2d(input_, output_dim=32 , k_h=5, k_w=5, d_h=1, d_w=1, padding='SAME',name='one')
           h1 = tf.nn.relu(h1)
           h2 = self.conv2d(h1,16,k_h=1,k_w=1,d_h=1,d_w=1,padding='SAME',name='two')
           h2 = tf.nn.relu(h2)
           h3 = self.conv2d(h2,1,k_h=3,k_w=3,d_h=1,d_w=1,padding='SAME',name = 'last')
           return h3


    def conv2d(self,input_, output_dim,
               k_h, k_w, d_h, d_w, stddev=0.02, padding='SAME',
               name="conv2d"):
        with tf.variable_scope(name):
            print("variable_scope is %s" % tf.get_variable_scope()._name)
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
            # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
            return conv
