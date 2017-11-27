# 17-11-27
# Trung-Hieu Tran @ IPVS


import numpy as np
import utils as mutils
import matplotlib.pyplot as plt

from sporco.admm import bpdn
from sporco.admm import cmod
from sporco.admm import dictlrn
from sporco import util
from sporco import plot


class DSC_V1(object): # Version 1
    def __init__(self,config):
        self.config = config
        self.patch_size = config['patch_size']
        self.overlap_size = config['overlap_size']
        self.dict_size = config['dict_size']
        self.data_dir = config['data_dir']
        self.dataset = config['dataset']
        self.output_dir = config['output_dir']

    def train(self):
        noisyPatches, gtPatches = mutils.get_training_patches(self.data_dir,self.dataset,self.patch_size,self.overlap_size)

        # subtract the mean from bothdata
        SL = np.reshape(noisyPatches,[noisyPatches.shape[0],self.patch_size**2])
        mSL = np.expand_dims(np.mean(SL,axis=1),axis = 1)
        rmSL = np.tile(mSL,[1,SL.shape[1]])
        SL = SL - rmSL

        SH = np.reshape(gtPatches,[gtPatches.shape[0],self.patch_size**2])
        mSH = np.expand_dims(np.mean(SH,axis=1),axis = 1)
        rmSH = np.tile(mSH,[1,SH.shape[1]])
        SH = SH - rmSH

        # make sure patch is stored in column vector
        SL = np.transpose(SL,[1,0])
        SH = np.transpose(SH,[1,0])

        hDim = SH.shape[0]
        lDim = SL.shape[0]

        # normalize signals
        SL = cmod.normalise(SL)
        SH = cmod.normalise(SH)

        # Joint training
        S = np.concatenate((SL,SH),axis = 0)
        # S = cmod.normalise(S)

        # initialize dictionary
        np.random.seed(1133221)
        DL = np.random.randn(SL.shape[0],self.dict_size)
        DH = np.random.randn(SH.shape[0],self.dict_size)

        # join learning dict
        D0 = np.concatenate((DL,DH),axis = 0)

        # X and D update options
        lmbda = 0.1
        optx = bpdn.BPDN.Options({'Verbose':False, 'MaxMainIter':1,
                                  'rho':50.0*lmbda + 0.5})
        optd = cmod.CnstrMOD.Options({'Verbose':False,'MaxMainIter':1,
                                      'rho':S.shape[1]/200.0})
        # update D update options
        optd.update({'Y0':D0,'U0':np.zeros((S.shape[0],D0.shape[1]))})

        # create X update object
        xstep = bpdn.BPDN(D0,S,lmbda,optx)
        # create D update object
        dstep = cmod.CnstrMOD(None,S,(D0.shape[1],S.shape[1]),optd)

        # create dict learn object
        opt = dictlrn.DictLearn.Options({'Verbose':True,'MaxMainIter':2})
        d = dictlrn.DictLearn(xstep=xstep,dstep=dstep,opt=opt)
        Dmx = d.solve()
        print("DictLearn solv time: %.2fs" %d.timer.elapsed('solve'))

        # get dictionary
        DL1 = Dmx[0:self.patch_size**2,:]
        DL1 = DL1.reshape(self.patch_size,self.patch_size,DL1.shape[1])
        DH1 = Dmx[self.patch_size**2:,:]
        DH1 = DH1.reshape(self.patch_size,self.patch_size,DH1.shape[1])

        itsx = xstep.getitstat()
        itsd = dstep.getitstat()
        # write logs
        mutils.write_logs(itsx,itsd,d.timer.elapsed('solve'),self.output_dir,'logs_%d_5'%self.dict_size)
        # write dictionary
        mutils.write_dicts(DL1,DH1,self.output_dir,'dicts_%d_5'%self.dict_size)






