# 2017-11-27
# Trung-Hieu Tran @IPVS
import numpy as np
import os
import errno
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import scipy.misc
import h5py

def main():
    dataname = "BSD68"
    noise_type = "poisson"
    process_images(dataname,noise_type)
    inputDir = "../../output/"
    dataset = "%s_%s"%(dataname,noise_type)
    patchSize = 10
    overlapSize = 3
    prepare_patches(inputDir,dataset,patchSize,overlapSize,"train")
    #noisyPatches,gtPatches = get_training_patches(inputDir,dataset,patchSize,overlapSize)

def get_training_patches(inputDir,dataset,patchSize,overlapSize):
    h5Filename = "%s_train_%03d_%03d.h5" % (dataset, patchSize, overlapSize)
    h5FilePath = os.path.join(inputDir, h5Filename)
    f = h5py.File(h5FilePath,'r')
    gtPatches = np.asarray(f['gt'],dtype=np.float32)
    noisyPatches = np.asarray(f['noisy'],dtype=np.float32)
    return noisyPatches,gtPatches

def prepare_patches(inputDir,dataset,patchSize,overlapSize,type):
    train_noisy,train_gt = load_data_pair(inputDir,dataset,type)
    noisyPatches, gtPatches = extract_patches(train_noisy,train_gt,patchSize,overlapSize)
    #print(len(noisyPatches))
    h5Filename = "%s_train_%03d_%03d.h5"%(dataset,patchSize,overlapSize)
    h5FilePath = os.path.join(inputDir,h5Filename)
    h = h5py.File(h5FilePath,'w')
    h.create_dataset('noisy',data=noisyPatches)
    h.create_dataset('gt',data=gtPatches)
    h.close()
def extract_patches(noisy,gt,patchSize, overlap):
    ny = gt[0].shape[0]
    nx = gt[0].shape[1]
    noPatchX = int(math.floor((nx - overlap)*1.0/(patchSize-overlap)))
    noPatchY = int(math.floor((ny - overlap)*1.0/(patchSize-overlap)))
    startX = np.asarray(np.linspace(0,(noPatchX-1)*(patchSize-overlap),noPatchX+1),dtype=np.int)
    endX = startX + patchSize

    startY = np.asarray(np.linspace(0,(noPatchY-1)*(patchSize-overlap),noPatchY),dtype=np.int)
    endY = startY + patchSize

    [indexXStart,indexYStart] = np.meshgrid(startX,startY)
    [indexXEnd,indexYEnd] = np.meshgrid(endX,endY)
    noisyPatches = []
    gtPatches = []
    for i in range(0,indexXEnd.shape[1]):
        for j in range(0,indexXEnd.shape[0]):
            patches = [ img[indexYStart[j,i]:indexYEnd[j,i],indexXStart[j,i]:indexXEnd[j,i]] for img in noisy]
            noisyPatches.extend(patches)
            patches = [ img[indexYStart[j,i]:indexYEnd[j,i],indexXStart[j,i]:indexXEnd[j,i]] for img in gt]
            gtPatches.extend(patches)
    return noisyPatches,gtPatches


def process_images(dataname,noise_type):
    print(" Preprocessing the input images")
    # configuration
    img_dir = "../../images/%s/"%dataname
    out_dir = "../../output/"
    # set noisy type to : gauss , s&p, poisson, speckle
    noisy_type = "%s"%noise_type
    dataset = "%s_%s"%(dataname,noisy_type)
    data_dir = os.path.join(out_dir,dataset)


    images = read_images(img_dir)
    print( "Reading images from %s, total %d images"%(img_dir,len(images)))
    # make sure images are the same resolution for easier processing.
    [ny,nx] = images[0].shape
    for i in range(0,len(images)):
        [tny, tnx] = images[i].shape
        if tny == nx and tnx == ny: # rotate it
            images[i] = images[i].T
        elif tnx != nx or tny !=ny:
            images[i] = np.resize(images[i],[ny,nx])
    print("... Preparing data with %s noise"%noisy_type)
    noisy_images = generate_noisy_images(images,noisy_type)

    # devide into traning set and test set
    # 90% training 20% for testing
    noTrain = int(math.floor(len(images)*90.0/100.0))
    noTest = len(images) - noTrain
    trains_noise = noisy_images[0:noTrain]
    trains_gt= images[0:noTrain]
    tests_noise = noisy_images[noTrain:]
    tests_gt  = images[noTrain:]
    print("... Writing training data")
    write_pair_images(os.path.join(data_dir,"train"),trains_gt,trains_noise)
    print("... Writing testing data")
    write_pair_images(os.path.join(data_dir,"test"),tests_gt,tests_noise)



def load_data_pair(inputDir,dataset,type):
    dataDir = os.path.join(inputDir,dataset)
    # read training data
    trainDir = os.path.join(dataDir,type)
    train_noisy = []
    train_gt = []
    noisyDir = os.path.join(trainDir,"in")
    gtDir = os.path.join(trainDir,"gt")
    for index in range(1,1000):
        noisyFile = os.path.join(noisyDir,"img_%d.png"%index)
        gtFile = os.path.join(gtDir,"img_%d.png"%index)
        if os.path.exists(gtFile) and os.path.exists(noisyFile):
            img = mpimg.imread(noisyFile)
            img = np.asarray(img,dtype=np.float32)
            train_noisy.append(img)
            img = mpimg.imread(gtFile)
            img = np.asarray(img,dtype=np.float32)
            train_gt.append(img)

    return train_noisy,train_gt

def get_testing_images(inputDir,dataset):
    noisyList, gtList =  load_data_pair(inputDir,dataset,"test")

    noisyImages = np.asarray(noisyList,dtype=np.float32)
    gtImages = np.asarray(gtList,dtype=np.float32)
    return noisyImages,gtImages

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
def write_pair_images(path,images_gt, images_in):
   gt_dir = os.path.join(path,"gt")
   in_dir = os.path.join(path,"in")
   mkdir_p(gt_dir)
   mkdir_p(in_dir)
   count = 0
   for img in images_gt:
       count = count +1
       fname = "img_%d.png"%count
       fpath = os.path.join(gt_dir,fname)
       scipy.misc.imsave(fpath,img)
       # to reserve the intensity level
       # scipy.misc.toimage(img,cmin=0.0, cmax=...).save(fpath)
   count = 0
   for img in images_in:
       count = count+1
       fname = "img_%d.png"%count
       fpath = os.path.join(in_dir,fname)
       scipy.misc.imsave(fpath,img)
       # to reserve the intensity level
       # scipy.misc.toimage(img,cmin=0.0, cmax=...).save(fpath)
def write_outputs(path,dataset,images):
    outDir = os.path.join(path,dataset)
    outDir = os.path.join(outDir,"out")
    mkdir_p(outDir)
    for i in range(1,len(images)+1):
        fname = "img_%d.png"%i
        fpath = os.path.join(outDir,fname)
        scipy.misc.imsave(fpath,images[i-1])

def generate_noisy_images(images,type):
    noisy = []
    for image in images:
        noisy.append(noising_it(image,type))
    return noisy

def write_dicts(Dl,Dh,outDir,fname):
    fname_h5 = "%s.h5"%fname
    fpath = os.path.join(outDir,fname_h5)
    mkdir_p(outDir)
    h5file = h5py.File(fpath,'w')
    h5file.create_dataset('/Dl',data=Dl)
    h5file.create_dataset('/Dh',data=Dh)
    h5file.close()

def write_logs(itsx,itsd,runTime,outDir,fname):
    fname_h5 = "%s.h5"%fname
    fpath = os.path.join(outDir,fname_h5)
    mkdir_p(outDir)
    h5file = h5py.File(fpath,'w')
    h5file.create_dataset('/itsx/ObjFun',data=np.asarray(itsx.ObjFun,dtype=np.float32))
    h5file.create_dataset('/itsx/PrimalRsdl',data=np.asarray(itsx.PrimalRsdl,dtype=np.float32))
    h5file.create_dataset('/itsx/DualRsdl',data=np.asarray(itsx.DualRsdl,dtype=np.float32))
    h5file.create_dataset('/itsd/PrimalRsdl',data=np.asarray(itsd.PrimalRsdl,dtype=np.float32))
    h5file.create_dataset('/itsd/DualRsdl',data=np.asarray(itsd.DualRsdl,dtype=np.float32))
    h5file.create_dataset('/itsx/Rho',data=np.asarray(itsx.Rho,dtype=np.float32))
    h5file.create_dataset('/itsd/Rho',data=np.asarray(itsd.Rho,dtype=np.float32))
    h5file.create_dataset('/time',data=np.asarray([runTime],dtype=np.float32))
    h5file.close()

class Bunch:
    def __init__(self,**kwds):
        self.__dict__.update(kwds)

def read_logs(outDir,fname):
    fname_h5 = "%s.h5"%fname
    fpath = os.path.join(outDir,fname_h5)
    f = h5py.File(fpath)
    itsx = Bunch(ObjFun=[],PrimalRsdl=[],DualRsdl=[],Rho=[])
    itsd = Bunch(ObjFun=[],PrimalRsdl=[],DualRsdl=[],Rho=[])
    itsx.ObjFun = np.asarray(f['/itsx/ObjFun'],dtype=np.float32)
    itsx.PrimalRsdl = np.asarray(f['/itsx/PrimalRsdl'],dtype=np.float32)
    itsx.DualRsdl = np.asarray(f['/itsx/DualRsdl'],dtype=np.float32)
    itsd.PrimalRsdl = np.asarray(f['/itsd/PrimalRsdl'],dtype=np.float32)
    itsd.DualRsdl = np.asarray(f['/itsd/DualRsdl'],dtype=np.float32)
    itsx.Rho = np.asarray(f['/itsx/Rho'],dtype=np.float32)
    itsd.Rho = np.asarray(f['/itsd/Rho'],dtype=np.float32)
    runTime = np.asarray(f['/time'],dtype=np.float32)[0]
    f.close()
    return itsx,itsd,runTime
def read_dicts(outDir,fname):
    fname_h5 = "%s.h5"%fname
    fpath = os.path.join(outDir,fname_h5)
    f = h5py.File(fpath)
    Dl = np.asarray(f['/Dl'],dtype=np.float32)
    Dh = np.asarray(f['/Dh'],dtype=np.float32)
    f.close()
    return Dl,Dh



def noising_it(image, type):
    [ny,nx] = image.shape
    if type == "gauss":
        level = 0.5
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(ny,nx))
        gauss = gauss.reshape([ny,nx])
        noisy = image + level* gauss
        return noisy
    elif type == "s&p":
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(image)
        # salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0,i-1,int(num_salt)) for i in image.shape]
        out[coords] = 1
        # pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif type =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy

def read_images(img_dir):
    images = []
    for file in os.listdir(img_dir):
        if file.endswith(".png"):
            file_path = os.path.join(img_dir,file)
            #print(file_path)
            img = mpimg.imread(file_path)
            img = np.asarray(img,dtype=np.float32)
            images.append(img)
    return images

if __name__ == "__main__":
    main()
