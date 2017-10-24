# 2017-10-24
# Trung-Hieu Tran @IPVS
import numpy as np
import os
import errno
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import scipy.misc

def main():
    print(" Preprocessing the input images")
    # configuration
    img_dir = "../../images/BSD68/"
    out_dir = "../../output/"
    # set noisy type to : gauss , s&p, poisson, speckle
    noisy_type = "poisson"


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
    write_pair_images(os.path.join(out_dir,"train"),trains_gt,trains_noise)
    print("... Writing testing data")
    write_pair_images(os.path.join(out_dir,"test"),tests_gt,tests_noise)

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
def generate_noisy_images(images,type):
    noisy = []
    for image in images:
        noisy.append(noising_it(image,type))
    return noisy
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