#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:55:07 2017

@author: panjiao
"""
import numpy as np
import math
from random import random, randint
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm

size=[512,512]

class img:
    lena = []
    lena_noisy = []
    lena_N = []
    
DTYPE = np.float
def imread(fn, dtype=DTYPE) : # img as array
    return np.array(Image.open(fn).convert('L'), dtype=dtype)
#   return scipy.misc.imread(fn).astype(dtype)
#   return cv2.imread(fn, cv2.IMREAD_GRAYSCALE).astype(dtype)
    
def corrupt_image(im,size,noise_params) :
    # read image
    signal = im if type(im) != type("") else imread(im)
    # noise
    noise = make_noise(size,noise_params)
    # corrupt it
    signal_noisy = signal + noise
    # voila!
    return signal, noise, signal_noisy


def get_value(max_value,s):
    size=len(s)-1
    #print(size)
    value_2 = 0.0
    for each in s:
        #print(each,size)
        value_2 += each*(math.pow(2,size))
        #print(value_2)
        #print(value_2)
        size-=1
    #print(value_2)    
    value = 0.0
    value = value_2/(math.pow(2,64))* max_value
    return value

def get_params(s):
    part1=s[:64]
    p1=get_value(30,part1)
    #print(p1)
    #print(len(part1))
    #print(part1)
    part2=s[64:128]
    p2=get_value(0.01,part2)
    #print(p2)
    #print(len(part2))
    #print(part2)
    part3=s[128:]
    p3=get_value(0.01,part3)
    #print(p3)
    
    p=p1,p2,p3
    return p
    
    
    #print(len(part3))
    #print(part3)
    
    #return params
def make_noise(size, params) :
    NoiseAmp, NoiseFreqRow, NoiseFreqCol = params
    h, w = size
    zero_offset = 0
    zero_offset = 1 # Matlab starts with 1
    y = np.arange(h) + zero_offset
    x = np.arange(w) + zero_offset
    col, row = np.meshgrid(x, y, sparse=True)
    noise = NoiseAmp * np.sin(2*np.pi * NoiseFreqRow * row + 2*np.pi * NoiseFreqCol * col)
    #for each in noise:
        #print(each)
    return noise

def lena_init() :
    GBL.lena = imread("lena.png")
    GBL.lena_noisy = imread("lena.png_noisy_NA_XXX_NFRow_XXX_NFCol_XXX.png")
    GBL.lena_N = np.prod(GBL.lena.shape[:2])
    
def lena_fitness(gene) :
    noise_params = get_params(gene)
    lena, noise, lena_noisy = corrupt_image(GBL.lena,size,noise_params)
    noisy_diff = GBL.lena_noisy - lena_noisy
    noisy_diff = np.sum(np.abs(noisy_diff)) / GBL.lena_N # normalized
    fitness = - noisy_diff # negative / minimize
    #mat_visual(noise)
    return fitness

def lena_print(next_population):
    mini=100
    temp=[]
    for each in next_population:
        if(abs(lena_fitness(each))<mini):
            mini=lena_fitness(each)
            temp=each
    print(get_params(temp),mini)    
    return next_population

def lena_check_stop(p):
    for s in p:
        ch,chs=s
        noise=make_noise(size,get_params(ch))
        lena_out=GBL.lena_noisy-noise
        miss_out=GBL.lena-lena_out
        if(abs(chs)<=0.5):
            print(get_params(ch),chs)
            misc.toimage(noise,cmax=255,cmin=0).save("noise.png")
            misc.toimage(lena_out,cmax=255,cmin=0).save("lean_out.png")
            misc.toimage(miss_out,cmax=255,cmin=0).save("miss_out.png")
            return 1
GBL=img()
lena_init()

def draw(p,ymax):
    plt.plot(range(len(p)),p)
    plt.axis([1,ymax,0,-30])
    plt.show()
    
def mat_visual(noise,fitness_history,iter_num_visual):
    
    loss=plt.subplot(3,1,1)
    plt.title('fitness')
    plt.axis([1,iter_num_visual,0,-25])
    draw(fitness_history,len(fitness_history))
    loss.grid(True)
    #loss.figure.savefig("loss")
    
    plt.subplot(3,3,4)
    plt.title('original lena')
    plt.imshow(GBL.lena,cmap=cm.gray)

    plt.subplot(3,3,5)
    plt.title('original noisy_lena')
    plt.imshow(GBL.lena_noisy,cmap=cm.gray)
    
    plt.subplot(3,3,6)
    plt.title('original noise')
    plt.imshow(GBL.lena_noisy-GBL.lena,cmap=cm.gray)
    
    plt.subplot(3,3,7)
    plt.title('diff')
    plt.imshow((GBL.lena_noisy-GBL.lena)-noise,cmap=cm.gray)

    plt.subplot(3,3,8)
    plt.title('lena_noisy')
    plt.imshow(noise+GBL.lena,cmap=cm.gray)
    
    plt.subplot(3,3,9)
    plt.title('noise')
    plt.imshow(noise,cmap=cm.gray)
