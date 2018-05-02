#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 13:51:49 2017

@author: tanakalab03
"""
import os
import matplotlib.pyplot as plt
from random import random, randint
import lena
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import misc

dna_bits =192
#def random_individual(n) : return [coin (0.5)  for i in range(n)]

#random 0,1
def coin(p):
    
    x = random()
    if x<p: return 1
    else: return 0

#        
def random_individual(n):
    s=[]
    for i in range(n):
        v= coin(0.5)
        s.append(v)
    return s

def fitness(s):
    f=0
    for v in s:
        if v==1 : f+=1
    return f

def initial_population(N,n):
    p=[]
    for i in range(N):
        c=random_individual(n)
        p.append(c)
    return p

'''
print("This is generation 0")
for s in p:
    print (s,fitness(s))
    
pop_fit =[]
for s in p:
    s_f = s,fitness(s)
    pop_fit.append(s_f)
'''
#print(len(pop_fit))   50
#print("i dont know")
#print (pop_fit) 是[([s1],f1),([s2],f2)。。。。。。([sn],fn)]
    
def crossover(c):
    c1,c2=c
    n = len(c1)
    p = randint(0,n)
    #print (p)  # can see where to cross
    c1[:3]
    c1[3:]
    nc1 = c1[:p] + c2[p:]
    nc2 = c2[:p] + c1[p:]
    return nc1, nc2

def tournament(c1_f1,c2_f2):
    c1,f1 =c1_f1
    c2,f2 =c2_f2
    if f1>=f2 : return c1
    return c2

def selection(population_fitness):
    N=len(population_fitness)
    #print(N)
    new_population = []
    for i in range(N):
        i1,i2 = randint(0,N-1),randint(0,N-1)
        c1_f1 = population_fitness[i1]
        #print(i1,i2)
        c2_f2 = population_fitness[i2]
        nc1 = tournament(c1_f1,c2_f2)
        new_population.append(nc1)
    return new_population

def select_parents(population_fitness):
    new_population=selection(population_fitness)
    #print(new_population)
    next_population=[]
    parent_pair=[]
    i=0
    while(i<len(new_population)):
        #print(len(new_population))
        parent_pair=new_population[i],new_population[i+1]
        #print(new_population[i])
        #print(parent_pair)
        i=i+2
        next_population.append(parent_pair)
        #print(next_population)
    #print(len(next_population))
    return next_population
        
        
    
def mutation(c1):
    n=len(c1)
    p=randint(0,n-1)
    nc1=c1[:]
    nc1[p]=1-nc1[p] # change 1 to 0 and 0 to 1 amazing
    return nc1

def check_stop(p):
    for s in p:
        ch,chs=s
        #print("ch=/.................",ch)
        #print("chs/////////////////",chs)
        print(ch,fitness(ch))
        if(chs==dna_bits):
            return 1; 
        #else:
          #  return 0

def run(iter,pop,p_c,p_m):
        n = pop
        prob_c=p_c
        prob_m=p_m
        population = initial_population(n,dna_bits)
        generation=1
        #print(population)
        fitness_average=[]
        fitness_history=[]
        iter_num=iter
        iter_num_visual =iter_num
        plt.close()
        fig=plt.figure(num='visualize',figsize=(36,36))
        plt.grid(True)
        plt.ion()
        while iter_num>=0:
            print("********************************This is Generation:",generation,"*****************************")
            '''
            This is maxone 
            '''
            #fits_pops = [(ch,fitness(ch)) for ch in population]
            #fitness_history.append(visualize(fits_pops))
            #if check_stop(fits_pops): break
            '''
            This is lena problem
            '''
            fits_pops = [(ch,lena.lena_fitness(ch)) for ch in population]
            mini,noise,average= visualize(fits_pops)             
            fitness_history.append(mini)
            fitness_average.append(average)
            lena.mat_visual(noise,fitness_history,iter_num_visual)
            if((iter_num%(iter_num_visual/10))==0):
                misc.toimage(noise,cmax=255,cmin=0).save(os.path.abspath('.')+"\images\\noise_"+str(iter_num)+".png")
                lena_out=lena.GBL.lena_noisy-noise
                miss_out=lena.GBL.lena-lena_out
                misc.toimage(lena_out,cmax=255,cmin=0).save(os.path.abspath('.')+"\images\\lean_out_"+str(iter_num)+".png")
                misc.toimage(miss_out,cmax=255,cmin=0).save(os.path.abspath('.')+"\images\\miss_out_"+str(iter_num)+".png")
            #if lena.lena_check_stop(fits_pops): break
            population = breed_population(fits_pops,prob_c,prob_m) 
            iter_num-=1
            generation+=1
            plt.pause(0.05)
        #plt.savefig("figure.png")
        return population,fitness_history,fitness_average
          
        
        
def test_iter():
    fit_draw=[]
    fit_ave=[]
    iter_num_list=[10,50,100,1000]
    #color_list=['b','g','r','k']
    for each in iter_num_list:
        iter_num=each
        p,f,a=run(iter_num,100,0.6,0.05)
        fit_draw.append(f)
        fit_ave.append(a)
    #s=zip(iter_num_list,fit_draw)
    fig_iter=plt.figure(num='iteration',figsize=(20,5))
    plt.title('iteration=xxx,pop_num=100,crossover=0.6,mutation=0.05')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    #plt.xlim((1,20))
    #plt.ylim((-25,0))
    #i=0
    for each in fit_draw:
        draw(each,1000)
        #draw(each,color_list[i])
        #i=i+1
    fig_iter.savefig("loss_iter.png")
    plt.clf()
    
    fig_iter_ave=plt.figure(num='iteration',figsize=(20,5))
    plt.title('iteration=xxx,pop_num=100,crossover=0.6,mutation=0.05')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    #plt.xlim((1,20))
    #plt.ylim((-25,0))
    #i=0
    for each in fit_ave:
        draw(each,1000)
        #draw(each,color_list[i])
        #i=i+1
    fig_iter_ave.savefig("loss_iter_ave.png")
    
def test_pop():
    fit_draw=[]
    fit_ave=[]
    pop_num_list=[4,10,100,1000]
    for each in pop_num_list:
        pop_num=each
        p,f,a=run(100,pop_num,0.6,0.05)
        fit_draw.append(f)
        fit_ave.append(a)
    #s=zip(iter_num_list,fit_draw)
    fig_iter=plt.figure(num='iteration',figsize=(20,5))
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('iteration=100,pop_num=xxx,crossover=0.6,mutation=0.05')
    plt.grid(True)
    #plt.xlim((1,20))
    #plt.ylim((-25,0))
    for each in fit_draw:
        draw(each,100)
    fig_iter.savefig("loss_pop.png")
    plt.clf()
    fig_iter_ave=plt.figure(num='iteration',figsize=(20,5))
    plt.title('iteration=100,pop_num=xxx,crossover=0.6,mutation=0.05')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    #plt.xlim((1,20))
    #plt.ylim((-25,0))
    #i=0
    for each in fit_ave:
        draw(each,100)
        #plt.plot(range(len(each)),each)
       # fig_iter_ave.axis([1,10,0,-30])
        #fig_iter_ave.show()
        
    fig_iter_ave.savefig("loss_pop_ave.png")
    
def test_cross():
    fit_draw=[]
    fit_ave=[]
    cross_num_list=[0.0,0.2,0.6,1.0]
    for each in cross_num_list:
        cross_num=each
        p,f,a=run(100,100,cross_num,0.05)
        fit_draw.append(f)
        fit_ave.append(a)
    #s=zip(iter_num_list,fit_draw)
    fig_iter=plt.figure(num='iteration',figsize=(20,5))
    plt.title('iteration=100,pop_num=100,crossover=xxx,mutation=0.05')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    #plt.xlim((1,20))
    #plt.ylim((-25,0))
    for each in fit_draw:
        draw(each,100)
    fig_iter.savefig("loss_cross.png")
    
    plt.clf()
    fig_iter_ave=plt.figure(num='iteration',figsize=(20,5))
    plt.title('iteration=100,pop_num=100,crossover=xxx,mutation=0.05')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    #plt.xlim((1,20))
    #plt.ylim((-25,0))
    #i=0
    for each in fit_ave:
        draw(each,100)
        #plt.plot(range(len(each)),each)
       # fig_iter_ave.axis([1,10,0,-30])
        #fig_iter_ave.show()
    fig_iter_ave.savefig("loss_cross_ave.png")

def test_mut():
    fit_draw=[]
    fit_ave=[]
    mutation_num_list=[0.0,0.01,0.05,0.5,1]
    for each in mutation_num_list:
        mutation_num=each
        p,f,a=run(100,100,0.6,mutation_num)
        fit_draw.append(f)
        fit_ave.append(a)
    #s=zip(iter_num_list,fit_draw)
    fig_iter=plt.figure(num='iteration',figsize=(20,5))
    plt.title('iteration=100,pop_num=100,crossover=0.6,mutation=xxx')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    #plt.xlim((1,20))
    #plt.ylim((-25,0))
    for each in fit_draw:
        draw(each,100)
    fig_iter.savefig("loss_mut.png")
    
    plt.clf()
    fig_iter_ave=plt.figure(num='iteration',figsize=(20,5))
    plt.title('iteration=100,pop_num=100,crossover=0.6,mutation=xxx')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    #plt.xlim((1,20))
    #plt.ylim((-25,0))
    #i=0
    for each in fit_ave:
        draw(each,100)
        #plt.plot(range(len(each)),each)
       # fig_iter_ave.axis([1,10,0,-30])
        #fig_iter_ave.show()
    fig_iter_ave.savefig("loss_mut_ave.png")
        
def breed_population(fitness_population,prob_c,prob_m):
    parent_pairs = select_parents(fitness_population)
    size = len(parent_pairs)
    next_population = []
    for k in range(size) :
        parents = parent_pairs[k]
        cross = random() < prob_c
        children = crossover(parents) if cross else parents
        for ch in children:
            mutate = random() < prob_m
            next_population.append(mutation(ch) if mutate else ch)
    '''
    This is printer for lena problem.
    '''
    #lena.lena_print(next_population)
    
    return next_population
    
def visualize(p):
    mini=100
    sum=0
    temp=[]
    for s in p:
        ch,chs=s
        sum=sum+chs
        if(abs(chs)<abs(mini)):
            sum=sum+chs
            mini=chs
            temp=ch
        noise_params = lena.get_params(temp)
        original_lena, noise, lena_noisy = lena.corrupt_image(lena.GBL.lena,lena.size,noise_params)
    #lena.mat_visual(noise)
    #print("dddddd")
    #print(mini)
    #print(sum)
    #print(len(p))
    average=sum/len(p)
    print(lena.get_params(temp),mini)
    return mini,noise,average
        
def draw(p,ymax):
    plt.plot(range(len(p)),p)
    plt.axis([1,ymax,0,-30])
    plt.show()
    
lena.lena_init()

test_iter()
test_pop()
test_cross()
test_mut()

input('Press Enter to exit...')
