import os as os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import pylab as pylab
from mtens import *

el_g=[]
plt_limits=[0,20,-10,10]     # Standard Limits for plots
elmax=[]
Lmax=[]
tg=[]
Area=[]
mt8=[]
elt=0

 #EXCESS LENGHT CALCULATOR AND PLOTTER
def el_calc(args,glv):         # args = 0 read data
  if args==0: 
      
    # Specify Input variables
    ds=glv[0]                       # segment length in mtb.dat dat
    vp=glv[1]                       # polymerization speed in sims 
    il=glv[2]                       # initial length of mt
    a=read('mtb.dat')
    
    # Specify Output variables
    el_mt=[]                        # list of excess length
    
    #Processing starts here
    l=0                         
    tl=il+a[l][0]*vp
    while tl<=8.04:
        
        r    = a[l][1]
        time = a[l][0]
        tl=il+time*vp
        uvx=(r[-2]-r[-4])/np.sqrt((r[-2]-r[-4])**2+(r[-1]-r[-3])**2)
        uvy=(r[-1]-r[-3])/np.sqrt((r[-2]-r[-4])**2+(r[-1]-r[-3])**2)
        
        if tl%ds<0.0950001: 
                ex=r[-2]+uvx*(tl%ds)
                ey=r[-1]+uvy*(tl%ds)      
        el_t=tl-np.sqrt((ex)**2+(ey)**2)   # el_t excess length at given time        
        el_mt.append(el_t)                 # el_mt -> time series for el in a mt   
        lmax=l
        l=l+1 
    mt8.append([a[l-1][1][2:len(a[l-1][1]):2],a[l-1][1][3:len(a[l-1][1]):2]])
    el_g.append(el_mt)                     # el_g  -> list of time series for mt ensemble 
    elmax.append(max(el_mt))               # set of max el in a mt for ensemble 
    return el_mt

    
  if args==1:
        a=read('el.dat')
        plt.figure(figsize=(8,8))
        plt.title='el_analysis'
        plt.hist(a,np.linspace(0,1,101))
       
        plt.axis([0,1,0,30])
        plt.xlabel('el ')
        plt.ylabel('number of MTs')
        plt.savefig('el_distro.svg')
        plt.savefig('el_distro.png')
        plt.clf() 


  if args==2:
        a=read('area.dat')
        plt.figure(figsize=(8,8))
        plt.title='el_analysis'
        plt.hist(a,np.linspace(0,10,41))
        plt.axis([0,10,0,30])
        plt.xlabel('area')
        plt.ylabel('number of MTs')
        plt.savefig('sweep.svg')
        plt.savefig('sweep.png')
        plt.clf()
        
 #PLOT mtb.r from data
def mtb_plotter():                           
    fh=open('mtb.dat','rb')
    a=pickle.load(fh)
    for l in np.arange(len(a)):
        x=a[l][1][0:len(a[l][1]):2]
        y=a[l][1][1:len(a[l][1]):2]
        
        plt.figure(figsize=(8,8))
        plt.axis(plt_limits)
        #plt.axes().set_aspect('equal')
        time =str(a[l][0])
        label = "time: " + time
       
        plt.plot(x,y,label=label)
        plt.savefig('mtb_'+str(l)+'.png')
        plt.close()
def curvature(glv):
    a=read('mt8.dat')
    cv_g=[]
    avg_cv=np.zeros(len(a[0][0][0:len(a[0][0]):4])-2)
    for i in np.arange(len(a)):
        xc=a[i][0][0:len(a[i][0]):4]
        yc=a[i][1][0:len(a[i][0]):4]
        delx=xc[1:]-xc[:-1]
        dely=yc[1:]-yc[:-1]
        v1x=delx[0:-1]
        v2x=delx[1:]
        v1y=dely[0:-1]
        v2y=dely[1:]
        dp=v1x*v2x+v1y*v2y
        th=np.arccos(dp/(np.sqrt(v1x**2+v1y**2)*np.sqrt(v2x**2+v2y**2)))
        cv=(1-np.cos(th))/(glv[0]*4*np.cos(np.pi/2-th/2))
        mod_cv=np.sqrt(cv**2)
        cv_g.append(mod_cv)
        avg_cv=avg_cv+mod_cv/len(a)
    
    plt.plot((np.arange(len(mod_cv))+1)*glv[0]*4,mod_cv)
    plt.axis([0,8,0,2.0])
    plt.savefig('avg_cv.svg')
            
def tree(adrs):                 # adrs -> address of folder containing simulation for mt ensemble to be analyzed
    os.chdir(adrs)
    
    x0=os.listdir(adrs)
    
    os.mkdir('DA') # make data_analysis directory
    y0=[]
    for i in x0:
        if os.path.isdir(i)==True and i!='data_analysis':
            y0.append(i)             # Read for directories in sim folder except for data_analysis
    
    for j in y0:
        os.chdir(j)
        x1=os.listdir(os.getcwd())
        y1=[]
   
    for i in x1:
        if os.path.isdir(i)==True:
            y1.append(i)
    return y1
   
def data_analysis(adrs):
    y1=tree(adrs)
    glv=[0.1,0.4,0.15]           # Global variables [ds,vp,il]
    
    #ds=0.1
    for k in y1:
            os.chdir(k)
            el_mt=el_calc(0,glv)       # data collector
            xt,yt,area=sweep(0,glv)
            vp_set=sweep(2,glv)
            tg.append([xt,yt])
            Area.append(area)
            print os.getcwd()  #Plotting routine should be here
            os.chdir('..')
    os.chdir('../DA')
    plt.clf()  
    write('vp_set.dat',vp_set)
    write('tip_traj.dat',tg)
    write('el.dat',elmax)
    write('el_g.dat',el_g)
    write('area.dat',Area)
    write('mt8.dat',mt8)               # mt co-ordinates at a given time

    mt_ensemble()
    el_calc(1,glv)  #data plotter
    el_calc(2,glv)
    sweep(1,glv)
    var_plotter() 
    plt.close() 
    curvature(glv)
    fmode()

if __name__ == "__main__":
	import sys
	dir_list=['/vp0.4/BC0111/']
	subdir_list=['2d2k','k2']
	path_string='/work/spring2016/'
	for tt in dir_list:
	    for ut in subdir_list:
	       vp_set=[]
	       path_n=path_string+tt+ut 
	       os.chdir(path_n)
	       print os.getcwd()  
	       el_g=[]
               plt_limits=[0,20,-10,10]     # Standard Limits for plots
               elmax=[]
               Lmax=[]
               tg=[]
               Area=[]
               mt8=[]
               elt=0     
	       data_analysis(path_n)
	       
	#os.chdir('/homedir/work/fall2015')
               
                        
                                         
