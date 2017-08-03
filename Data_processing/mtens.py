import os as os
import cPickle as pickle
import matplotlib.pyplot as plt
import pylab as pylab
import numpy as np
from scipy.spatial import ConvexHull
from scipy.fftpack import fft
from scipy import interpolate
import plotter
vp_set=[]
def write(fname,dump_name):        # enter FILENAME (fname) for dumping and the VARIABLE (dump_name) to be dumped
    fh=open(fname,'wb')
    pickle.dump(dump_name,fh)
    file.close(fh)              

def read(fname):        # enter FILENAME (fname) for reading stored numpy array   [[time,r]...] 
    fh=open(fname,'rb')
    a = pickle.load(fh)
    file.close(fh)        
    return a


def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def mt_ensemble():
    a=read('mt8.dat')
    for k in np.arange(len(a)):
        plt.plot(a[k][0],a[k][1],'b')
        plt.hold=True
        plt.axis([0,10,-5,5])
        pylab.axes().set_aspect('equal')
    plt.savefig('mtens.svg')
    plt.savefig('mtens.png')
    plt.close()
    
    plt.clf()
    th=np.arange(len(a))/(len(a)*1.0)*2*np.pi
    for i in range(len(th)):
        z=np.arctan2(a[i][1],a[i][0])
        zsq=np.sqrt(a[i][0]**2+a[i][1]**2)
        plt.plot(zsq*np.cos(th[i]+z),zsq*np.sin(th[i]+z))
        plt.hold=True
    plt.axis([-8,8,-8,8])
    pylab.axes().set_aspect('equal')
    plt.savefig('mt_span.png')
    plt.savefig('mt_span.svg')
    plt.close()
    plt.clf()


def sweep(args,glv):
    if args==0:
        a=read('mtb.dat')
        xt=[0]
        yt=[0]
        it=0
        ds=glv[0]
        vp=glv[1]
        il=glv[2]   
        points=[]
        while (a[it][0]*vp+il)<8.04:
            r=a[it][1]
            for jt in (np.arange(len(r)/2-1)+1):           # ignore 1st point-->-ds due to clamped BC
                points.append([r[2*jt],r[2*jt+1]])   
            #plt.plot(r[0:len(r):2],r[1:len(r):2],'b')
            #plt.hold=True
            #plt.axis([0,10,-5,5])
            #pylab.axes().set_aspect('equal')
            if (a[it][0]*vp+il+0.01)%1<=0.001 or (a[it][0]*vp+il+0.01)%1>=0.999 :
                xt.append(r[-2])
                yt.append(r[-1])
            it=it+1  
        
        #Hull Calculations
        points=np.array(points) 
        hull=ConvexHull(points)
        xh=points[hull.vertices,0]
        yh=points[hull.vertices,1]   
        area=PolyArea(xh,yh)
        
        #plt.title(str(area))
        #plt.savefig((os.getcwd()+'.png'))
        #plt.clf()
        return xt,yt,area
    
    if args==1:
            a= read('tip_traj.dat')
            plt.clf()         
            for i in np.arange(len(a)):
                plt.plot(a[i][0],a[i][1],'b')
                plt.hold=True
            plt.axis([0,10,-5,5])
            pylab.axes().set_aspect('equal')
            plt.savefig('tip_traj.svg')
     
    if args==2:
            a=read('mtb.dat')
            ds=glv[0]
            vp=glv[1]
            il=glv[2]
            it=0
            while (a[it][0]*vp+il)<7.99:
                ro=a[it][1]
                rn=a[it+1][1]
                vp_set.append(np.sqrt((rn[-1]-ro[-1])**2+(rn[-2]-ro[-2])**2)/0.1)
                it=it+1
            return vp_set
                
def var_plotter():
    a=read('tip_traj.dat')
    varz=[]
    plt.clf()
    for i in np.arange(len(a[0][0])):
        itz=0
        for j in np.arange(len(a)):
           it=a[j][1][i]*a[j][1][i]
           itz=itz+it
        varz.append((itz/(len(a))))
    plt.plot(np.arange(len(varz)),varz,'ko')
    plt.axis([0,10,0,3])
    plt.savefig('var.svg')
    plt.savefig('var.png')
    plt.clf()
    write('var.dat',varz) 

'''def f_mode(glv):
    plt.clf()
    c1=read('mt8.dat')
    c2=read('tip_traj.dat')
    ds=glv[0]
    vp=glv[1]
    il=glv[2]
    del_s=0.5
    pltv=[]
    cl=[c1,c2]
    clr=['bo','ro']
    
    for j in np.arange(len(cl)):
        a=cl[j]
        Fc_g=[]
        for i in np.arange(len(a)):
            r=a[i] 
            x=np.array(r[0])                # size = r/2-1
            y=np.array(r[1])
            dx=np.array(x[1:]-x[:-1])        # r/2 -2
            dy=np.array(y[1:]-y[:-1])
            th=np.arctan2(dy,dx)
            #A=np.zeros(1)
            A=np.cumsum(np.sqrt(dx**2+dy**2))
            #C=np.concatenate((A,B))       #S 
            #D=np.concatenate((A,th))      #theta 
            tck = interpolate.splrep(A, th, s=0)
            S_new  = np.arange(0,8.1,del_s)+del_s/2
            th_new = interpolate.splev(S_new, tck, der=0)
            n=np.arange(th_new.size)
            #L=il+a[197][0]*vp #(vp=0.4,time=a[197][0])
            L=S_new[-1]
            q=n*np.pi/L
            
            #n equations n unknowns --> algebra
            matA=np.zeros([th_new.size,th_new.size])
            for i in np.arange(th_new.size):
                matA[i][:]=np.sqrt(2/L)*np.cos(q*(S_new[i]))
            Fc = np.linalg.solve(matA, th_new)
            Fc_g.append([Fc,q])
            #Numerical approximated coeffs
            for i in q:
                    coeff[i]=np.sqrt(2/L)*
        write('Fmode'+str(j)+'.dat',Fc_g)
        xv=np.zeros(len(Fc_g[0][0]))
        for i in np.arange(len(Fc_g)):
            xv=xv+Fc_g[0][0]**2/500
        plt.loglog(Fc_g[0][1],xv,clr[j])
        plt.axis([0,10,10**-7,10**01])
        plt.hold=True
    plt.savefig('fourier.svg')
    plt.savefig('fourier.png')
'''
def fmode():
    a1=read('mt8.dat')
    a2=read('tip_traj.dat')
    name=['f_mode_mt','f_mode_tips']
    it=-1
    for itz in [a1,a2]:
        it=it+1
        at=itz    
        coeff_g=[]
        for j in np.arange(len(at)):
            print j
            r=at[j]
            x=np.array(r[0])
            y=np.array(r[1])
            dx=x[1:]-x[:-1]
            dy=y[1:]-y[:-1]
            th=np.arctan2(dy,dx)
            dist=np.sqrt(dx**2+dy**2)
            s=np.cumsum(dist)
            s_mid=s-dist/2
            coeff=np.zeros(len(s_mid)-1)
            for i in (np.arange(len(s_mid)-1)+1):
                n=dist*th*np.cos(i*s_mid*np.pi/s[-1])
                coeff[i-1]=np.sqrt(2/s[-1])*np.sum(n)
            coeff_g.append(coeff[1:])
        plt.clf()   
        plt.plot((np.arange(7)+1),coeff_g[0:7],'ro')
        plt.savefig(str(name[it])+'.svg')
        
        return coeff_g

def curvature(glv):
    a=read('mt8.dat')
    cv_g=[]
    avg_cv=np.zeros(len(a[0][0])-2)
    for i in np.arange(len(a)):
        xc=a[i][0]
        yc=a[i][1]
        delx=xc[1:]-xc[:-1]
        dely=yc[1:]-yc[:-1]
        v1x=delx[0:-1]
        v2x=delx[1:]
        v1y=dely[0:-1]
        v2y=dely[1:]
        dp=v1x*v2x+v1y*v2y
        th=np.arccos(dp/(np.sqrt(v1x**2+v1y**2)*np.sqrt(v2x**2+v2y**2)))
        cv=(1-np.cos(th))/(0.1*np.cos(np.pi/2-th/2))
        mod_cv=np.sqrt(cv**2)
        cv_g.append(mod_cv)
        avg_cv=avg_cv+mod_cv/len(a)
    plt.plot((np.arange(len(mod_cv))+1)*glv[0],mod_cv)
    plt.savefig('avg_cv.svg')
    
        
                          