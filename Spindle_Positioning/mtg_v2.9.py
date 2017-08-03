# Primary simulations for Rigid  Mitotic Spindle Centering

import numpy as np
import scipy as sp
import scipy.integrate as spint
import numpy.random as npr
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import pickle
NMT=100
nmt=NMT
D=2
C_r=10

def spparms(C_r,L_arc=10.0,L_sp=4.0):                     # Instances of parameter class for bound and unbound state of MT                
    class SP:
        def __init__(self,C_r,L_arc,L_sp,Vp,Vd,Kc,Kr):
            self.c_r   = C_r				  # Radius of the Circular Cell
            self.l_arc = L_arc			          # Microtubules originating from the spindle are distributed in an arc lenght of cell of both sides
            self.l_sp  = L_sp                             # Lenght of Mitotic spindle
            self.g     = 0.1                              # Background friction 
            self.vp    = Vp				  # Polymerization Velocity of microtubule
            self.vd    = Vd				  # Depolymerization velocity of microtubule
            self.kc    = Kc                               # Rate of Catastrophe
            self.kr    = Kr                               # Rate of Recovery
            self.fm    = 8.0				  # Force on microtubule which is applied when microtubule is captured
            self.koff  = 1.0                              # Off Rate, rate at which captured microtubules lets the microtubule go
            self.mden  = 1.0                              # Motor Density
    parms=[0]*2
    parms[0] = SP(C_r,L_arc,L_sp,Vp=0.1,Vd=0.3,Kc=0.01,Kr=0.04)    
    parms[1] = SP(C_r,L_arc,L_sp,Vp=0,Vd=0,Kc=0.01,Kr=0.01)
    return parms
		
def mt_sp_inst(nmt,parms,N=2,M=0,L=3.0):                      # Initiate list of instances if class MT in spindle
    class MT:
        def __init__(self,N,M,L):
            self.n = N										  # Number of nodes
            self.m = M										  # Number of motors
            self.l = L                                        # Length of microtubule
            self.cat = 0.0                                    # Flag for determining the length of microtubule depending on polymerization/depolymerization --> refer line 187 to see the use
            self.time = 0.0                                   # Time of microtubule (Adjustment for to observe avg. lifetime of microtubule in simulations)
            self.r = np.zeros(N*D)                            # Nodes which make the filament (Astral Microtubules are nearly straight in real cells during mitosis,hence number of nodes are limited to 2)
            self.at = 0   									  # (0:right,1:left,aster_assigner before entering the loop)
            self.uv = np.zeros(D)    						  # (unit vector)
            self.st = 0                                       # (state of microtubule : 0 -> unbound, 1 -> bound)  
            self.l_cut=1                                      # Microtubule do not shrink beyond lcut
	    self.i_uv= np.zeros(D)
    mtg = [0]*(2*nmt+1)                                       # Last entry in list referes to aster co-ordinates
    for i in np.arange(2*nmt):
        mtg[i] = MT(N,M,L)
    mtg[2*nmt]=np.array([parms[0].l_sp/2,0,-1*parms[0].l_sp/2,0])
    mtg=aster_assigner(mtg)
    return mtg
	
def mt_sp_shape(mtg,parms):                                 # Initalize the simulation by providing initial configuration
    mtg       = aster_assigner(mtg)                         # Spindle has 2 asters (bipolar), this routine determines the aster to which microtubule belongs microtubule
    l_ex      = parms[0].c_r-parms[0].l_sp/2.0              # Used for determination of angle subtended at poles by specified l_arc
    t_angle   = parms[0].l_arc/l_ex                         # angle subtended at poles
    i_angle_r = npr.random(nmt)*t_angle-t_angle/2.0         # Determining the direction of microtubules for initialization --> Randomly Assigned
    i_angle_l = npr.random(nmt)*t_angle-t_angle/2 +np.pi    # Determining the direction of microtubules for initialization 
    i_angle   = np.zeros(2*nmt)
    for i in np.arange(nmt):
        i_angle[i]=i_angle_r[i]
        i_angle[nmt+i]=i_angle_l[i]
    mtg[2*nmt]= np.array([parms[0].l_sp/2+3,0,(-1)*parms[0].l_sp/2+3,0])
    for i in np.arange(2*nmt):
        mtg[i].i_uv[0] = np.cos(i_angle[i])
	mtg[i].i_uv[1] = np.sin(i_angle[i])
        mtg[i].r[0:D] = mtg[2*nmt][mtg[i].at*D:mtg[i].at*D+D]                   # First node of micortuble is same as aster
        mtg[i].r[D]   = mtg[i].l*np.cos(i_angle[i])+mtg[2*nmt][mtg[i].at*D]     # Last node x-co-ordinate is determined using assigned angle
        mtg[i].r[D+1]   = mtg[i].l*np.sin(i_angle[i])+mtg[2*nmt][mtg[i].at*D+1] # Last node y-co-ordinate is determined using assigned angle
    return mtg

def tip_chkr(mtg,parms):                                   # ChecK the postion of the tip to flag it for gripping or to avoid extension outside the cell
    """Objective : To evaluate if the microtbules are at capturing distance from cell wall.
    I/p : mtg,c_r
    O/p : 
    Associated Variables
    cde  -> capture distance evaluater
    ccm  -> conditional capture matrix
    """
    cde = np.array(np.zeros(2*nmt))  # Evaluate distance from boundary to determine whether microtubule can be captured or not
    ccm = np.array(np.zeros(2*nmt))  # Flag for capturing event
    for i in np.arange(2*nmt):
        N = mtg[i].n
        cde[i] = (parms[1].c_r - np.sqrt((mtg[i].r[N*D-2])**2+(mtg[i].r[N*D-1])**2))
        if cde[i]<0:
            while cde[i]<0:
                mtg[i].r[D:2*D]=mtg[i].r[D:2*D]*0.99   # Adjusting length to avoid crossover over the cell boundary
                cde[i] = (parms[1].c_r - np.sqrt((mtg[i].r[N*D-2])**2+(mtg[i].r[N*D-1])**2))
        if cde[i]<0.11:
            if mtg[i].cat==0:
                ccm[i]=1
            else:
                ccm[i]=0
        else: ccm[i]=0
        mtg[i].st=ccm[i]
    return mtg
              	
def force_calc(mtg,parms):                            # Force exerted on the spindle
    fr=np.array(np.zeros(D))
    fl=np.array(np.zeros(D))
    F=np.array(np.zeros(2*D))
    for i in np.arange(nmt):
       st=int(mtg[i].st)
       N=mtg[i].n
       dvr=np.array(np.zeros(D))
       dvr=mtg[i].r[D:N*D]-mtg[i].r[0:D]
       duvr=dvr/np.sqrt(vdot(dvr,dvr,D)) #      fR=mtg[i].m*parms[st].fm*duvr
       fR = mtg[i].l*4*duvr
       fr=fr+fR
       st=int(mtg[nmt+i].st)
       dvl=np.array(np.zeros(D))
       dvl=mtg[nmt+i].r[D:N*D]-mtg[nmt+i].r[0:D]
       duvl=dvl/np.sqrt(vdot(dvl,dvl,D)) #      fR=mtg[i].m*parms[st].fm*duvl
       fL=mtg[nmt+i].l*4*duvl
       fl=fl+fL
    F[0:D]=fr[0:D]
    F[D:2*D]=fl[0:D]
    return F

def r_at_upd(dt,F,tl,mtg,parms):                     # Motion of aster in response to the force
    u_r      = np.zeros(D)                           # Velocity of right aster    
    u_l      = np.zeros(D)                           # Velocity of Left aster
    x_r      = np.zeros(D)                           # Position of Right aster
    x_l      = np.zeros(D)                           # Position of Left aster
    u_r[0:D] = F[0:D]/(tl[0]*parms[0].g)             # Half spindle friction on each aster
    u_l[0:D] = F[D:2*D]/(tl[1]*parms[0].g)
    x_r[0:D] = mtg[2*nmt][0:D]
    x_l[0:D] = mtg[2*nmt][D:2*D]
    del_u    = u_r-u_l
    del_x    = x_r-x_l
    A        = vdot(del_x,del_x,D)
    B        = vdot(del_u,del_u,D)
    C        = vdot(del_u,del_x,D)
    a        = 4*A[0]
    b        = -4*(A[0]+C[0]*dt)
    c        = 2*C[0]*dt+B[0]*(dt**2)
    coeff    = np.array([c,b,a])
    roots    = poly.polyroots(coeff)
    lm       = np.min(roots)
    mtg[2*nmt][0:D]=x_r+u_r*dt-lm*(x_r-x_l)
    mtg[2*nmt][D:2*D]=x_l+u_l*dt+lm*(x_r-x_l)
    mtg=aster_assigner(mtg)
    return mtg
	
def mtg_len_upd(dt,mtg,parms):                      # Determine new length on based on the state and the growth dynamics of MT
   for i in np.arange(2*nmt):
        st = int(mtg[i].st)
        mtg[i].l = mtg[i].l+(parms[st].vp*(1-mtg[i].cat) \
                                        - parms[st].vd*mtg[i].cat)*dt 

        if mtg[i].cat==0:
            if npr.random()<parms[st].kc*dt:
                mtg[i].cat=1
                if mtg[i].st==1:
                    mtg[i].st=0
        else:
            if npr.random()<parms[st].kr*dt:mtg[i].cat=0
   return mtg

def variable_calc(z,Time,time,Fr,Fl,Xas,th,nt,F,mtg):
    Fr[z-1]=np.sqrt(F[0]**2+F[1]**2)
    Fl[z-1]=np.sqrt(F[2]**2+F[3]**2)
    Xas[z-1]=np.sqrt(((mtg[2*nmt][0]+mtg[2*nmt][D])/2)**2+((mtg[2*nmt][1]+mtg[2*nmt][D+1])/2)**2)        #Distance moved by centre of aster
    thn=(mtg[2*nmt][1]-mtg[2*nmt][D+1])
    thd=(mtg[2*nmt][0]-mtg[2*nmt][D])
    th[z-1]=(180/np.pi)*np.arctan2(thn,thd)
    time[z-1] =Time
    return Fr,Fl,Xas,th,time

def r_mtg_upd(mtg):                               # Determine new co-ordinates based on new aster position and new length 
     for i in np.arange(2*nmt):
        mtg[i].r[0:D]   = mtg[2*nmt][mtg[i].at*D:mtg[i].at*D+D]
        st=mtg[i].st
        if st==0:
           mtg[i].r[D:mtg[i].n*D] = mtg[i].l*mtg[i].uv+mtg[2*nmt][mtg[i].at*D:mtg[i].at*D+D]
        else:
            mtg[i].r[D:mtg[i].n*D]= mtg[i].r[D:mtg[i].n*D]
            mtg[i].l=np.sqrt((mtg[i].r[3]-mtg[i].r[1])**2+(mtg[i].r[2]-mtg[i].r[0])**2)
        if mtg[i].l<mtg[i].l_cut:
            mtg[i].cat=0
            mtg[i].st=0
	    mtg[i].r[D:mtg[i].n*D]= mtg[i].i_uv+ mtg[i].r[0:D]
     return mtg  
           
def mt_sp_plot(Time,mtg, parms, limits=None, fname='', loc='upper left', ext='png', init='yes', hold='no'):                   # Plotting Routine
	"""     Plot the mitotic spindle shape using matplotlib
		Takes the object mtb as input
		Limits is a tuple [xmin, xmax, ymin, ymax]
		If a name is given then a file is created (default png)
		Keywords 'init' and hold are for multiple plots - default is single
		The keyword 'init' should be set to yes on the first plot
		The keyword 'hold' should be set to 'no' at the last plot
		Default legend position is upper left
	"""

	SIZE=20
        time = "%g" % Time
	label = "time: " + time
	if (init == 'yes'): plt.figure(figsize=(SIZE,SIZE))
        plott=[0]*(2*nmt)
        for i in np.arange(2*nmt):
            N=mtg[i].n
            plott[i]=[mtg[i].r[0:N*D:D],mtg[i].r[1:N*D:D]]
            plt.plot(plott[i][0],plott[i][1],'.')
            plt.plot(plott[i][0],plott[i][1])
            plt.hold=True
        plt.Circle((0,0),radius=parms[0].c_r)
        a=np.arange(1000)*2*np.pi/1000
        x1=parms[0].c_r*np.cos(a)
        y1=parms[0].c_r*np.sin(a)
        plt.plot(x1,y1,'b',label=label)
        spx=np.array([mtg[2*nmt][D],mtg[2*nmt][0]])
        spy=np.array([mtg[2*nmt][D+1],mtg[2*nmt][1]])
        u_th=np.linspace(0,2*np.pi,1000)
        th0=np.arctan2((mtg[2*nmt][3]-mtg[2*nmt][1]),(mtg[2*nmt][2]-mtg[2*nmt][0]))
        rmax=parms[0].l_sp/2-0.1
        Rmin=np.arange(7)*rmax/10
        for i in np.arange(7):               # Spindle Plotting
          rmin=Rmin[i]
          xe=rmax*np.cos(u_th)
          ye=rmin*np.sin(u_th)
          new_u=np.arctan2(ye,xe)
          newth=new_u+th0
          l=np.sqrt((xe)**2+(ye)**2)
          alpha=[0]*2
          alpha[0]=np.zeros(1000)
          alpha[1]=np.zeros(1000)
          as_uv=(mtg[2*nmt][0:D]-mtg[2*nmt][D:2*D])/parms[0].l_sp
          alpha[0][250:750]=as_uv[0]*0.1
          alpha[0][0:250]=-as_uv[0]*0.1
          alpha[0][750:1000]=-as_uv[0]*0.1
          alpha[1][250:750]=as_uv[1]*0.1
          alpha[1][0:250]=-as_uv[1]*0.1
          alpha[1][750:1000]=-as_uv[1]*0.1
          xep=l*np.cos(newth)+0.5*(mtg[2*nmt][2]+mtg[2*nmt][0])+alpha[0]
          yep=l*np.sin(newth)+0.5*(mtg[2*nmt][3]+mtg[2*nmt][1])+alpha[1]
          xset1=xep[0:250]
          yset1=yep[0:250]
          xset2=xep[250:750]
          yset2=yep[250:750]
          xset3=xep[750:1000]
          yset3=yep[750:1000]
          plt.plot(xset1,yset1,'g')
          plt.plot(xset2,yset2,'g')
          plt.plot(xset3,yset3,'g')
          plt.hold=True
        
        as_uv_n=np.zeros(2)
        as_uv_n[0]=as_uv[1]
        as_uv_n[1]=-as_uv[0]
        c_e1=as_uv_n*parms[0].l_sp/2
        c_e2=-1*as_uv_n*parms[0].l_sp/2
        c_e1[0]=c_e1[0]+0.5*(mtg[2*nmt][2]+mtg[2*nmt][0])-0.01*as_uv[0]
        c_e2[0]=c_e2[0]+0.5*(mtg[2*nmt][2]+mtg[2*nmt][0])-0.01*as_uv[0]
        c_e1[1]=c_e1[1]+0.5*(mtg[2*nmt][3]+mtg[2*nmt][1])-0.01*as_uv[1]
        c_e2[1]=c_e2[1]+0.5*(mtg[2*nmt][3]+mtg[2*nmt][1])-0.01*as_uv[1]

          
        cex=np.array([c_e1[0],c_e2[0]])
        cey=np.array([c_e1[1],c_e2[1]])
        plt.plot(cex,cey,'b',linewidth=1)
        plt.plot(spx,spy)
        plt.title('2D simulations of Mitotic Spindle Centering')
        if (hold == 'yes'):
		plt.hold=True
	else:
		if (limits != None):
			plt.axis(limits)

		plt.legend(loc=loc)
		plt.hold=False
		if len(fname) == 0:
			plt.show()
		else:
 			fname = fname + '.' + ext
			plt.savefig(fname, format=ext)
			plt.close()
#start debugging here
def data_plot(time,Fr,Fl,Xas,th):
    ext='png'
    plt.plot(time,Fr,'g')
    name='Fr'+'.'+ext
    plt.xlabel('time (sec)')
    plt.ylabel('Fr in pN')
    plt.title('Force on the right aster')
    plt.grid(True)
    plt.savefig(name,format=ext)
    plt.close()

    plt.plot(time,Fl,'g')
    name='Fl'+'.'+ext
    plt.title
    plt.xlabel('time (sec)')
    plt.ylabel('Fl in pN')
    plt.title('Force on the left aster')
    plt.grid(True)
    plt.savefig(name,format=ext)
    plt.close()

    plt.plot(time,Xas,'g')
    name='Xas'+'.'+ext
    plt.xlabel('time (sec)')
    plt.ylabel('Distance (mcirometers)')
    plt.title('Distance of centre of aster original position')
    plt.grid(True)
    plt.savefig(name,format=ext)
    plt.close()

    plt.plot(time,th,'g')
    name='th'+'.'+ext
    plt.xlabel('time (sec)')
    plt.ylabel('Angle (degrees)')
    plt.title('Rotation of centre of aster from original configuration')
    plt.grid(True)
    plt.savefig(name,format=ext)
    plt.close()

    data=[Fr, Fl, Xas, th]
    fob=open('data.dat','wb')
    pickle.dump(data,fob)
    file.close(fob)
#import pdb
def mt_uv_upd(mtg):
    for i in np.arange(2*nmt):
        t_old=np.zeros(D)
        t_new=np.zeros(D)
        t=np.zeros(D)
        t_old[0:D]=(mtg[i].r[D:2*D]-mtg[i].r[0:D])
        t_old[0:D]=t_old/np.sqrt(vdot(t_old,t_old,D))
        t_new[0:D]=(mtg[i].r[D:2*D]-mtg[2*nmt][mtg[i].at*D:mtg[i].at*D+D])
        t_new[0:D]=t_new/np.sqrt(vdot(t_new,t_new,D))
        w=np.e**(-(mtg[i].l-mtg[i].l_cut))
        t=w*t_old+(1-w)*t_new
        mtg[i].uv[0:D]=t/(vdot(t,t,D))          
    return mtg

def vdot(x,y,D):
        """Computes the dot products of two vectors stored in a single array
		Inputs: x, y, D
			x, y: input vectors of length n*D - n D-dimensional vectors
			D: dimension
		Returns dot product of each pair of D-vectors, repeated D times
	"""

	ND = len(x)
	if ND != len(y):
		print "ERROR: x and y must be the same length"
		return

	z  = x*y
	for i in range(1,D):
		z[0:ND:D] = z[0:ND:D] + z[i:ND+i:D]

	for i in range(1,D):
		z[i:ND:D] = z[0:ND:D]

	return z

def aster_assigner(mtg):                                           # Assign the aster to the microtubule
    for i in np.arange(2*nmt): 
        if i<nmt:
            mtg[i].at=0              #0 for right
        else: 
            mtg[i].at=1              #1 for left
    return mtg

def l_sum(mtg,parms):                                             # Summation of the length for calculation of frictional resisitance to the motion 
    th=(np.pi/2)*np.arange(nmt)/nmt-np.pi/4                       
    l =parms[0].l_sp/(2*np.cos(th))
    tl=np.zeros(2)
    nkmt_fr = 0.95                                                # non kinetochore microtubule consist of 95% of spindle microtubules                                          
    for i in np.arange(nmt):
	b=mtg[i].l
	c=mtg[nmt+i].l
        a=l[i]
        tl[0]=tl[0]+a+b+a*nkmt_fr/(1-nkmt_fr)
        tl[1]=tl[1]+a+c+a*nkmt_fr/(1-nkmt_fr)
    return tl

def spsiml(nsim,dt):                                             # This is where magic happens !!!
    npr.seed(1000)
    nmt=NMT
    T=2400
    Time=0
    nt= int(T/dt)
    Fl=np.zeros(nt)
    Fr=np.zeros(nt)
    Xas=np.zeros(nt)
    th=np.zeros(nt)
    time=np.zeros(nt)
    loop=0
    parms =  spparms(C_r,L_arc=C_r*0.7*np.pi,L_sp=0.8*C_r)
    mtg   =  mt_sp_inst(nmt,parms,N=2,M=0,L=3.0)
    mtg   =  mt_sp_shape(mtg,parms)
    for z in range(1,nt+1):
        Time=Time+dt
        mtg=tip_chkr(mtg,parms)
        tl=l_sum(mtg,parms)
        F  =force_calc(mtg,parms) 
        mtg=r_at_upd(dt,F,tl,mtg,parms)  #aster_assigner is called here
        mtg=mt_uv_upd(mtg) 
        mtg=mtg_len_upd(dt,mtg,parms) 
        mtg=r_mtg_upd(mtg)                      
#       pdb.set_trace()  
        Fr,Fl,Xr,Xl,time=variable_calc(z,Time,time,Fr,Fl,Xas,th,nt,F,mtg)        # unit vector followed by microtubule
        if (z%5)==0:
            loop=loop+1
            fname = 'mtb_' + "%d" %loop                    # loop construct for printing every 5th step
            mt_sp_plot(Time,mtg,parms, [-10,10,-10 ,10], fname=fname)
        print F 
    data_plot(time,Fr,Fl,Xr,Xl)
        
if __name__=="__main__":
    import sys
    spsiml(int(sys.argv[1]),float(sys.argv[2]))
#tips.force.aster.uv.length.microtubules

