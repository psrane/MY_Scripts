# microtubule simulation module - v1
from __future__ import division
from mt import *
from mtutil import *
import math
import numpy as np
import numpy.random as npr
import os as os
import cPickle as pickle
	
def mtparm(km=1000.0, lm=0.1, vm=0.8,vm_k=0.5, fm=8.0, fm_k=5.0, koff=1.0, mden=4, poff=0, pden=0, mav=0.0):
	"""Parameters for microtubule and motor (dynein)
		Units are picoNewtons, microns, seconds
	"""
	# Check motorvel on changing vm,vm_k & fm,fm_k  --> current expression uses vm>vm_k,fm>fm_k to eliminate loops
	class PARMS:
		def __init__(self, km, lm, vm, fm, koff, mden, poff, pden, mav):
			self.Yb = 25.0					        # bending modulus
			self.Ye = 1.75e4				        # extensional modulus
			self.g  = 10				                # background friction
			self.vp = 0.4					        # polymerization velocity
			self.vd = 0.0					        # depolymerization velocity
			self.kc = 0.0 					        # catastrophe rate
			self.kr = 0.0 					        # recovery rate
			self.km = km					        # motor force constant
			self.lm = lm					        # motor rest length
			self.vm = vm					        # motor velocity(-)
			self.fm = fm					        # motor stall force (-)
			self.fm_k = fm_k                                        # motor stall force (+)
			self.mav  = mav		 			        # motor anchor speed
			self.koff = koff				        # motor off rate
			self.mden = mden				        # motor density
			self.vm_k = vm_k                                        # motor velocity (+)
			self.pden = pden                                        # pinning density (non -motor cytoskeletal cross linkers)
			self.poff = poff                                        # pinning off rate  

	parms = PARMS(km, lm, vm, fm, koff, mden, poff, pden, mav)

	return parms							        # return instance of PARMS class



def motorini(mtb, dt, parms):
	"""Update the motor array for binding and unbinding
	"""

	mdel = []							        # delete motors list
	p = 1 - np.exp(-parms.koff*dt)			                        # unbinding probability
	xr = npr.random(mtb.m)
	

	for i in range(0,mtb.m):				                # unbinding or lost at ends
		if xr[i] < p or \
		mtb.s[i] < 0 or \
		mtb.s[i] > mtb.l: mdel.append(i)	                        # add to delete list
	
	nb = p*parms.mden*mtb.l					                # mean number of motors binding
	Nb = int(npr.poisson(nb,1))
	ms = np.zeros(Nb)						        # Create motors lists
	ma = np.zeros(Nb*D)
	mv = np.zeros(Nb*D)
	ms = npr.random(Nb)*mtb.l
	x=npr.random(Nb)
	mi = 2*(1+x-(parms.mden-1.5)/parms.mden).astype(int)-1                  # assign index which determines the motor type 2*(1+x-fraction of kinesin).astype(int)-1
	th = npr.random(Nb)*2*math.pi
	ma[0:Nb*D:D] = parms.lm*np.cos(th)
	ma[1:Nb*D:D] = parms.lm*np.sin(th)
	th = npr.random(Nb)*2*math.pi
	mv[0:Nb*D:D] = parms.mav*np.cos(th)
	mv[1:Nb*D:D] = parms.mav*np.sin(th)
	
	pdel=[]                                                                 # pinning point delete list
	pp = 1 - np.exp(-parms.poff*dt)	                                        # unbinding probabilty of pinning points  
	pr = npr.random(mtb.p)
	
        for j in range(0,mtb.p):
            if pr[j] < pp or \
            mtb.pl[j] < 0 or \
            mtb.pl[j] > mtb.n: pdel.append(j)
            
        pb = pp*parms.pden*mtb.l
        Pb = int(npr.poisson(pb,1))
        ps = np.zeros(Pb)
        ps = ((npr.random(Pb)*mtb.l)/mtb.ds).astype(int)
	return mdel, ms, ma, mv, mi, pdel, ps


def mtupd(mtb, dt, parms):
	"""Update the MT instance for a time dt
		Uses the scipy interface to LSODE
	"""

	t = np.array([mtb.time,mtb.time+dt])
	y = np.zeros(len(mtb.r)+len(mtb.s))
	y[0:mtb.n*D] = mtb.r
	y[mtb.n*D:mtb.n*D+mtb.m] = mtb.s

	yt = spint.odeint(mtvel, y, t, (mtb,parms), mxstep=1000)

	mtb.r = yt[1][0:mtb.n*D]
	mtb.s = yt[1][mtb.n*D:mtb.n*D+mtb.m]
	mtb.time = mtb.time + dt
	mtb.l = mtb.l + (parms.vp*(1-mtb.cat) \
                  - parms.vd*mtb.cat)*dt					# poly/depoly
	mtb.a = mtb.a + mtb.v*dt					        # add actin flow
	if mtb.l < mtb.ds: mtb.l = mtb.ds					# set min length

	if mtb.cat == 0:
		if npr.random() < parms.kc*dt: mtb.cat = 1			# catastrophe
	else:
		if npr.random() < parms.kr*dt: mtb.cat = 0			# recovery
	return mtb

	
def mtvel(y, t, mtb, parms):
	"""Calculate the segment velocities from MT forces
	"""

	N = mtb.n*D
	M = mtb.m
	mtb.r = y[0:N]
	mtb.s = y[N:N+M]

	f = mtforce(mtb, parms)						        # elastic forces
	fm,fn = motorforce(mtb, parms)				                # motor and nodal forces
	vm = motorvel(mtb, fm, parms)				                # motor velocities
        
# Calculate velocities
	G = parms.g*mtb.ds
	v = np.zeros(len(mtb.r)+len(mtb.s))
	v[0:N] = (f+fn)/G
	v[0:D] = v[0:D]*mtb.bc[0]					        # pinned ends (bc=0) -  v=0
	v[D:D*2] = v[D:D*2]*mtb.bc[1]
	v[N-D*2:N-D] = v[N-D*2:N-D]*mtb.bc[2]
	v[N-D:N]     = v[N-D:N]*mtb.bc[3]
	v[N:N+M] = vm
	
	for i in mtb.pl:
	    v[i*D:(i+1)*D]=v[i*D:(i+1)*D]*0
	
	
	return v


def mtforce(mtb, parms):
	"""Calculate the nodal forces from MT bending and extension
	"""

	N  = mtb.n*D
	D2 = 2*D
	D3 = 3*D
	sc = np.zeros(N-D)						        # scale factors
	b  = np.zeros(N-D)						        # bond vectors
   	b1 = np.zeros(N-D)						        # bond unit vectors
   	rr = np.zeros(N-D2)						        # vector products
   	fm = np.zeros(N-D)						        # bending force (- side)
	fp = np.zeros(N-D)						        # bending force (+ side)
	fb = np.zeros(N)						        # bending force (total)
	fe = np.zeros(N)						        # extensional force

	b  = mtb.r[D:N]-mtb.r[0:N-D]			                        # bond vectors r_{i,i-1}
	sc = np.sqrt(vdot(b, b, D))				                # scale factors
	
# Bending forces
	b1 = b/sc								# unit bond vectors
	rr = vdot(b1[0:N-D2], b1[D:N-D], D)			                # r_{i,i-1}.r_{i+1,i}
	fm[0:N-D2] = b1[D:N-D]  - rr[0:N-D2]*b1[0:N-D2]	                        # left and right forces
	fp[D:N-D]  = b1[0:N-D2] - rr[0:N-D2]*b1[D:N-D]	                        # project out extension
	fb[0:D]  = -fm[0:D]						        # i = 0
	fb[D:D2] =  fm[0:D]- fm[D:D2] - fp[D:D2]			        # i = 1
	fb[D2:N-D2]  = fm[D:N-D3] - fm[D2:N-D2] - fp[D2:N-D2] + fp[D:N-D3]
	fb[N-D2:N-D] = fm[N-D3:N-D2] - fp[N-D2:N-D] + fp[N-D3:N-D2]	        # i = N-1
	fb[N-D:N] = fp[N-D2:N-D]					        # i = N

# Extensional forces
	sc = mtb.ds/sc - 1
	fe[0:D]   = b[0:D]*sc[0:D]						# i = 0
	fe[D:N-D] = b[D:N-D]*sc[D:N-D] - b[0:N-D2]*sc[0:N-D2]
	fe[N-D:N] =-b[N-D2:N-D]*sc[N-D2:N-D]					# i = N

	B  = parms.Yb/(mtb.ds*mtb.ds)
	E  = parms.Ye/mtb.ds
	return fb*B - fe*E

def motorforce(mtb, parms):
	"""Calculate the nodal forces from motors
	"""

	i1, i2, w1, w2 = mtintr(mtb)						# interpolation
	xm = w1*mtb.r[i1] + w2*mtb.r[i2]
	b  = mtb.a - xm								# motor vectors
	bl = np.sqrt(vdot(b, b, D))						# motor lengths
	
	fm = parms.km*(1 - np.minimum((parms.lm/bl),1))*b			# motor forces, np.minimum sorts elements to avoid motor force when motor-linkage is under is compressed
	fn = np.zeros(mtb.n*D)
	fn[i1] = fn[i1] + w1*fm
	fn[i2] = fn[i2] + w2*fm

	rt = mtb.r[i2]-mtb.r[i1]						# tangent vectors
	sc = np.sqrt(vdot(rt, rt, D))
	rt = rt/sc
	fm = vdot(fm, rt, D)
	return fm[0:len(fm):D], fn


def motorvel(mtb, fm, parms):
	"""Calculate the nodal forces from motors
	"""
	vm = np.zeros(len(fm))
	'''
        for i in np.arange(len(mtb.id)):
            if mtb.id[i]==-1:
                vm[i] = -parms.vm*np.maximum(1-fm[i]/parms.fm, 0)		# - directed
            if mtb.id[i]==1:        
 	        vm[i] =  parms.vm_k*np.maximum(1+fm[i]/parms.fm_k, 0)	'''	# + directed
 	vm=np.minimum(mtb.id*parms.vm,mtb.id*parms.vm_k)*np.maximum((1+fm/np.minimum(mtb.id*parms.fm,mtb.id*parms.fm_k)),0)        
        return vm

def mtsave(mtb, fname='mtb.dat'):
	"""Save current state of MT
		Default filename is mtb.dat
	"""

	dummy=0
	a  = [mtb.time, mtb.r]
	fh = open(fname, 'rb')
	m_a = pickle.load(fh)
	m_a.append(a)
	file.close(fh)
	fh = open(fname, 'wb')
	pickle.dump(m_a, fh)
	file.close(fh)
	       

def mtsiml(dir):
	"""User supplied routine to run a simulation
		Inputs:
			nsim = number of simulations
			dt   = time step
	"""

	
	BC= [0,0,1,1]
	L = 0.15
	N = 3
	T = 50
	p = -1
	A = 0.0
	dt= 0.1
	nt = int(T/dt+0.5)
	
	Lmax = 25
	parms = mtparm(km=1000.0, fm=8.0, fm_k=5.0, koff=1.0, mav=0.00)
        
	Mavg = 0
	nsim=10
	for m in np.arange(0,nsim):
	        npr.seed(int((dir)*2000)+m*100)
	        os.mkdir(str(int(dir)*10+m))
                os.chdir(str(int(dir)*10+m))
                m_a=[]
                fh=open('mtb.dat','wb')
	        pickle.dump(m_a,fh)
	        file.close(fh)
	        
		mtb = mtinst(N, 0, 0, BC)
		mtb = mtshape(mtb, L, p, A)
		for n in range(1,nt+1):
		        
                        mtb = mtupd(mtb, dt, parms)
 			mdel, ms, ma, mv, mi, pdel, ps = motorini(mtb, dt, parms)
			mtb = mtinit(mtb, mdel, ms, ma, mv, mi, pdel, ps)
			Mavg = Mavg + mtb.m
			print "time: ", mtb.time, mtb.l, mtb.n, mtb.m
			#for z in mtb.pl:
			#    print mtb.r[z*D:(z+3)*D]
			#fname = 'mtb_' + "%d" %n  
 			#mtplot(mtb, [0,20,-10,10], fname=fname)
			if (mtb.l > Lmax): break
			mtsave(mtb,fname='mtb.dat')

		
	        os.chdir('..')
	return 


if __name__ == "__main__":
	import sys
	mtsiml(int(sys.argv[1]))

