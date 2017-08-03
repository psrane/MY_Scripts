# microtubule simulation module: Utilities
# ToDo: increase tolerance in fsolve
#       solve eigenvalue equation

import mtutil
import numpy as np
import scipy as sp
import scipy.integrate as spint
import scipy.optimize as spopt
import matplotlib.pyplot as plt
import cPickle as pickle
D = 2									        # Dimensions - global variable

def mtinst(N=100, M=0, P=0, BC=[1,1,1,1]):
	"""Create an MT class object to contain the instantaneous state
	Inputs: N the number of nodes (int)
			M the number of motors (int)
			BC a tuple defining the boundary conditions at each end;
			0 = pinned, 1 = free. Node order is 0,1,N-2,N-1
			The class contains these attributes (lower case) plus 
			numpy arrays of coordinates r[N*D], motor coordinates s[M],
			motor anchor a[M*D], MT length l, segment length ds, and time
			Position vectors ordered x0,y0,(z0),x1,y1,(z1),...
	"""
	class MT:
		def __init__(self, N, M, P, BC):
			self.n = N
			self.m = M
			self.l = 0.0
			self.ds = 0.0
			self.cat = 0
			self.time = 0.0
			self.bc = np.array(BC)
			self.r = np.array(np.zeros(N*D))
			self.s = np.array(np.zeros(M))
			self.a = np.array(np.zeros(M*D))
			self.v = np.array(np.zeros(M*D))
			
			self.id = np.array(np.zeros(M))                         # motor index tracker which maintains the motor type as per mdel  -1 -->dynein; 1--> kinesin
			self.p = P                                              # Number of pinning points
			self.pl = np.array(np.zeros(P))                         # array of node_number which are pinned
	mtb = MT(N, M, P, BC)
	return mtb								# return new MT instance


def mtshape(mtb, L=10.0, p=0, A=0.1):
	"""Initialize MT coordinates to a given shape
	   Inputs: mtb (instance of MT class), L, p, A
			L:		MT length
			p > 0:	eigenmode with amplitude A/L
			p = 0:	curved filament with length A*L
			p < 0:	curved filament with curvature A/L
			Nodal positions are offset according to the boundary conditions
			Free:	x0 = 0.5*ds - [1,1], [1,1]
			Hinged:	x0 = 0      - [0,1], [1,0]
			Clamped:x0 = -ds    = [0,0], [0,0]
		2nd order convergence for equilibrium shapes and forces
	"""

	o1 = mtb.bc[0]+2*mtb.bc[1]-2					        # offsets
	o2 = mtb.bc[3]+2*mtb.bc[2]-2
	mtb.l  = L
	mtb.ds = mtb.l/(mtb.n+0.5*(o1+o2-2))					# segment len
	if p > 0:
		if o1+o2-2 == 0:
			sgn = 1
		elif o1+o2-2 == -3:
			sgn = -1
		else:
			print "Mode requires free boundary conditions at right end"
			stop

		if p > 10:
			print "Insufficient precision in solver - use p <= 10"

		ev = spopt.fsolve(eval, (p+sgn*0.5)*np.pi, args=(sgn))
		B = (np.cosh(ev)-sgn*np.cos(ev))/(np.sinh(ev)-sgn*np.sin(ev))
		x = (np.array(range(0,mtb.n))+0.5*o1)*mtb.ds
		evx = ev*x/L
		y = 0.5*A*(np.cosh(evx)+sgn*np.cos(evx) \
		      - B*(np.sinh(evx)+sgn*np.sin(evx)))
		mtb.r[0:D*mtb.n:D]   = x
		mtb.r[1:D*mtb.n+1:D] = y
		if sgn < 0:
			mtb.r[1] = 0.0

	else:
		if p == 0:
			t = spopt.fsolve(flen, np.pi/2, A)/mtb.l		# fixed len
		else:
			t = 0.5*A/mtb.l						# fixed curve

		sn = np.array(range(0,mtb.n)) + 0.5*o1
		if t > 0:
			th = (2*sn*mtb.ds - mtb.l)*t
			ct =  np.cos(th)
			st = -np.sin(th)
			mtb.r[0:D*mtb.n:D]   = ct[:]/(2*t) - 1/(2*t)
			mtb.r[1:D*mtb.n+1:D] = st[:]/(2*t)
		else:
			mtb.r[0:D*mtb.n:D] = sn*mtb.ds
			mtb.r[1:D*mtb.n+1:D] = 0.0

	return mtb


def mtinit(mtb, mdel=[], ms=[], ma=[], mv=[], mi=[], pdel=[], ps=[]):
	"""Create a new MT instance from mt
			Inputs:	mtb current MT instance (MT class)
				mdel list of motors to be deleted (int)
				ms list of motors positions (float)
				ma list of motor anchors (float)
				mv list of anchor velocities (float)
			Initialize new MT:
				Delete unbinding motors - add binding motors
				Adjust number of nodes to filament length
	"""

	o1 = mtb.bc[0]+2*mtb.bc[1]-2						# offsets
	o2 = mtb.bc[3]+2*mtb.bc[2]-2
	N = int(mtb.l/mtb.ds - 0.5*(o1+o2-2) + 0.5)			        # correct node count
	Madd = len(ms)
	Mdel = len(mdel)
	M = mtb.m + Madd - Mdel
        
        Padd = len(ps)
        Pdel = len(pdel)
        P = mtb.p + Padd - Pdel
        
        
	mtbnew = mtinst(N, M, P, mtb.bc)					# new MT instance
	mtbnew.l = mtb.l
	mtbnew.ds = mtb.ds
	mtbnew.bc = mtb.bc
	mtbnew.cat = mtb.cat
	mtbnew.time = mtb.time
	

	ND = N*D								# add/subtract nodes
	nD = mtb.n*D
	if ND <= nD:
		mtbnew.r = mtb.r[0:ND]						# copy first N nodes
	else:
		mtbnew.r[0:nD] = mtb.r					        # copy first n nodes
		t = mtbnew.r[nD-D:nD] - mtbnew.r[nD-2*D:nD-D]
		for i in range(nD, ND, D):
			mtbnew.r[i:i+D] = mtbnew.r[i-D:i] + t		        # add extra nodes

	mdel.append(-1)								# delete motors
	j = mdel.pop(0)
	k = 0
	for i in range(0,mtb.m):
		if i == j:
			j = mdel.pop(0)
			k = k+1
		else:
			iD = i*D
			kD = k*D
			mtbnew.s[i-k]=mtb.s[i]
			mtbnew.id[i-k]=mtb.id[i]
			
			mtbnew.a[iD-kD:iD-kD+D] = mtb.a[iD:iD+D]
			mtbnew.v[iD-kD:iD-kD+D] = mtb.v[iD:iD+D]
        
        pdel.append(-1)
        
        j=pdel.pop(0)
        k=0
        for i in range(0,mtb.p):
		if i == j:
			j = pdel.pop(0)
			k = k+1
		else:
			iD = i*D
			kD = k*D
			mtbnew.pl[i-k]=mtb.pl[i]
			
        pd = mtb.p-Pdel
        pD = P
        mtbnew.pl[pd:pD]=ps[ 0:pD-pd]   

	md = mtb.m-Mdel								# add motors
	MD = M*D
	mD = md*D
	mtbnew.s[md:M] = ms[0:M-md]
	mtbnew.id[md:M] = mi[0:M-md] 
	xm = mtposn(mtbnew)							# motor coordinates
	mtbnew.a[mD:MD] = ma[0:MD-mD] + xm[mD:MD]
	mtbnew.v[mD:MD] = mv[0:MD-mD]
	print mtb.pl

	return mtbnew


def mtplot(mtb, limits=None, fname='', loc='upper left', ext='png', init='yes', hold='no'):
	"""Plot the filament shape using matplotlib
		Takes the object mtb as input
		Limits is a tuple [xmin, xmax, ymin, ymax]
		If a name is given then a file is created (default png)
		Keywords 'init' and hold are for multiple plots - default is single
		The keyword 'init' should be set to yes on the first plot
		The keyword 'hold' should be set to 'no' at the last plot
		Default legend position is upper left
	"""

	SIZE=8

	x = mtb.r[0:D*mtb.n:D]							# nodal coordinates
	y = mtb.r[1:D*mtb.n+1:D]
	xn = mtb.r[D*mtb.n-2]+10.0
	yn = mtb.r[D*mtb.n-1]
	xm = mtposn(mtb)							# motor coordinates
	mx = xm[0:D*mtb.m:D]
	my = xm[1:D*mtb.m+1:D]
	ax = mtb.a[0:D*mtb.m:D]
	ay = mtb.a[1:D*mtb.m+1:D]
	time = "%g" % mtb.time
	label = "time: " + time
	if (init == 'yes'): plt.figure(figsize=(SIZE,SIZE))
	plt.plot(x, y, label=label)
	#plt.plot(mx, my, 'ro')
	#for i in range(0,mtb.m):
		#plt.plot([mx[i],ax[i]], [my[i],ay[i]], 'r')

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


def mtsave(mtb, fname='mtb.dat'):
	"""Save current state of MT
		Default filename is mtb.dat
	"""

	fh = open(fname, 'wb')
	a  = [mtb.r, mtb.s, mtb.a, mtb.time]
	pickle.dump(a, fh)
	file.close(fh)


def mtload(mtb, fname='mtb.dat'):
	"""Restore current state of MT
		Default filename is mtb.dat
	"""

	fh = open(fname, 'rb')
	a  = pickle.load(fh)
	mtb.r = a[0]
	mtb.s = a[1]
	mtb.a = a[2]
	mtb.time = a[3]
	file.close(fh)

	return mtb


def mtposn(mtb):
	"""Return interpolated motor positions
			Inputs:	mtb current MT instance (MT class)
			Outputs:	xm - motor positions
		Output format is indexed to D-dimensional vectors
	"""

	i1, i2, w1, w2 = mtintr(mtb)						# interpolation 
	xm = w1*mtb.r[i1] + w2*mtb.r[i2]

	return xm


def mtintr(mtb):
	"""Return interpolation vectors from motor positions
			Inputs:	mtb current MT instance (MT class)
			Outputs:	i1, i2 - lists of left and right node indexes
						w1, w2 - lists of weights
		Extrapolates if motors outside nodal positions
		Output formats are indexed to D-dimensional vectors
	"""

	sn = mtb.s/mtb.ds - 0.5*(mtb.bc[0]+2*mtb.bc[1]-2)	                # find nodes
	s1 = np.array(sn, dtype=int)
	s1 = np.minimum(np.maximum(s1,0),mtb.n-2)			        # bounds check
	s2 = s1 + 1

	M  = mtb.m								# index and weights
	MD = mtb.m*D
	i1 = np.zeros(MD,dtype=int)
	i2 = np.zeros(MD,dtype=int)
	w1 = np.zeros(MD)
	w2 = np.zeros(MD)
	for i in range(0,D):
		i1[i:MD+i:D] = s1[0:M]*D+i
		i2[i:MD+i:D] = s2[0:M]*D+i
		w1[i:MD+i:D] = s2 - sn
		w2[i:MD+i:D] = sn - s2 + 1

	return i1, i2, w1, w2


def eval(x, sgn):
	"""Eigenvalue equation: cosh(x)cos(x)=sgn
		Inputs: x
			x: eigenvalue
	"""

	return np.cosh(x)*np.cos(x)-sgn


def flen(t, u):
	"""Relate projected length to subtended angle
		Inputs: t, u
			t: half-angle (to solve for)
			u: ratio of segment length to filament curvature (0 < u < 1)
	"""

	return u*t - np.sin(t)


def vdot(x, y, D):
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


