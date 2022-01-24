import numpy as np
from scipy.linalg import toeplitz
from math import pi,sin,cos,sqrt

import gidl.utils as gc
from gidl.regular_dl import RegularDL

class CtsShiftDL(RegularDL):
    """
    Stores input dataset and contains helpers for continuous shift invariance.
    Initializes a learned dictionary *if not provided*.
    Inherits from RegularDL with main changes to coding step and dictionary update step.

    Args:
        input_data (`numpy.ndarray`): input dataset
        learned_dict (`numpy.ndarray`, optional):
            initialize learned dictionary for the dataset, if available (default: None)
        reg_param (`float`, optional): 
            regularization parameter, lambda. 
            suggested to be set as large as possible while ensuring learned_dict is not degenerate (default: 0.05)
        num_loops (`int`, optional): 
            number of inner loops to attempt for solving convex program during the coding step (default: 5)
        num_generators (`int`, optional): 
            q number of generators for initializing dictionary, *if learned_dict is not provided* (default: 1)
        three_cycle (`bool`, optional):
            modifies number of inner-inner loops for projector onto the set of PSD Toeplitz matrices during coding step
            see regCtsShift helper function for details (default: True)
    """
    def __init__(
        self, 
        input_data, 
        learned_dict=None, 
        reg_param=0.2, 
        num_loops=5, 
        num_generators=1, 
        three_cycle=True
    ):
        super().__init__(input_data, learned_dict, reg_param, num_loops, num_generators)

        self.d = int((self.datum_dim-1)/2)
        self.dft = gc.DFT(self.d)

        self.input_data = self.dft @ self.input_data  # FT

        self.suc_dict_type = 'cts'
        self.classification = 'ctsshift'

        self.three_cycle = three_cycle

    def coding_step(self):
        """
        Task: using current self.learned_dict, update (i) self.X, (ii) self.apx_err, (iii) self.reg_val
            Compute and update the sparse vector X for the current dictionary (fix dictionary, update X)
            by solving a convex program.
            In addition, update the approximation error and regularization value
        *Note: self.X here is in the Fourier transform domain*
        """
        # Re-initialize *Fourier transformed* sparse vector X_ft (as well as error vector and normsize vector)
        X_ft = np.zeros((self.d+1,self.num_generators,self.n)).astype(complex)
        err_vec = np.zeros((self.n,))
        normsize_vec = np.zeros((self.n,))

        # Initialize *FT* dictionary
        learned_dict_ft = self.dft @ self.learned_dict

        # Fix dictionary, update *FT* sparse vector X_ft (as well as error vector and normsize vector)
        for jj in range(self.n):
            X_ft[:,:,jj],err_vec[jj,],normsize_vec[jj,] = regCtsShift(
                self.input_data[:,jj],learned_dict_ft,self.reg_param,self.num_loops,self.three_cycle
            )
        self.X = X_ft
        
        # Get approximation error and regularization value
        self.apx_err = 0.5 * np.sum(err_vec) / self.n
        self.reg_val = self.reg_param * np.sum(normsize_vec) / self.n

    def dict_update_step(self):
        """
        Task: using current self.X, update self.learned_dict
            Compute and update the learned dictionary for the current sparse vector X (fix X, update dictionary)
            by solving a least squares problem.
        """
        X = self.X
        xx = np.zeros(((self.d+1)*self.num_generators,(self.d+1)*self.num_generators)).astype(complex)
        for i in range(self.num_generators):
            for k in range(self.num_generators):
                xx_s = np.zeros((self.d+1,self.d+1)).astype(complex)
                for jj in range(self.n):
                    xx_s = xx_s + np.diag( np.multiply(np.conjugate(X[:,k,jj]),X[:,i,jj]) )
                xx[k*(self.d+1):(k+1)*(self.d+1),i*(self.d+1):(i+1)*(self.d+1)] = xx_s
                
        xy = np.zeros(((self.d+1)*self.num_generators,)).astype(complex)
        for k in range(self.num_generators):
            xy_s = np.zeros((self.d+1,)).astype(complex)
            for jj in range(self.n):
                xy_s = xy_s + np.multiply(np.conjugate(X[:,k,jj]),self.input_data[self.d:,jj])
            xy[k*(self.d+1):(k+1)*(self.d+1)] = xy_s
            
        bb = _linsyssolve(xx,xy,self.num_generators,self.datum_dim)
        
        A_ft_int = np.reshape(bb,(self.num_generators,self.d+1)).T
        A_ft_full = np.zeros((self.datum_dim,self.num_generators)).astype(complex)
        A_ft_full[self.d:,:] = A_ft_int
        A_ft_full[:self.d,:] = np.flipud(A_ft_int[1:,:]).conjugate()

        self.learned_dict = np.dot(self.dft.T.conjugate(),A_ft_full).real


def regCtsShift(data,A,reg_param,nIterates,THREECYCLE,stepsize=1.0):
    """
    Regularized Least Squares (but in the Fourier Basis)
    Solve the convex program - helper function for coding step
    (For continuous shift dictionary learning)

    For THREECYCLE, set False for ECG data (although can set to True too)
    """
        
    if THREECYCLE:
        nInnerLoops = 1
    else:
        nInnerLoops = 5
    
    dd,q = A.shape
    d = int((dd-1)/2)
    
    Xp  = np.zeros((d+1,q)).astype(complex)
    Xm  = np.zeros((d+1,q)).astype(complex)
    
    
    for ii in range(nIterates):
        # Take gradient step with respect to the Loss Function
        AXp  = np.multiply(A[d:,:],Xp) # Hadamard product
        AXm  = np.multiply(A[d:,:],Xm) # Hadamard product
        diff = np.sum(AXp-AXm,axis=1) - data[d:,]
        
        for jj in range(q):
            Xp[:,jj] -= stepsize * np.multiply(A[d:,jj].conjugate(),diff)
            Xm[:,jj] += stepsize * np.multiply(A[d:,jj].conjugate(),diff)
        
        # Take gradient step with respect to the Regularization
        Xp[0,:] -= stepsize * reg_param * np.ones((q,))
        Xm[0,:] -= stepsize * reg_param * np.ones((q,))
        
        # Project onto Toep PSD
        for jj in range(q):
            Xp[:,jj] = gc._projToepPSDvec(Xp[:,jj],nIterates=nInnerLoops)
            Xm[:,jj] = gc._projToepPSDvec(Xm[:,jj],nIterates=nInnerLoops)
            
    apx_err = np.linalg.norm(diff)**2
    reg_size = np.real(np.sum(Xp[0,:])+np.sum(Xm[0,:]))
    return Xp-Xm,apx_err,reg_size


def _linsyssolve(A,b,q,dd):
    """
    For solving the least squares problem - helper function for dictionary update step
    """
    d = int((dd-1)/2)
    b_REAL = np.zeros((q*dd,))
    A_REAL = np.zeros((q*dd,q*dd))
    
    # Map the elements of b
    for i in range(q):
        b_orig_seg = b[i*(d+1):(i+1)*(d+1)]
        b_map_seg = np.zeros((dd,))
        b_map_seg[0,] = np.real(b_orig_seg[0,],)
        b_map_seg[1:d+1,] = np.real(b_orig_seg[1:,])
        b_map_seg[d+1:,] = np.imag(b_orig_seg[1:,])
        b_REAL[i*dd:(i+1)*dd] = b_map_seg
        
    # Map the elements of A
    for i in range(q):
        for j in range(q):
            A_REAL_seg = np.zeros((dd,dd))
            A_orig_seg = A[i*(d+1):(i+1)*(d+1),i*(d+1):(i+1)*(d+1)]
            A_REAL_seg[0,0] = np.real(A_orig_seg[0,0])
            for k in range(d):
                A_REAL_seg[1+k,1+k] = np.real(A_orig_seg[1+k,1+k])
                A_REAL_seg[1+k+d,1+k] = np.imag(A_orig_seg[1+k,1+k])
                A_REAL_seg[1+k,1+k+d] = -np.imag(A_orig_seg[1+k,1+k]) # NEGATIVE!! Because of complex multiplication!!
                A_REAL_seg[1+k+d,1+k+d] = np.real(A_orig_seg[1+k,1+k])
            A_REAL[i*dd:(i+1)*dd,i*dd:(i+1)*dd] = A_REAL_seg
    
    # Solve the linear system
    x_REAL = np.linalg.lstsq(A_REAL,b_REAL,rcond=None)[0]
    
    # Map back
    x_out = np.zeros((q*(d+1),)).astype(complex)
    for i in range(q):
        x_out_seg = np.zeros((d+1,)).astype(complex)
        x_REAL_seg = x_REAL[i*dd:(i+1)*dd]
        x_out_seg[0,] = x_REAL_seg[0,]
        x_out_seg[1:,] = x_REAL_seg[1:d+1,] + 1j * x_REAL_seg[d+1:,]
        x_out[i*(d+1):(i+1)*(d+1)] = x_out_seg
    
    return x_out