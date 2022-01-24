import numpy as np
from scipy.linalg import circulant

import gidl.utils as gc
from gidl.conv_dl import ConvDL

class ConvInterpDL(ConvDL):
    """
    Stores input dataset and contains helpers for convolutional DL with interpolation (integer shift invariant w interpolation).
    Initializes a learned dictionary *if not provided*.
    Inherits from ConvDL with main changes to coding step.

    Args:
        input_data (`numpy.ndarray`): input dataset
        nDivs (`int`): number of subdivisions within a single integer shift
        learned_dict (`numpy.ndarray`, optional):
            initialize learned dictionary for the dataset, if available (default: None)
        reg_param (`float`, optional): 
            regularization parameter, lambda. 
            suggested to be set as large as possible while ensuring learned_dict is not degenerate (default: 0.1)
        num_loops (`int`, optional): 
            number of inner loops to attempt for solving convex program during the coding step (default: 5)
        num_generators (`int`, optional): 
            q number of generators for initializing dictionary, *if learned_dict is not provided* (default: 1)
    """
    def __init__(
        self, 
        input_data, 
        nDivs, 
        learned_dict=None, 
        reg_param=0.1, 
        num_loops=5, 
        num_generators=1, 
    ):

        self.nDivs = nDivs
        super().__init__(input_data, learned_dict, reg_param, num_loops, num_generators)

        self.classification = 'convinterp'+str(nDivs)  # prepends output filenames

        self.d_half = int((self.d-1)/2)
        self.DFT = gc.DFT(self.d_half)  # discrete fourier transform matrix

    def coding_step(self):
        """
        Task: using current self.learned_dict, update (i) self.X, (ii) self.apx_err, (iii) self.reg_val
            Compute and update the sparse vector X for the current dictionary (fix dictionary, update X)
            by solving a convex program.
            In addition, update the approximation error and regularization value
        """
        # Re-initialize sparse vector X (as well as error vector and normsize vector)
        X = np.zeros((self.d, self.d, self.num_generators, self.n))
        err_vec = np.zeros((self.n,))
        normsize_vec = np.zeros((self.n,))

        # Fix dictionary, update sparse vector X (as well as error vector and normsize vector)
        ## Create the full matrix
        ## Embed dictionary into a larger d by d*q matrix (q is the number of generators)
        A_full = np.zeros((self.d,self.d*self.num_generators*self.nDivs))
        for i in range(self.num_generators):
            d_elem = self.learned_dict[:,i]
            for j in range(self.d*self.nDivs):
                A_full[:,i*self.d*self.nDivs + j] = np.real(
                    self.DFT.T.conjugate() @ _interp_roll(self.DFT @ d_elem,j/(self.d*self.nDivs))
                )

        ## Then perform the solve
        X_long = np.zeros((self.d*self.num_generators*self.nDivs,self.n))
        for jj in range(self.n):
            X_long[:,jj],err_vec[jj,],normsize_vec[jj,] = sv_lasso(
                self.input_data[:,jj],A_full,self.reg_param,self.num_loops
            )

        ## Reshape the solution into a matrix
        for j in range(self.d*self.nDivs):
            v = np.zeros((self.d,)).astype(complex)
            for i in range(self.d):
                arg = 2 * np.pi * (j/(self.d*self.nDivs)) * (i-self.d_half)
                v[i,] = np.cos(arg) + 1j * np.sin(arg)
            V = np.diag(v)
            T = np.real( self.DFT.T.conjugate() @ (V @ self.DFT) ) # The translation matrix
            
            for i in range(self.num_generators):
                for k in range(self.n):
                    X[:,:,i,k] += T * X_long[i*self.d*self.nDivs + j,k]
        self.X = X

        # Get approximation error and regularization value
        self.apx_err = 0.5 * np.sum(err_vec)/self.n
        self.reg_val = self.reg_param * np.sum(normsize_vec)/self.n


def _interp_roll(vec,amt):
    # Convex combination interpolation
    # Weights are reversed: 0 means start -> full weight
    d = vec.shape[0]
    d_half = int((d-1)/2)
    
    v = np.zeros((d,)).astype(complex)
    for i in range(d):
        arg = 2 * np.pi * amt * (i-d_half)
        v[i,] = (np.cos(arg) + 1j * np.sin(arg)) * vec[i,]
    
    return v


def sv_lasso(data,D,l,nLoops):
    """
    Solve the convex program - helper function for coding step
    (For convolutional dictionary learning with interpolation)
    """
    d,q = D.shape
    
    # Initialize at the origin
    x = np.zeros((q,))
    
    # Outer Loop iterations
    for i in range(nLoops):
        # Compute the gradient
        rem = np.dot(D,x)-data
        g = np.dot((D.T) , rem)
        ag = np.dot(D,g)
        nn = np.linalg.norm(ag)**2
        eta = 1.5*np.dot(ag,rem) / (nn)
        
        # Take a gradient step
        x = x - (g * eta)
        
        # Proximal Map
        x = gc.softthreshold(x,l)
        
    err = np.linalg.norm(rem)**2
    normsize = np.sum(np.sum(np.abs(x)))
    return x,err,normsize