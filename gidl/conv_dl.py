import numpy as np
from scipy.linalg import circulant

import gidl.utils as gc
from gidl.regular_dl import RegularDL

class ConvDL(RegularDL):
    """
    Stores input dataset and contains helpers for convolutional DL (integer shift invariant).
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
    """
    def __init__(
        self, 
        input_data, 
        learned_dict=None, 
        reg_param=0.05, 
        num_loops=5, 
        num_generators=1, 
    ):

        super().__init__(input_data, learned_dict, reg_param, num_loops, num_generators)

        self.classification = 'convdl'  # prepends output filenames

        # framework used for learning dictionary
        # for evaluating distances of successive dictionaries during training
        self.suc_dict_type = 'conv'

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
        for jj in range(self.n):
            X[:,:,:,jj], err_vec[jj,], normsize_vec[jj,] = sparsecode_conv(
                self.input_data[:,jj], self.learned_dict, self.reg_param, self.num_loops
            )
        self.X = X

        # Get approximation error and regularization value
        self.apx_err = 0.5 * np.sum(err_vec)/self.n
        self.reg_val = self.reg_param * np.sum(normsize_vec)/self.n

    def dict_update_step(self):       
        """
        Task: using current self.X, update self.learned_dict
            Compute and update the learned dictionary for the current sparse vector X (fix X, update dictionary)
            by solving a least squares problem.
        """
        X = self.X
        xx = np.zeros((self.d*self.num_generators,self.d*self.num_generators))
        for i in range(self.num_generators):
            for k in range(self.num_generators):
                xx_s = np.zeros((self.d,self.d))
                for jj in range(self.n):
                    xx_s = xx_s + np.dot(np.transpose(X[:,:,k,jj]),X[:,:,i,jj])
                xx[k*self.d:(k+1)*self.d,i*self.d:(i+1)*self.d] = xx_s
                
        xy = np.zeros((self.d*self.num_generators,))
        for k in range(self.num_generators):
            xy_s = np.zeros((self.d,))
            for jj in range(self.n):
                xy_s = xy_s + np.dot(np.transpose(X[:,:,k,jj]),self.input_data[:,jj])
            xy[k*self.d:(k+1)*self.d] = xy_s
            
        bb = np.linalg.lstsq(xx,xy,rcond=None)[0]
        
        self.learned_dict =  np.reshape(bb,(self.num_generators,self.d)).T


def sparsecode_conv(data,dictionary,l,num_loops):
    """
    Solve the convex program - helper function for coding step
    (For convolutional dictionary learning)
    """
    d,q = dictionary.shape
    
    # Run the vectorized version
    x,err,normsize = sparsecode_conv_original(data,dictionary,l,num_loops)
    X = np.zeros((d,d,q))
    
    # Map values to an array
    for j in range(d):
        t = np.zeros(d)
        t[j] = 1
        T = circulant(t)
        for i in range(q):
            X[:,:,i] = X[:,:,i] + T * x[i*d+j]
            
    return X,err,normsize


def sparsecode_conv_original(data,dictionary,l,nLoops,regType='Lasso'):
    """
    Solve the convex program - helper function for coding step
    (For convolutional dictionary learning)
    """
    d,q = dictionary.shape
    
    # Embed dictionary into a larger d by d*q matrix
    D = np.zeros((d,d*q))
    for i in range(q):
        d_elem = dictionary[:,i]
        for j in range(d):
            D[:,i*d+j] = np.roll(d_elem,j)
    
    # Initialize at the origin
    x = np.zeros((d*q,))
    
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