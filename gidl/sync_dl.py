import numpy as np

import gidl.utils as gc
from gidl.regular_dl import RegularDL

class SyncDL(RegularDL):
    """
    Stores input dataset and contains helpers for regular dictionary learning.
    Initializes a learned dictionary *if not provided*.

    Args:
        input_data (`numpy.ndarray`): input dataset
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
        learned_dict=None, 
        reg_param=0.1, 
        num_loops=5, 
        num_generators=1, 
    ):

        self.classification = 'syncdl'  # prepends output filenames
        self.input_data = input_data
        self.datum_dim, self.num_elem, self.n = self.input_data.shape  # (dimension of datum, num elements, num datapoints)
        self.reg_param = reg_param
        self.num_loops = num_loops
        self.num_generators = num_generators

        # framework used for learning dictionary
        # for evaluating distances of successive dictionaries during training
        self.suc_dict_type = 'sync'

        self.learned_dict = learned_dict
        if not self.learned_dict:  # if no learned dictionary is provided, initalize one
            self.learned_dict = np.random.randn(self.datum_dim, self.num_elem, self.num_generators)
            self.normalize_step()
        self.d, self.num_elem, self.num_generators = self.learned_dict.shape  # dictionary matrix shape

        self.apx_err = None
        self.reg_val = None

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
            X[:,:,:,jj],err_vec[jj,],normsize_vec[jj,] = sparsecode_sync(
                self.input_data[:,:,jj],self.learned_dict,self.reg_param,self.num_loops
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
        d, r, q, n = self.d, self.num_elem, self.num_generators, self.n

        xx = np.zeros((d*r,d*r,q))
        for l in range(q):
            for k in range(r):
                xx_s = np.zeros((d,d))
                for jj in range(n):
                    xx_s += X[:,:,l,jj].T @ X[:,:,l,jj]
                xx[k*d:(k+1)*d,k*d:(k+1)*d,l] = xx_s
            
        xy = np.zeros((d*r,q))
        for l in range(q):
            for k in range(r):
                xy_s = np.zeros((d,))
                for jj in range(n):
                    xy_s += X[:,:,l,jj].T @ self.input_data[:,k,jj]
                xy[k*d:(k+1)*d,l] = xy_s
        
        bb = np.zeros((d*r,q))
        for l in range(q):
            bb[:,l] = np.linalg.lstsq(xx[:,:,l],xy[:,l],rcond=None)[0]
            self.learned_dict[:,:,l] =  np.reshape(bb[:,l],(r,d)).T

    def normalize_step(self):
        """
        Normalize the dictionary
        """
        A = self.learned_dict
        for l in range(self.num_generators):
            A[:,:,l] = gc.normalize(A[:,:,l])
        self.learned_dict = A


def sparsecode_sync(data,dictionary,reg_param,nLoops):
    """
    Solve the convex program - helper function for coding step
    (For dictionary learning for synchronization)
    """
    d,r,q = dictionary.shape
    
    # Initialize at the origin
    x = np.zeros((d,d,q))
    
    # Outer Loop iterations
    for i in range(nLoops):
        # Compute the Residual
        rem = - data
        for l in range(q):
            rem += x[:,:,l] @ dictionary[:,:,l]
        
        # Compute the gradient
        g = np.zeros(x.shape)
        for l in range(q):
            for j in range(r):
                g[:,:,l] += np.outer(rem[:,j],dictionary[:,j,l]) 
            
        ag = np.zeros(data.shape)
        for l in range(q):
            ag += g[:,:,l] @ dictionary[:,:,l]

        ag = np.reshape(ag,(d*r,))
        nn = np.linalg.norm(ag)**2
        
        rem = np.reshape(rem,(d*r,))
        eta = 1.5*np.dot(ag,rem) / nn
        
        # Take a gradient step
        x -= (g * eta)
        
        # Proximal Map
        normsize = 0.0
        for l in range(q):
            x[:,:,l],normsize0 = spec_prox(x[:,:,l],reg_param)
            normsize += normsize0
            
    err = np.linalg.norm(rem)**2
    
    return x,err,normsize


def spec_prox(X,l):
    """
    Helper function for sparsecode_sync
    """
    # Compute the proximal operator with respect to the spectral norm
    # First compute SVD
    u,s,v = np.linalg.svd(X)
    s_max = np.max(s)
    t_max = np.max((s_max-l,0.0))
    
    # Clip all values to at most t_max
    # where t_max is spec norm - l, or 0, whichever is bigger
    s = np.clip(s,0.0,t_max)
    
    Z = (u @ np.diag(s)) @ v
    
    return Z,t_max