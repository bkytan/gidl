import numpy as np
import pickle
import os
import time
from matplotlib import pyplot as plt

import gidl.utils as gc

class RegularDL(object):
    """
    Stores input dataset and contains helpers for regular dictionary learning.
    Initializes a learned dictionary *if not provided*.

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
            q number of generators for initializing dictionary, *if learned_dict is not provided* (defaults to size of datum)
    """
    def __init__(
        self, 
        input_data, 
        learned_dict=None, 
        reg_param=0.05, 
        num_loops=5, 
        num_generators=None, 
    ):

        self.classification = 'regdl'  # prepends output filenames
        self.input_data = input_data
        self.datum_dim, self.n = self.input_data.shape  # (dimension of datum, num datapoints)
        self.reg_param = reg_param
        self.num_loops = num_loops

        # framework used for learning dictionary
        # for evaluating distances of successive dictionaries during training
        self.suc_dict_type = 'regular'

        self.num_generators = num_generators
        if not self.num_generators:  
            self.num_generators = self.datum_dim * 1  # default number of generators if learned_dict is not provided

        self.learned_dict = learned_dict
        if not self.learned_dict:  # if no learned dictionary is provided
            self.learned_dict = gc.generate_linearmap(self.datum_dim, self.num_generators)  # initialization
        self.d, self.num_generators = self.learned_dict.shape  # dictionary matrix shape

        self.apx_err = None
        self.reg_val = None

    def learn_dict_mul(
        self, 
        num_iterates, 
        COMPARE=None, 
        ORA_DICT_TYPE='conv', 
        SAVE_FILE=True, 
        SHOW_PROGRESS=True, 
        PLOT=0, 
        SAVE_FOLDER_NAME=f"output/{time.strftime('%Y-%m-%d-%H%M%S')}/",
    ):
        """
        Perform multiple iterations of alternating minimizations for learning dictionary and updates self.learned_dict

        Args:
            num_iterates (`int`): number of iterations of alternating minimizations to perform
            COMPARE (`np.ndarray`, optional):
                ground truth dictionary, if available, for comparison against the learned dictionary every iteration (default: None)
            ORA_DICT_TYPE (`str`, optional):
                type of ground truth dictionary, to be specified if COMPARE is provided.
                available types are: (i) 'regular'; (ii) 'conv'; (iii) 'csp'; (iv) 'cts' (default: 'conv')
            SAVE_FILE (`bool`, optional): save learning progress every iteration (default: True)
            SHOW_PROGRESS (`bool`, optional): print learning progress every iteration (default: True)
            PLOT (`int`, optional): 
                plot learned dictionary. 0 means no graphs will be plotted; 
                1 saves graphs as pdf; 2 saves graphs as png; 3 saves as both pdf and png (default: 0)
            SAVE_FOLDER_NAME (`str`, optional): folder directory for SAVE_FILE and PLOT (default: 'output/(datetime)')
        Returns:
            self.learned_dict: updated learned dictionary after num_iterates
        """
        # Initialize table to track training progress
        TABLE_apx = np.zeros((num_iterates+1,))
        TABLE_suc = np.zeros((num_iterates+1,))
        TABLE_obj = np.zeros((num_iterates+1,))
        TABLE_tim = np.zeros((num_iterates+1,))
        TABLE_dict = [self.learned_dict.copy(),]
        if not COMPARE is None:
            TABLE_ora = np.zeros((num_iterates+1,))

        # Prepare for training progress saving/printing
        if SAVE_FILE:
            if not os.path.exists(SAVE_FOLDER_NAME):
                os.makedirs(SAVE_FOLDER_NAME)
            fname = SAVE_FOLDER_NAME + self.classification + '_q' + str(self.num_generators)
        if SHOW_PROGRESS:
            header = 'It# | Suc. Err | Apx. Err | Reg. Val | Obj. Val | Ora. Err | Time |'
            if COMPARE is None:
                header = 'It# | Suc. Err | Apx. Err | Reg. Val | Obj. Val | Time |'
            print (header)
        if PLOT:
            fig = plt.figure()

        # Training loop
        for i in range(num_iterates):
            time_code, time_taken = self.learn_dict()  # Calls one iteration of alternating min (one learning loop)

            # Evaluate learned dictionary
            TABLE_suc[i+1,] = gc.dict_distance_reg(TABLE_dict[-1], self.learned_dict, dicttype=self.suc_dict_type)
            TABLE_dict.append(self.learned_dict.copy())
            TABLE_apx[i+1,] = self.apx_err
            TABLE_obj[i+1,] = self.apx_err + self.reg_val
            if not COMPARE is None:
                TABLE_ora[i+1,] = gc.dict_distance_reg(self.learned_dict, COMPARE, dicttype=ORA_DICT_TYPE)
            TABLE_tim[i+1,] = time_taken
            OUTPUT = {'TABLE_A' : TABLE_dict, 'TABLE_apx' : TABLE_apx, 
                    'TABLE_suc' : TABLE_suc, 'TABLE_obj' : TABLE_obj,
                    'ELAPSED_TIME' : TABLE_tim}

            # Training progress saving/printing
            if SAVE_FILE:
                with open(fname + '.pickle', 'wb') as handle:
                    pickle.dump(OUTPUT, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if SHOW_PROGRESS:
                print(str(i+1) + ' | ', end='')
                print(str(TABLE_suc[i+1]) + ' | ', end='')
                print(str(TABLE_apx[i+1]) + ' | ', end='')
                print(str(self.reg_val) + ' | ', end='')
                print(str(TABLE_obj[i+1]) + ' | ', end='')
                if not COMPARE is None:
                    print(str(TABLE_ora[i+1]) + ' | ', end='')
                print(str(TABLE_tim[i+1]) + ' | ', end='')
                print('')
            if PLOT:
                fig.clf()
                for j in range(self.num_generators):
                    plt.plot(self.learned_dict[:,j])
                if PLOT==1 or PLOT==3:
                    fig.savefig(fname + '_' + str(i+1) + '.pdf', bbox_inches='tight')
                if PLOT==2 or PLOT==3:
                    fig.savefig(fname + '_' + str(i+1) + '.png', bbox_inches='tight')

        # return self.learned_dict

    def learn_dict(self):
        """
        Perform one iteration of alternating minization algorithm
        Consists of the (i) coding step, (ii) dictionary update step, (iii) normalization step
        """
        time_start = time.time()
        self.coding_step()
        time_code = time.time() - time_start
        self.dict_update_step()
        self.normalize_step()
        time_taken = time.time() - time_start

        return time_code, time_taken

    def coding_step(self):
        """
        Task: using current self.learned_dict, update (i) self.X, (ii) self.apx_err, (iii) self.reg_val
            Compute and update the sparse vector X for the current dictionary (fix dictionary, update X)
            by solving a convex program.
            In addition, update the approximation error and regularization value
        """
        # Re-initialize sparse vector X
        X = np.zeros((self.num_generators, self.n))

        # Fix dictionary, update sparse vector X
        for i in range(self.n):
            X[:,i] = svlasso(self.input_data[:,i], self.learned_dict, self.reg_param, self.num_loops)
        self.X = X

        # Get approximation error and regularization value
        self.apx_err = 0.5 * np.linalg.norm(self.learned_dict @ X - self.input_data, 'fro')**2 / self.n
        self.reg_val = np.sum(np.sum(np.absolute(X))) * self.reg_param / self.n

    def dict_update_step(self):
        """
        Task: using current self.X, update self.learned_dict
            Compute and update the learned dictionary for the current sparse vector X (fix X, update dictionary)
            by solving a least squares problem
        """
        self.learned_dict = np.dot(self.input_data, np.linalg.pinv(self.X))
    
    def normalize_step(self):
        """
        Normalize the dictionary
        """
        self.learned_dict = gc.normalize(self.learned_dict)


def svlasso(y,A,reg,nLoops):
    """
    Solve the convex program - helper function for coding step
    (For regular dictionary learning)
    """
    _,q = A.shape
    X = np.zeros((q,)) # Initialize at 0
    
    for i in range(nLoops):
        
        # Compute the gradient
        rem = np.dot(A,X)-y
        g = np.dot((A.T) , rem)
        ag = np.dot(A,g)
        nn = np.linalg.norm(ag)**2
        eta = 1.5*np.dot(ag,rem) / (nn)
        # Take a gradient step
        X = X - (g * eta)
        
        # Soft-Threshold
        X = gc.softthreshold(X,reg)
        
    return X