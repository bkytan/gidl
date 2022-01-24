import numpy as np
from math import pi,sin,cos,sqrt
from scipy.linalg import toeplitz


def softthreshold(vec,reg_param):
    # Soft Threshold function
    # Compatible with complex input
    q = vec.shape[0]
    scalevec = np.zeros(vec.shape)
    for i in range(q):
        c = np.absolute(vec[i,])
        if c > reg_param:
            scalevec[i,] = abs((c - reg_param) / c)
    svec = np.multiply(scalevec,vec)    
    return svec


def generate_linearmap(d,q,cplx=False):
    A = np.random.randn(d,q)
    
    if cplx==True:
        B = np.random.randn(d,q)
        A = A + B * 1j
        
    A = normalize(A)
    
    return A
    

def normalize(A):
    d,q = A.shape
    for i in range(q):
        A[:,i] *= 1/np.linalg.norm(A[:,i])
    return A


def dict_distance_reg(A,B,dicttype='regular'):
    if dicttype in ['regular', 'conv', 'csp', 'cts']:
        _,qA = A.shape
        _,qB = B.shape
        err = 0.0
        for i in range(qA):
            emin = 10000.0
            for j in range(qB):
                if dicttype == 'regular':
                    e = _vec_dist_regular(A[:,i],B[:,j])
                if dicttype == 'conv':
                    e = _vec_dist_conv(A[:,i],B[:,j])
                if dicttype == 'csp':
                    e = _vec_dist_csp(A[:,i],B[:,j])
                if dicttype == 'cts':
                    e = _vec_dist_cts(A[:,i],B[:,j])
                emin = min(e,emin)
                
            err = err + emin**2/qA
        err = np.sqrt(err)
        
        return err

    elif dicttype == 'sync':
        d = 0.0
        _,_,l1 = A.shape
        _,_,l2 = B.shape
        
        for i in range(l1):
            min_d = 10E8
            for j in range(l2):
                dd = dict_distance_block(A[:,:,i], B[:,:,j])
                min_d = min((min_d,dd))
            d += min_d
            
        return d      


def _vec_dist_regular(a,b):
    e1 = np.linalg.norm(a-b)
    e2 = np.linalg.norm(a+b)
    return min(e1,e2)


def _vec_dist_conv(a,b):
    d = b.shape[0]
    norm_val = np.linalg.norm(a-b)
    for k in range(d):
        e = np.linalg.norm(a-np.roll(b,k))
        norm_val = min(e,norm_val)
        e = np.linalg.norm(-a-np.roll(b,k))
        norm_val = min(e,norm_val)
    return norm_val


def DFT(n):
    """
    Discrete Fourier Transform Matrix
    """
    W = np.zeros((2*n+1,2*n+1)).astype(complex)
    for i in range(2*n+1):
        for j in range(2*n+1):
            ii = i - n
            arg = ii * j * (2 * pi) / (2*n+1)
            W[i,j] = cos(arg) + 1j * sin(arg)
    W = W / sqrt(2*n+1)
    return W


def _vec_dist_csp(a,b):
    
    d = b.shape[0]
    d_half = int((d-1)/2)
    W = DFT(d_half)
    
    a = W @ a
    b = W @ b

    # Begin with a crude upper bound
    err = 10 * (np.linalg.norm(a)**2 + np.linalg.norm(b)**2)
    
    # Perform a grid search
    #n = 50
    #ints = np.arange(n) * (1.0/n)
    #for i in ints:
    #    for k in ints:
    #        ee = _vec_dist_theta(a,b,i,k)
    #        err = min(err,ee)


    # Perform a binary search
    n = 20
    argmin_theta = 0.5
    argmin_phi = 0.5
    width = (1.0/n)
    nDepths = 20
    for jj in range(nDepths):
        ints_theta = (np.arange(n) - np.ones((n,))*(n/2))*width + argmin_theta*np.ones((n,)) 
        ints_phi = (np.arange(n) - np.ones((n,))*(n/2))*width + argmin_phi*np.ones((n,)) 
        for theta in ints_theta:
            for phi in ints_phi:
                e = _vec_dist_theta(a,b,theta,phi)
                if e < err:
                    err = e
                    argmin_theta = theta
                    argmin_phi = phi
                    
        #print(argmin_theta)
        #print(argmin_phi)
        
        width *= 4.0/n
        
    return err


def _vec_dist_theta(a,b,theta,phi):
    # Compute the l2-norm between a and e^i theta times b
    dd = np.shape(b)[0]
    d = int((dd-1)/2)
    
    bb = b.copy()
    arg = 2*theta*pi
    bb *= cos(arg) + 1j*sin(arg) # Multiply by e^ i theta
    
    for ii in range(dd):
        arg = phi * (2*pi) * (ii-d)
        bb[ii,] *= cos(arg) + 1j*sin(arg)
    e = np.linalg.norm(a - bb)
    
    return e


def _vec_dist_cts(a,b):
    d = len(a)
    d_half = int((d-1)/2)
    W = DFT(d_half)
    
    a = W @ a
    b = W @ b

    # Begin with a crude upper bound
    err = 10 * (np.linalg.norm(a)**2 + np.linalg.norm(b)**2)
    
    # Perform a binary search
    n = 20
    argmin = 0.0
    width = (1.0/n)
    nDepths = 20
    for jj in range(nDepths):
        
        ints = (np.arange(n) - np.ones((n,))*(n/2))*width + argmin*np.ones((n,)) 
        for i in ints:
            e = _vec_dist_theta(a,b,0,i)
            if e < err:
                err = e
                argmin = i
        
        width *= 4.0/n
    
    return err


def dict_distance_block(X1,X2):
    """
    Returns the distance between two fat matrices by setting the U component
    in the SVD to be equal
    
    Approximately solving
    min_{Q} || X1 - Q X2 || over the space of orthogonal matrices
    """
    AB = X1 @ X2.T
    U,_,V = np.linalg.svd(AB)
    Q = np.dot(U,V)
    err = X1 - Q @ X2
    d = np.linalg.norm(err,'fro')
        
    return d**2


def _projPSD(X):
    """
    Projector onto the cone of PSD matrices
    """
    d,U = np.linalg.eig(X)
    dplus = np.maximum(d,np.zeros(d.shape))
    Xplus = np.dot(np.dot(U,np.diag(dplus)),U.T.conjugate())
    
    return Xplus


def _isToeplitz(X):
    X2 = toeplitz(X[:,0])
    err = np.linalg.norm(X-X2)

    return err
    

def _projToep(X):
    """
    Projector onto the subset of Hermitian Toeplitz matrices
    """
    d,_ = X.shape
    XToep = np.zeros((d,d)).astype(complex)
    
    for i in range(d):
        c = 0.0
        # Sum all entries that should be equal
        for j in range(i,d):
            c = c + X[j-i,j].conjugate() + X[j,j-i]
        # Take the arithmetic mean
        c = c / (2*(d-i)) # Double counting everything, including the diagonal
        # Plug in all the entries
        for j in range(i,d):
            XToep[j,j-i] = c
            XToep[j-i,j] = c.conjugate()
    
    return XToep
    

def _projToepPSDvec(v,nIterates=10):
    """
    Projector onto the set of PSD Toeplitz matrices
    """
    
    # Set the Top 
    v = v.copy()
    
    # Set the diagonal entries to be REAL
    v[0,] = np.real(v[0,])
    
    X = toeplitz(v)
    P = np.zeros(X.shape).astype(complex)
    Q = np.zeros(X.shape).astype(complex)
    
    for i in range(nIterates):
        Y = _projPSD(X+P)
        P = X + P - Y
        X = _projToep(Y+Q)
        Q = Y + Q - X
        
    return X[:,0]


def _projToepPSDvec_orig(v,nIterates=10):
    """
    Projector onto the set of PSD Toeplitz matrices
    """
    
    # Set the Top 
    v = v.copy()
    
    # Set the diagonal entries to be REAL
    v[0,] = np.real(v[0,])
    
    X = toeplitz(v)
    
    for i in range(nIterates):
        X = _projPSD(X)
        X = _projToep(X)
        
    return X[:,0]


def _projLowerBlockToepPSDvec(x,t,z,nIterates=50):
    """
    Projector onto the set of PSD Toeplitz matrices
    """
    
    # Set the Top 
    x = x.copy()
    z = z.copy()
    
    # Set the diagonal entries to be REAL
    z[0,] = np.real(z[0,])
    d = z.shape[0]
    
    X = np.zeros((d+1,d+1)).astype(complex)
    X[0,0] = t
    X[1:,1:] = toeplitz(z)
    X[0,1:] = x.T.conjugate()
    X[1:,0] = x
    
    P = np.zeros((d+1,d+1)).astype(complex)
    Q = np.zeros((d+1,d+1)).astype(complex)
    
    for i in range(nIterates):
        #Y = _projPSD(X+P)
        Y = X + P
        Y[1:,1:] = _projToep(X[1:,1:]+P[1:,1:])
        P = X + P - Y
        #X[1:,1:] = _projToep(Y[1:,1:]+Q[1:,1:])
        X = _projPSD(Y+Q)
        Q = Y + Q - X
        
    z = X[0,1:]
    z[0,] = np.real(z[0,])
    return X[1:,0],np.real(X[0,0]),z