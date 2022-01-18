import numpy as np

import ekde.ekdefunc
import hyperclip.hyperfunc
from hyperclip import Hyperplane
from .whitening_transformer import WhiteningTransformer
import pandas as pd
import time

def count_diff(A):
    n, d = A.shape
    
    B = np.ones_like(A)
    
    for i in range(n - 1)[::-1]:
        for j in range(d):
            if A[i,j] == A[i+1, j]:
                if j == 0:
                    B[i,j] = B[i+1,j] + 1
                elif B[i, j-1] > 1:
                    B[i,j] = B[i+1,j] + 1
    return(B)

def discretize(X, x_min, dx):
    Z = ((X - x_min) / dx).astype(int)
    return(Z)

def compute_centers(Z, x_min, dx):
    C = x_min + Z * dx + 0.5 * dx
    return(C)

class BKDE():
    def __init__(self, 
                 h='scott', 
                 q=11, 
                 bounds=[],
                 verbose=0):
        if q%2 == 0:
            raise(ValueError("Unexpected q value. q should be an odd int."))
            
        self.h = h
        self._h = None
        self.q = q
        self.bounds = bounds
        self.verbose = verbose
    
    def fit(self, X):
        if self.verbose > 0:
            print('BKDE fit process, X.shape=', X.shape)
        # preprocessing
        if len(X.shape) == 1:
            X = X[:,None]
        
        self._real_X_min = np.min(X, axis=0)
        self._real_X_max = np.max(X, axis=0)
        
        # get data dimensions
        self._n, self._d = X.shape
        
        # preprocessing
        self._wt = WhiteningTransformer()
        X = self._wt.fit_transform(X)
        
        self._x_min = np.min(X, axis=0)
        
        # BOUNDARIES INFORMATIONS
        A, R = self._set_boundaries(x = X[0])
        self.A = A
        self.R = R
        # BANDWIDTH SELECTION
        self._compute_bandwidth(X)
        
        self._x_min = X.min(axis=0)
        dx = self._h / self.q
        
        Z = discretize(X, self._x_min, dx=dx).astype(np.intc)
        Z = pd.DataFrame(Z)
        
        U_nu = Z.groupby(by=Z.columns.tolist()).size().reset_index(name="nu")
        
        self._U = U_nu[Z.columns.tolist()].values.astype(np.intc)
        self._nu = U_nu["nu"].values.astype(np.double)
        
        print('U.shape=', self._U.shape)
        
        self._U_diff_asc = np.ones((self._U.shape[0], self._d), dtype=np.intc)
        ekde.ekdefunc.count_diff_asc(self._U, self._U_diff_asc)
        
        self._U_diff_desc = np.ones((self._U.shape[0], self._d), dtype=np.intc)
        ekde.ekdefunc.count_diff_desc(self._U, self._U_diff_desc)
        
        C = compute_centers(self._U, self._x_min, dx).astype(np.double)
        self._nu /= np.array(hyperclip.hyperfunc.volumes(A, R, C, self._h))
        
        return(self)
        
    def predict(self, X):
        st_all = time.time()
        if self.verbose > 0:
            print('BKDE predict process, X.shape=', X.shape)
        X = self._wt.transform(X)
        
        id_out_of_bounds = np.zeros(X.shape[0]).astype(np.bool)
        for hyp in self._bounds_hyperplanes:
            id_out_of_bounds = np.any((id_out_of_bounds, ~hyp.side(X)), axis=0)
        st = time.time()
        Z = discretize(X, self._x_min, dx=self._h / self.q)
        print('discretize time', time.time()-st)
        # sort Z
        Z = pd.DataFrame(Z)
        Z['j'] = np.arange(Z.shape[0])
        print('sort')
        st = time.time()
        Z = Z.sort_values(by=[i for i in range(self._d)])
        print('sort done in', time.time()-st)
        Z_indices = Z['j'].values.astype(np.intc)
        Z = Z[[i for i in range(self._d)]].values.astype(np.intc)
        
        print('U.shape', self._U.shape)
        print('Z.shape=', Z.shape)
        Z_diff_asc = np.ones((Z.shape[0], self._d), dtype=np.intc)
        ekde.ekdefunc.count_diff_asc(Z, Z_diff_asc)
        
        Z_diff_desc = np.ones((Z.shape[0], self._d), dtype=np.intc)
        ekde.ekdefunc.count_diff_desc(Z, Z_diff_desc)
        time.sleep(0.5)
        st = time.time()
        f = np.array(ekde.ekdefunc.merge(U=self._U,
                                           U_diff_asc = self._U_diff_asc,
                                           U_diff_desc = self._U_diff_desc,
                                           nu=self._nu,
                                           Z=Z,
                                           Z_indices=Z_indices,
                                           Z_diff_asc=Z_diff_asc,
                                           Z_diff_desc=Z_diff_desc,
                                           q=self.q,
                                           h=self._h))
        print('merge in ', time.time()-st)
        f[id_out_of_bounds] = 0.0
        
        f = f / (self._h ** self._d * self._n)
        
        f /= self._wt.scale_
        print(time.time()-st_all)
        return(f)
    
    def _set_boundaries(self, x):
        self._bounds_hyperplanes = []

        for k, pos in self.bounds:
            if pos == 'left':
                self._add_boundary(k=k,
                                   value=self._real_X_min[k],
                                   x=x)
            elif pos == 'right':
                self._add_boundary(k=k,
                                   value=self._real_X_max[k],
                                   x=x)
            elif pos == 'both':
                self._add_boundary(k=k,
                                   value=self._real_X_min[k],
                                   x=x)
                self._add_boundary(k=k,
                                   value=self._real_X_max[k],
                                   x=x)
            else:
                raise(TypeError('Unexpected bounds parameters'))
        
        A = np.zeros((self._d, len(self._bounds_hyperplanes)))
        R = np.zeros(len(self._bounds_hyperplanes))
        
        for i_hyp, hyp in enumerate(self._bounds_hyperplanes):
            A[:, i_hyp] = hyp.a
            R[i_hyp] = hyp.r
        
        return(A, R)
    
    def _add_boundary(self, k, value, x):
        P = np.diag(np.ones(self._d))
        
        P[:, k] = value

        P_wt = self._wt.transform(P)

        hyp = Hyperplane().set_by_points(P_wt)
        hyp.set_positive_side(x)
        self._bounds_hyperplanes.append(hyp)
    
    def _compute_bandwidth(self, X):
        if type(self.h) is int or type(self.h) is float:
            self._h = float(self.h)

        elif type(self.h) is str:
            if self.h == 'scott' or self.h == 'silverman':
                # the scott rule is based on gaussian kernel
                # the support of the gaussian kernel to have 99%
                # of the density is 2.576
                self._h = 2.576 * scotts_rule(X)
            else:
                raise (ValueError("Unexpected bandwidth selection method."))
        else:
            raise (TypeError("Unexpected bandwidth type."))

        if self.verbose > 0:
            print('Bandwidth selection done : h=' + str(self._h))
            
    def set_params(self, **params):
        """
        Set parameters.

        Parameters
        ----------
        **params : kwargs
            Parameters et values to set.

        Returns
        -------
        self : DensityEstimator
            The self object.

        """
        for param, value in params.items():
            setattr(self, param, value)
    
    
def scotts_rule(X):
    """
    Scott's rule according to "Multivariate density estimation", Scott 2015, p.164.
    The Silverman's rule is exactly the same in "Density estimation for statistics and data analysis", Silverman 1986, p.87.
    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features).

    Returns
    -------
    h : float
        Scotts rule bandwidth. The returned bandwidth should be then factored
        by the data variance.

    """
    n = X.shape[0]
    d = X.shape[1]
    return(n**(-1/(d + 4)))