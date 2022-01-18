# -*- coding: utf-8 -*-

from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import ekde
from time import time

#%% Dataset

X_min = np.array([-0.5, 0.3])
x1_max = 1.5

mean = np.array([0,0])
rho = 0.7
cov = np.array([[1,rho],[rho,1]])

def _pdf(X, X_min):
    f = stats.multivariate_normal.pdf(X, mean=mean, cov=cov) 

    f[np.any(X<X_min, axis=1)] = 0
    f[X[:,1]>=x1_max] = 0

    return(f)

def _rvs(X_min, n,seed=None):
    np.random.seed(seed)
    X = stats.multivariate_normal.rvs(mean=mean, cov=cov, size=n)
    np.random.seed(None)
    X = X[np.all(X >= X_min, axis=1)]
    X = X[X[:,1] < x1_max]

    return(X)

def bounded_set(n, seed):
    X = _rvs(X_min, n, seed=seed)
    
    xk = (np.linspace(X[:,0].min()-X[:,0].std(),X[:,0].max()+X[:,0].std(),300),
          np.linspace(X[:,1].min()-X[:,1].std(),X[:,1].max()+X[:,1].std(),300))
    X_grid = np.meshgrid(xk[0], xk[1])
    X_grid = np.vstack((X_grid[0].flat, X_grid[1].flat)).T
    
    pdf_grid = np.sum(_pdf(X_grid, X_min)) * np.product(X_grid.max(axis=0)-X_grid.min(axis=0)) / X_grid.shape[0]
    
    Y = np.vstack((np.ones(100)*0.5,
                   np.linspace(0,2,100))).T
    
    pdf_Y = _pdf(Y, X_min) / pdf_grid
    
    return(X, Y, pdf_Y, X_grid)
    
def hyp_func(hyp, x):
    # a0 x0 + a1 x1 + r = 0
    # x1 = -r - a0 x0 / a1
    return((-hyp.r - hyp.a[0] * x) / hyp.a[1])

# %%
X, Y, pdf_Y, X_grid = bounded_set(10**7, 42)
print(X.shape[0]/10**6)
pdf_grid = np.sum(_pdf(X_grid, X_min)) * np.product(X_grid.max(axis=0)-X_grid.min(axis=0)) / X_grid.shape[0]
f_grid_exact = _pdf(X_grid, X_min) / pdf_grid
#%%
st = time()
bkde = ekde.BKDE(q=21,
                    bounds=[
                        (0, 'left'),
                        (1, 'both'),
                        ],
                    verbose=1)
bkde.fit(X)
print(time()-st)

#%%
X2, Y, pdf_Y, X_grid = bounded_set(10**6, 30)

#%%
f_X = bkde.predict(X2)
idx = np.random.choice(X2.shape[0], 10**4, replace=False)
plt.scatter(X2[idx,0], X2[idx,1], c=f_X[idx], s=2)

#%%
# U = np.array([[1,1,1],
#               [1,1,3],
#               [1,1,4],
#               [1,2,4],
#               [1,3,3],
#               [2,1,1],
#               [2,1,2],
#               [2,2,2]], dtype=np.intc)

# U_diff_desc = np.ones(U.shape, dtype=np.intc)
# ekde.ashfunc.count_diff_desc(U, U_diff_desc)

# bkde._U = U
# bkde._U_diff_desc = U_diff_desc

#%%
f_X = bkde.predict(X)
# 10**7 -> 42 sec 
# 10**6 -> 3.3 sec
# ([4, 249], 11.652764262505263)
#%%
import sparse
s = sparse.COO(bkde._U.T, bkde._nu)
s = sparse.GCXS(s, compressed_axes=[0])



#%%
f_eval = bkde.predict(Y[:])
plt.plot(Y[:,1], pdf_Y)
plt.plot(Y[:,1], f_eval)

#%%
f_X = bkde.predict(X)
print('mad', np.abs(f_X - _pdf(X, X_min) / pdf_grid).mean())

#%%
idx = np.random.choice(X.shape[0], 10**4, replace=False)
plt.scatter(X[idx,0], X[idx,1], c=f_X[idx], s=2)

#%%
import pandas as pd
Z = ekde.ash.discretize(bkde._wt.transform(X), bkde._x_min, bkde._h/bkde.q)
n_grid, d = Z.shape
Z = pd.DataFrame(Z)

#%%
Z['j'] = np.arange(n_grid)
Z = Z.sort_values(by=[i for i in range(X_grid.shape[1])])
Z_indices = Z['j'].values.astype(np.intc)
Z = Z[[i for i in range(d)]].values.astype(np.intc)
Z_diff = np.ones((n_grid, d), dtype=np.intc)
ekde.ashfunc.count_diff(Z, Z_diff)
f = np.array(ekde.ashfunc.merge(U=bkde._U,
                                   nu=bkde._nu,
                                   Z=Z,
                                   Z_indices=Z_indices,
                                   Z_diff=Z_diff,
                                   q=bkde.q))
#%%
print(f_grid.sum() * np.product(X_grid.max(axis=0)-X_grid.min(axis=0)) / X_grid.shape[0])
pdf_grid = np.sum(_pdf(X_grid, X_min)) * np.product(X_grid.max(axis=0)-X_grid.min(axis=0)) / X_grid.shape[0]
f_grid_exact = _pdf(X_grid, X_min) / pdf_grid
print(f_grid_exact.sum() * np.product(X_grid.max(axis=0)-X_grid.min(axis=0)) / X_grid.shape[0])

#%%
mad = []
for h in np.arange(0.1, 0.5, 0.02):
    
    bkde = ekde.BKDE(
                        h = float(h),
                        q=101,
                        bounds=[
                                (0, 'left'),
                                (1, 'both'),
                                ],
                        )
    bkde.fit(X)
    f_grid = bkde.predict(X_grid)
    
    mad.append(np.abs(f_grid - f_grid_exact).mean())

#%%
plt.plot(np.arange(0.1,0.5,0.02), mad)



#%%
a = ekde.ash.discretize(bkde._wt.transform(Y[[17],:]), bkde._x_min, dx=bkde._h/bkde.q)

#%%
plt.scatter(bkde._U[:,0], bkde._U[:,1], s=0.1, c=bkde._nu)
# plt.xlim([0,1000])
# plt.ylim([0,500])
plt.colorbar()

#%%
from hyperclip import Hyperplane
from hyperclip import hyperfunc, Hyperclip
h = 0.3165675288885327
z = np.array([[14,40]])
x = ekde.ash.compute_centers(z, bkde._x_min, bkde._h/bkde.q)[0]
a = x - bkde._h/2
b = x + bkde._h/2
x_r = np.random.random((1000,2)) * (b-a) + a
A = np.zeros((2,2))
R = np.zeros(2)
is_in = np.ones(x_r.shape[0]).astype(bool)
for i, i_hyp in enumerate([1]):
    hyp = Hyperplane(bkde.A[:,i_hyp], bkde.R[i_hyp])
    
    is_in = np.all((is_in, hyp.side(x_r)), axis=0)
    
    hyp.affine_transform(translation = - x + bkde._h/2, 
                         scale = np.ones(bkde._d) / bkde._h,
                         inplace=True)
    # hyp.set_positive_side(x)
    A[:, i] = hyp.a
    R[i] = hyp.r

# print
hyperclip = Hyperclip(cython=True).set_A_R(A, R)
# print('class', hyperclip.volume())

A = np.hstack((A, np.zeros((2,1))))
R = np.hstack((R, 0))
print("check A", hyperfunc.clipping_condition_A_according_m(hyperclip.A, hyperclip.R, m=2))
print('func', hyperfunc.volume_according_m(hyperclip.A, hyperclip.R, m=2, check_A=False, zero=hyperclip.zero))
# print(ekde.ashfunc.volume(hyperclip.A, hyperclip.R, check_A=False))
print(is_in.mean())
plt.scatter(x_r[:,0], x_r[:,1], c=is_in)
plt.colorbar()

#%%
from hyperclip import Hyperplane
i_hyp = 0

x = np.array([-1.57648396, -2.72955008])
hyp = Hyperplane(bkde.A[:,i_hyp], bkde.R[i_hyp])
a = x - bkde._h/2
b = x + bkde._h/2
x_r = np.random.random((1000,2)) * (b-a) + a
plt.scatter(x[0], x[1])
plt.plot([x[0] - bkde._h/2, x[0] + bkde._h/2],
         [hyp_func(hyp, x[0] - bkde._h/2), hyp_func(hyp, x[0] + bkde._h/2)])
plt.xlim([x[0] - bkde._h/2, x[0] + bkde._h/2])
plt.ylim([x[1] - bkde._h/2, x[1] + bkde._h/2])
plt.scatter(x_r[:,0], x_r[:,1], c=hyp.side(x_r).astype(int))
plt.colorbar()
plt.show()

#%%
hyp.affine_transform(translation = - x + bkde._h/2, 
                     scale = np.ones(bkde._d) / bkde._h,
                     inplace=True)
x = np.array([0.5,  0.5])
plt.scatter(x[0], x[1])
plt.plot([0, 1],
         [hyp_func(hyp, 0), hyp_func(hyp, 1)])
plt.xlim([0,1])
plt.ylim([0,1])
print(hyp.a, hyp.r)



#%%
# h = bkde._h
wt = bkde._wt
plt.scatter(wt.transform(X)[:,0], wt.transform(X)[:,1])



#%%
f_eval = bkde.predict(Y)
plt.plot(Y[:,1], pdf_Y)
plt.plot(Y[:,1], f_eval)

#%%
f_grid = bkde.predict(X_grid)
print(f_grid.sum() * np.product(X_grid.max(axis=0) - X_grid.min(axis=0)) / X_grid.shape[0])
print(_pdf(X_grid, X_min).sum() * np.product(X_grid.max(axis=0) - X_grid.min(axis=0)) / X_grid.shape[0])

#%%
# dx = bkde._h / bkde.q
# Y = bkde._x_min + bkde._U * dx + 0.5 * dx
# # Y = bkde._wt.inverse_transform(Y)
# plt.scatter(Y[:,0], Y[:,1], c=bkde._nu, s=2)
# plt.colorbar()

#%%
st = time()
f = bkde.predict(X)
print(time()-st)



#%%
ash = ekde.ASH(q=10,
        bounds=[
                # (0, 'left'),
                # (1, 'both')
                ],
        n_mc=10000,
        n_jobs=4)
st = time()

# ash._compute_bandwidth(X[:10**6])
# h = ash._h

#%%
st = time()
ash.fit(X)
print('fit exec time : ', time()-st)

#%%
plt.scatter(ash._U_nu[0], ash._U_nu[1], c=ash._U_nu["nu"])
# plt.xlim([10,15])
# plt.ylim([2,5])

#%%
# X = np.array([[0.2,0.4]])
st = time()
f = ash.predict(X)
print(time()-st)
# print(f)
# plt.plot(Y[:,1], pdf_Y)
# plt.plot(Y[:,1], f)
# plt.show()
plt.scatter(X[:,0], X[:,1],  c=f[1], s=4)
#%%
a = np.meshgrid(*(np.arange(10) for i in range(3)))
a = np.vstack([aa.flat for aa in a]).T

#%%
plt.scatter(X[:,0], X[:,1],  c=f[1], s=4)

#%%
plt.scatter(f[0][:,0], f[0][:,1], c=np.arange(f[0].shape[0]))



#%%
X = np.array([[1.1, 1.1],
              [1.1, 1.2],
              [2.1, 1.1]])
ash = ekde.ASH(h=3.0, q=3)
ash.fit(X)

#%%
X_eval = np.array([[0.5, 1.1],
                   [0.5, 0.5]])
f = ash.predict(X_eval)
print(f)
#%%
U_N = ash._U_N[0]

U = U_N[[i for i in range(X.shape[1])]].values
N = U_N["N"].values

i_shift = 0
U_shift = ((U - i_shift) / 100).astype(int)
print(U_shift)

#%%
%timeit ash.fit(X)
# cython : 174 ms
# npg : 103 ms


#%%
# X = np.array([[]])

#%%
import pandas as pd

class Digitize():
    def __init__(self, dx, shift=0):
        self.dx = dx
        self.shift = shift

    def fit(self, X):
        self._d = X.shape[1]
        self._bins = [np.arange(V.min() - self.dx + self.shift,
                                V.max() + self.dx + self.shift,
                                self.dx) for V in X.T]

        return (self)

    def transform(self, X):
        X = X.copy()
        for k in range(self._d):
            X[:, k] = np.digitize(X[:, k], bins=self._bins[k])
        return (X.astype(int))

    def fit_transform(self, X):
        self.fit(X)

        return (self.transform(X))

def pandas_func(X):
    X_wt = ash._wt.fit_transform(X)
    
    histograms = []
    for i_shift in range(ash.q):
        digitizer = Digitize(dx = ash._h, shift = i_shift * ash._h / ash.q) 
        df = pd.DataFrame(digitizer.fit_transform(X_wt))
        
        df_uniques = df.groupby(by=df.columns.to_list()).size().reset_index(name='P')
        df_uniques['P'] /= df_uniques['P'].sum()
        
        histograms.append(df_uniques)
    return(histograms)

def numpy_func(X):
    X_wt = ash._wt.fit_transform(X)
    
    histograms = []
    for i_shift in range(ash.q):
        digitizer = Digitize(dx = ash._h, shift = i_shift * ash._h / ash.q) 
        X_wt_digit = digitizer.fit_transform(X_wt)
        

st = time()
# histograms = numpy_func(X)
X_wt = ash._wt.fit_transform(X)

histograms = []
for i_shift in range(ash.q):
    digitizer = Digitize(dx = ash._h, shift = i_shift * ash._h / ash.q) 
    X_wt_digit = digitizer.fit_transform(X_wt)
    histograms.append(np.unique(X_wt_digit, axis=0, return_counts=True))
    
print(time()-st)

#%%
import numpy_indexed as npi
import numpy_groupies as npg
st = time()
npga = npg.aggregate(X_wt_digit.T, a=1, func='sum', fill_value=0)
print(time()-st)

#%%
import numpy_groupies as npg
npga = npg.aggregate(X_wt_digit.T, a=1, func='sum', fill_value=0)

#%%
%timeit pandas_func(X)

#%%
from numpy_indexed import *
# three sets of graph edges (doublet of ints)
edges = np.random.randint(0, 9, (3, 100, 2))
# find graph edges exclusive to one of three sets
ex = exclusive(*edges)
print(ex)
# which edges are exclusive to the first set?
print(contains(edges[0], ex))
# where are the exclusive edges relative to the totality of them?
print(indices(union(*edges), ex))
# group and reduce values by identical keys
values = np.random.rand(100, 20)
# and so on...
print(group_by(edges[0]).median(values))

#%%

# st = time()
# n_X = np.max([ash._fit_results[i_shift][0].shape[0] for i_shift in range(ash.q)])

# ash._X_digit_uniques = np.zeros((ash.q, n_X, ash._d))
# ash._P = np.zeros((ash.q, n_X))
# for i_shift in range(ash.q):
#     n_X_i = ash._fit_results[i_shift][0].shape[0]
    
#     ash._X_digit_uniques[i_shift, :n_X_i, :] = ash._fit_results[i_shift][0]
#     ash._P[i_shift, :n_X_i] = ash._fit_results[i_shift][1]

# ash._X_digit_uniques = ash._X_digit_uniques.astype(np.int32)

# print('self process exec time :', time()-st)

# print(len(ash._ret))
# f = ash.predict(X_grid)

# I = np.sum(f) * np.product(X_grid.max(axis=0)-X_grid.min(axis=0)) / X_grid.shape[0]
# print(I)

# st = time()
# f_grid = ash.predict(X_grid)
# print('grid exec time :', time()-st)

#%%
X_eval = np.vstack((np.ones(100)*0.5,
                        np.linspace(0,2,100))).T

st = time()
f_eval = ash.predict(X_eval)
print('eval exec time :', time()-st)

plt.plot(Y[:,1], pdf_Y)
plt.plot(Y[:,1], f_eval)
plt.show()

#%%
from sklearn.neighbors import KernelDensity
kde = KernelDensity(kernel='gaussian', bandwidth=ash._h / 2.576).fit(X)

st = time()
f_eval_kde = np.exp(kde.score_samples(X_eval))
print('eval exec time :', time()-st)

plt.plot(Y[:,1], pdf_Y)
plt.plot(Y[:,1], f_eval)
plt.plot(Y[:,1], f_eval_kde)
plt.show()
