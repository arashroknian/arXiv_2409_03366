import numpy as np
import scipy
from types import SimpleNamespace
trexc = scipy.linalg.lapack.dtrexc

class DKS:
    def __init__(self, matvec, n, l, m, b=1, seed=2023, tol=1e-5):
        assert callable(matvec)
        self.matvec = matvec

        self.l = l
        self.m = m
        self.b = b
        self.tol = tol

        self.S = np.zeros((m+1, m))
        Ut = np.zeros((m+1, n))
        self.U = Ut.T
        self.k = 0

        # import numba
        # print(numba.typeof(self.U))

        rng1 = np.random.default_rng(seed)
        v1 = rng1.random(n)
        v1 /= norm(b, v1)
        self.U[:,0] = v1

        self.mv_count = 0
        self.restart_count = 0

        v2 = rng1.random(n); v2 /= norm(b, v2)
        ortho(self.b, self.S, self.U, 0, v2)
        rl = RL(np.arange(10, dtype=np.float64), 3, 10)


    def itera(self, maxrestart=np.inf, maxmv=np.inf, log_freq=5):
        assert maxrestart != maxmv

        R,L = 0,0
        while 1:
            self.espandi()
            nc, eps, theta = self.errori()

            self.hist_(eps, theta)
            if log_freq > 0 and self.restart_count % log_freq == 0: self.info(nc, theta, eps, R, L)

            finito = (nc >= self.l) or (self.mv_count >= maxmv) or (self.restart_count >= maxrestart)
            if finito: break
            self.restart_count += 1

            self.k,R,L = self.riordina(theta)
            self.tronca()

    def espandi(self):
        S, U, l, m, k = self.S, self.U, self.l, self.m, self.k

        for j in range(k, m):
            u0 = U[:,j]
            self.mv_count += 1
            u = self.matvec(u0)
            ortho(self.b, S, U, j, u)

    def riordina(self, theta):
        S, U, l, m = self.S, self.U, self.l, self.m

        R,L = RL(np.real(theta), l, m)

        S_ = S[:m,:m]
        S_,Q = scipy.linalg.schur(S_)
        for i in range(R):
            theta_re = np.diag(S_)
            j = np.argmax(theta_re[i:]) + i
            if j == i: continue
            S_, Q, info = trexc(S_, Q, j+1, i+1); assert info == 0
        for i in range(R, R+L):
            theta_re = np.diag(S_)
            j = np.argmin(theta_re[i:]) + i
            if j == i: continue
            S_, Q, info = trexc(S_, Q, j+1, i+1); assert info == 0

        S[:m, :m] = S_ # = Q.T @ S[:m, :m] @ Q
        S[ m, :m] = S[m, :m] @ Q
        U[ :, :m] = U[:, :m] @ Q
        U[ m, :m] = U[m, :m] # lui non cambia

        k = R + L
        if S[k,k+1] != 0: k += 1

        return k, R, L

    def eig(self, completi=0):
        S, U, l, m = self.S, self.U, self.l, self.m
        
        S_ = S[:m,:m]
        theta,y = np.linalg.eig(S_)
        sort = np.argsort(theta)[::-1]; theta = theta[sort]; y = y[:,sort]

        if completi:
            lambda_ = theta
            x = U[:,:m] @ y
            return lambda_, x

        return theta, y

    def errori(self, completi=0):
        S, _, l, m = self.S, self.U, self.l, self.m

        if completi == 0:
            theta, y = self.eig()
            eps = np.abs(np.dot(S[m,:m], y) / theta) / np.linalg.norm(y, axis=0)
        else:
            theta, x = self.eig(completi=1)
            theta = theta[:l]; x = x[:,:l]
            assert np.allclose(np.imag(theta), 0)
            assert np.allclose(np.imag(x), 0)
            theta = np.real(theta); x = np.real(x)
            eps = np.array([ norm(self.b, self.matvec(x[:,i]) - theta[i]*x[:,i] ) / norm(self.b, theta[i]*x[:,i] ) for i in range(l) ])

        nc = 0
        for i in range(l):
            if eps[i] > self.tol: break
            nc += 1

        return nc, eps, theta

    def tronca(self):
        S, U, l, m, k = self.S, self.U, self.l, self.m, self.k

        U[:, k   ] = U[:,m]
        U[:, k+1:] = 0
        S[k   , :] = S[m,:]
        S[:   ,k:] = 0
        S[k+1:, :] = 0

        return k

    def hist_(self, eps, theta):
        if not hasattr(self, 'hist'): self.hist = { key: [] for key in ['theta', 'eps', 'mv_count'] }

        self.hist['eps'].append(eps)
        self.hist['theta'].append(theta)
        self.hist['mv_count'].append(self.mv_count)

    def info(self, nc, theta, eps, R, L):
        vals = ''.join([ f'{x:>10.3g}' for x in theta[nc:nc+3] ])
        err = ''.join([ f'{x:>10.3g}' for x in eps[nc:nc+3] ])

        print(f'[{self.restart_count:4d},{self.mv_count:5d}] R+L=k:{R:2d}+{L:2d}={self.k:2d} c:{nc:2d}' + ' ' + f'nc:[{vals}] eps:[{err}]')

from numba import njit

@njit
def dot(b, u, v): return np.dot(u*b, v)
@njit
def norm(b, u): return np.sqrt(dot(b,u,u))

@njit('void(float64[::1], float64[:,::1], float64[::1,:], int64, float64[::1])')
def ortho(b, S, U, j, u):
    # print('-', numba.typeof(b), numba.typeof(S), numba.typeof(U), numba.typeof(j), numba.typeof(u))

    for i in range(j+1):
        v = U[:,i]
        h = dot(b, u, v)
        S[i,j] = h
        u -= h*v
    for i in range(j+1): # fare almeno due giri serve se no si converge agli autovalori sbagliati (errori numerici?)
        v = U[:,i]
        h = dot(b, u, v)
        S[i,j] += h
        u -= h*v
    for i in range(j+1):
        v = U[:,i]
        h = dot(b, u, v)
        S[i,j] += h
        u -= h*v
    h = norm(b, u)
    if h < 1e-10: raise
    u /= h

    S[j+1,j] = h
    U[:,j+1] = u

@njit('UniTuple(int64,2)(float64[:], int64, int64)')
def RL(theta_re, l, m):
    combinazioni =  [ (R, L) for R in range(l, m-1) for L in range(0, m-l-1) if R+L < 2/3*m ]

    fs = {}
    f_max = 0
    rl = (-1, 1)
    for comb in combinazioni:
        (R, L) = comb; theta_p = theta_re[R]; theta_n = theta_re[m-1-L]
        if theta_p == theta_n: continue
        f = (m - L - R) * np.sqrt( (theta_re[l-1] - theta_p)/(theta_p - theta_n) )
        if f > f_max: 
            rl = comb
            f_max = f
        fs[comb] = f
    if rl[0] == -1: raise

    return rl


# pb = co.pkl(dict(cartella='../simulazioni', nome='eig_vg14_D1g'))
# A = pb.eig_ops().S
# b = pb._M.diagonal()
# print(pb._eig.vals)
# ks = KS(A, 30, b=b)
# ks.itera(500)
