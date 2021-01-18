###################################################################3###
#                    Nonnegative Matrix Factorization
#       Tailored for Methylation Deconvolution of multiple samples
#    with additional features such as fixing some of the columns in W
#                   Convention across the code: WH=V
#######################################################################

import os, inspect
from functools import wraps
import numpy as np
import pandas as pd
import scipy.optimize as OP
import numpy.linalg as LA
from cvxopt import matrix, solvers
import warnings, sys
import argparse
warnings.filterwarnings("ignore", category=DeprecationWarning)

def init_d(func):
    """
    Automatically assigns the parameters.
    """
    names, varargs, keywords, defaults = inspect.getargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper


class CF_NMF:
    """
    a matrix factorization class designed to solve the cf-DNA deconvolution problem
    """

    @init_d
    def __init__(self, V, W=None, H=None, new_cols=0, w_init=np.empty(0), h_init=np.empty(0), iters=50, runs=5, tol=1e-3, base_num=5, err_ord='fro'):
        np.set_printoptions(suppress=True)
        warnings.simplefilter(action='ignore', category=FutureWarning)
        solvers.options['show_progress'] = False
        self.W_G, self.W_h, self.W_A, self.W_b = np.full(4, None)
        self.history = np.empty(0)
        self.base_num = W.shape[1] + new_cols if W is not None else new_cols
        self.base_num = H.shape[0] if not self.base_num and H is not None else self.base_num  # if only H is provided

        # self.a, self.b = np.array(w_init.split(',')).astype(int)
        self.init_asserts()

    def init_asserts(self):
        assert self.base_num  # check the number of overall tissues is larger than 0
        assert self.W.shape[0] == self.V.shape[0] if self.W is not None else True
        assert self.base_num == self.H.shape[0] if self.H is not None else True
        assert self.H.shape[1] == self.V.shape[1] if self.H is not None else True

    def init_constraints(self):
        N = self.base_num
        self.W_A = matrix(np.zeros((N, N)))
        self.W_b = matrix(np.zeros((N, 1)))
        pos = np.diag(np.ones(N))
        neg = np.diag(-np.ones(N))
        self.W_G = matrix(np.vstack((pos, neg)))
        self.W_h = matrix(np.hstack((np.ones(N), np.zeros(N))))

    def init_W(self):
        M, K = self.V.shape[0], self.new_cols if self.W is not None else self.base_num
        W = self.w_init[0](self.w_init[1], self.w_init[2], M * K).reshape((M, K))
        # W = np.random.beta(self.a, self.b, M * K).reshape((M, K))
        return np.hstack((self.W, W)) if self.W is not None else W

    def init_H(self):
        K, N = self.base_num, self.V.shape[1]
        H = self.h_init[0](self.h_init[1], self.h_init[2], K * N).reshape((K, N))
        # H = np.random.beta(self.a, self.b, K * N).reshape((K, N))
        H /= np.sum(H, axis=0)
        return H if self.H is None else self.H

    def culc_H_given_W(self, W):
        H = np.apply_along_axis(self.culc_coeff_given_W, 0, self.V, W)
        return H

    def culc_coeff_given_W(self, v, W):
        """
        using scipy's nnls enforces the output to be non-negative but does not ensured ||x||_1 = 1.
        To enforce this constrain we add a row of ones to A and an additional element for b so that 1^T x = 1
        """
        A = np.vstack((W, np.ones(W.shape[1])))
        b = np.vstack((v.reshape((v.shape[0], 1)), [1]))
        h = OP.nnls(A, b.T[0])[0]
        # return h[:-1].reshape(h.shape[0] - 1, )
        return h

    def culc_W_given_H(self, H):
        P = matrix(H @ H.T)
        W = np.apply_along_axis(self.culc_meth_line_given_H, 1, self.V, H, P)
        return W

    def culc_meth_line_given_H(self, v, H, P):
        q = matrix(-v @ H.T)
        # solvers.options['kktreg'] = 1e-6
        w = solvers.qp(P, q, self.W_G, self.W_h, kktsolver='ldl2')['x']  # self.W_A, self.W_b,kktsolver='ldl'
        return np.array(w.T)[0]

    def error(self, mid=False, W=None, H=None):
        if mid:
            return LA.norm(self.V - W @ H, ord=self.err_ord)  # / np.sqrt(self.V.size)
        return LA.norm(self.V - self.W @ self.H, ord=self.err_ord)  # / np.sqrt(self.V.size)

    def get_fresh_wh(self):
        if self.W is None:
            H = self.init_H()
            W = self.culc_W_given_H(H)
        else:
            W = self.init_W()
            H = self.culc_H_given_W(W)
        return W, H

    def factorize(self):
        self.init_constraints()
        best_W = None
        best_H = None
        best_history = None
        for r in range(self.runs):
            W, H = self.get_fresh_wh()
            prev_err = self.error(True, W, H)
            history = np.array([prev_err])
            for i in range(self.iters):
                W = self.culc_W_given_H(H)
                H = self.culc_H_given_W(W)
                curr_error = self.error(True, W, H)
                history = np.append(history, curr_error)
                diff_err = np.abs(curr_error - prev_err)
                prev_err = curr_error
                if diff_err < self.tol:
                    break
            if r == 0 or history[-1] < best_history[-1]:
                best_H, best_W, best_history, best_r = [H, W, history, r]
        self.history = best_history
        self.W = best_W
        self.H = best_H

    def save_result(self):
        pd.DataFrame(self.W).round(3).to_csv('W.tsv', header=False, index=False, sep='\t')
        pd.DataFrame(self.H).round(3).to_csv('H.tsv', header=False, index=False, sep='\t')
        pd.DataFrame(self.history).round(3).to_csv('history.csv', header=False, index=False)


def load_file(path):
    seps = {'.csv': ',', '.tsv': '\t'}
    _, ext = os.path.splitext(path)
    return pd.read_csv(path, sep=seps[ext], header=None).values


def load_files(paths):
    fs = [load_file(path) if path else None for path in paths]
    return fs


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('V', help='path to .tsv or .csv file for V matrix. Assumes no header exists.')
    parser.add_argument('-W', help='path to .tsv or .csv file for W matrix. Can be partial. Assumes no header exists.', default='', type=str)
    parser.add_argument('-H', help='path to .tsv or .csv file for H matrix. Assumes no header exists.', default='', type=str)
    parser.add_argument('-c', '--free_w_cols', help='number of free columns to add to W. Default: 0', default=0, type=int)
    parser.add_argument('-iw', '--init_w',
                        help='Comma separated string, stating the type of distribution (first argument) and parameters '
                             '(second and so on) for W initialization. default="normal,0,1" \n '
                             'Valid distribution values: normal,beta.'
                        , default='beta,70,100', type=str)
    parser.add_argument('-ih', '--init_h',
                        help='Comma separated string, stating the type of distribution (first argument) and parameters '
                             '(second and so on) for H initialization. default="normal,0,1" \n '
                             'Valid distribution values: normal,beta.'
                        , default='beta,70,100', type=str)
    parser.add_argument('-t', '--iter_num', help='number of iteration for the algorithm. Default: 10', default=10, type=int)
    parser.add_argument('-r', '--reps', help='number of Repetitions for the algorithm. Default: 5', default=5, type=int)
    parser.add_argument('-o', '--tol', help='Tolerance Parameter. Default: 1e-5', default=1e-5, type=float)

    args = parser.parse_args()
    return args


def check_args_validity(args):
    if args.W == '' and args.H == '':
        assert args.free_w_cols
    assert args.W == '' or args.H == ''


def parse_init_param(iparams):
    vals = iparams.split(',')
    if vals[0] == 'normal':
        ifunc = np.random.normal
    elif vals[0] == 'beta':
        ifunc = np.random.beta
    else:
        print('Distribution not supported. Please insert normal or beta')
        sys.exit()
    try:
        p1, p2 = float(vals[1]), float(vals[2])
    except ValueError:
        print('Initial parameters not representing floats/ints. please try again.')
        sys.exit()
    return np.array([ifunc, p1, p2])


if __name__ == '__main__':
    args = parse()
    check_args_validity(args)
    V, W, H = load_files([args.V, args.W, args.H])
    wi_params, hi_params = parse_init_param(args.init_w), parse_init_param(args.init_h)
    c_nmf = CF_NMF(V, W=W, H=H, new_cols=args.free_w_cols, w_init=wi_params, h_init=hi_params, iters=args.iter_num, runs=args.reps, tol=args.tol)
    c_nmf.factorize()
    c_nmf.save_result()
