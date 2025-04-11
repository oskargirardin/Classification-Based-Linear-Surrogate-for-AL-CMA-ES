import itertools
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
import cma
from copy import deepcopy
from math import ceil
import scipy
import scipy.spatial
from sklearn.linear_model import LogisticRegression
from itertools import zip_longest

""" def debug(func):
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__}() returned {repr(value)}")
        return value
    return wrapper_debug """


class History(dict):
    
    def __init__(self) -> None:
        super().__init__(self)
        
    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError from Exception

    def add(self, new_vals, extend = False):
        for key, value in new_vals.items():
            if not isinstance(value, np.ndarray):
                value = np.array([value])
            if value.ndim == 0:
                value = np.expand_dims(value, axis=0)
            if key not in self:
                new_shape = (0, *value.shape[1:]) if extend else (0, *value.shape)
                super().__setitem__(key, np.empty(shape=(new_shape)))
            value_to_concat = value if extend else np.expand_dims(value, axis=0)
            value_to_add = np.concatenate((self.__getitem__(key), value_to_concat), axis = 0)
            super().__setitem__(key, value_to_add)


def get_list_dict_permutations(d):
    """Creates permutations of dictionary values

    Usage:
    ```
    > d = {'a' : [10, 20, 50]}
    > get_list_dict_permutations(surrogate_opts)
    >>> ([{'a': 10}, {'a': 20}, {'a': 50}], ['a'])
    ```
    """
    d_not_lists = {key: val for key,val in d.items() if not isinstance(val, list)}
    d_lists = {key: val for key,val in d.items() if isinstance(val, list)}
    if not d_lists: # If no lists among items
        return [d], []
    values_are_lists = list(d_lists.keys())
    keys, values = zip(*d_lists.items())
    return [{**dict(zip(keys, v)), **d_not_lists} for v in itertools.product(*values)], values_are_lists

def convert_tuple_dict_df(d, param_columns = None, vals_columns = None):
    key_list = list(d.keys())
    n_params = len(key_list[0])
    if param_columns:
        assert n_params == len(param_columns), "Number of parameters is not equal to number of given parameter names"
    val_list = list(d.values())
    if not vals_columns:
        vals_columns = range(len(val_list[0]))
    if not param_columns:
        param_columns = range(len(key_list[0]))
    df = pd.DataFrame(key_list, columns=param_columns)
    df.loc[:,vals_columns] = val_list
    return df

def convert_dict_dict_df(d, param_columns = None):
    param_vals = list(d.keys())
    n_params = len(param_vals[0])
    if param_columns:
        assert n_params == len(param_columns), "Number of parameters is not equal to number of given parameter names"
        
    val_list = list(d.values())

    vals_columns = list(val_list[0].keys())

    if not param_columns:
        param_columns = range(len(param_vals[0]))
    df = pd.DataFrame(param_vals, columns=param_columns)
    val_content = [[res_d[key] for key in vals_columns] for res_d in val_list]
    df.loc[:,vals_columns] = val_content
    return df


def compute_stats(arr, suffix = ""):
    res = {
        "mean"+suffix: np.mean(arr),
        "med"+suffix: np.median(arr),
        "std"+suffix: np.std(arr),
        "min"+suffix: np.min(arr),
        "max"+suffix: np.max(arr)
    }
    return res



def eval_on_data(logreg, X, y):
    N = len(X)
    y_class_pred = logreg.predict(X)
    ybin = binarize(y.squeeze())
    matches = y_class_pred == ybin
    correct_preds = np.sum(matches)
    return correct_preds/N

def print_coefs_n(log):
    coef = log.coef_
    coef_n = coef/log.coef_norm_true
    print("Coefs (norm.):", coef_n)
    print("Intercept (norm.):", log.intercept_/np.linalg.norm(coef))

def binarize(y, class_two_val = 0):
    return np.where(y > 0, 1, class_two_val)

def plot_norm_sample_model(model, X_old, ybin, X_norm, res = 1000):
    intercept, coef = model.intercept_, model.coef_
    xmin = np.min([X_old[:,0], X_norm[:,0]])
    xmax = np.max([X_old[:,0], X_norm[:,0]])
    ymin = np.min([X_old[:,1], X_norm[:,1]])
    ymax = np.max([X_old[:,1], X_norm[:,1]])

    a = -coef[0]/coef[1]
    b = -intercept/coef[1]
    xrange = np.linspace(xmin, xmax, num=res)
    ymodel = [a*x+b for x in xrange]
    
    fig, ax = plt.subplots()
    marker = np.where(ybin > 0, "x", "o")
    marker = "o"
    ax.scatter(*X_old.T, label = "Raw", c="black", marker=marker)
    ax.scatter(*X_norm.T, label = "Transformed", c="red", marker = marker)
    ax.plot(xrange, ymodel, label = "model")

    ax.set_ylim(ymin*1.10, ymax*1.10)

    ax.legend()
    plt.show()

def plot_sample_with_model(coefs, intercept, X, y, true_coefs = None, true_intercept = None, res = 1000, title = "", aspect_ratio = "auto"):
    y = binarize(y)
    intercept, coef = intercept.reshape(-1), coefs.reshape(-1)
    xmin, ymin = np.min(X, axis = 0)
    xmax, ymax = np.max(X, axis = 0)

    a = -coef[0]/coef[1]
    b = -intercept/coef[1]
    xrange = np.linspace(xmin, xmax, num=res)
    ymodel = [a*x+b for x in xrange]

    
    fig, ax = plt.subplots()
    ax.scatter(*X.T, c=y)
    ax.plot(xrange, ymodel, label = "model")
    ax.set_aspect(aspect_ratio)

    # True
    if not (true_coefs is None or true_intercept is None):
        a_true = -true_coefs[0]/true_coefs[1]
        b_true = -true_intercept/true_coefs[1]
        ytrue = [a_true*x+b_true for x in xrange]
        ax.plot(xrange, ytrue, label = "true", linestyle = "dashed")


    ax.set_ylim(ymin*1.10, ymax*1.10)
    ax.legend()
    plt.title(title)
    plt.show()


def print_accuracies(model, X_train, X_test, y):
    training_acc = eval_on_data(model, X_train, y)
    testing_acc = eval_on_data(model, X_test, y)
    print("Training accuracy:", training_acc)
    print("Testing accuracy:", testing_acc)


def get_last_na_vals(arr):
    row_idxs = pd.DataFrame(arr).apply(pd.Series.last_valid_index).to_numpy()
    idxs = list(zip(row_idxs, range(arr.shape[1])))
    return np.array([arr[idx] for idx in idxs])


def linear_prediction(X, coefs, intercept):
    return binarize(linear_decision_func(X, coefs, intercept))

def linear_decision_func(X, coefs, intercept):
    if coefs.ndim > 2:
        raise Exception("Coefficents should be have 2 or less dimensions")
    coefs = coefs.reshape(-1, 1)
    raw_pred =  X @ coefs + intercept
    return raw_pred.reshape(-1)

def compute_score(coefs, intercept, X, g):
    N = len(X)
    g_hats = linear_prediction(X, coefs, intercept)
    return np.sum(g == g_hats)/N



def subset_dict(d, key_list):
    return {k:d[k] for k in key_list}


def get_setup_differences_similarities(json_content):
    d = collections.defaultdict(list)
    json_ = deepcopy(json_content)
    for single_config in json_:
        single_config.pop("result")
        for key, value in single_config.items():
            if value not in d[key]:
                d[key].append(value)
    changing_params = []
    for key, value in d.items():
        if len(value) > 1:
            changing_params.append(key)
    constant_params = list(filter(lambda x: x not in changing_params, d.keys()))
    return subset_dict(d, changing_params), subset_dict(d, constant_params)

def contains_values(d, value_dict):
    return value_dict.items() <= d.items()


def get_figsize(nrows, ncols):
    default_figsize = plt.rcParams["figure.figsize"]
    return default_figsize[0]*ncols, default_figsize[1]*nrows

def get_subplot_grid(nplots):
    default_figsize = plt.rcParams["figure.figsize"]
    if nplots < 2:
        return 1, 1, default_figsize
    ncols = 2 if nplots <= 4 else 3
    nrows = ceil(nplots / ncols)
    figsize = default_figsize[0]/2*ncols, default_figsize[1]/2*nrows
    return nrows, ncols, figsize

def estimate_eval_model_X(optim_res, print_coefs = True, standardize = True, norm_coef = 1, dim = 0):
    X_raw = optim_res.history.X
    y = binarize(optim_res.history.g[:,dim])
    log = LogisticRegression(penalty=None, max_iter=int(1e7))

    if standardize:
        optim_res.surrogate_handler.normalization_X = "estimated"
        X, C_root_inv, mean = optim_res.surrogate_handler._normalize_data(X_raw)
    else:
        X = X_raw
    log.fit(X, y)
    intercept, coef = log.intercept_, log.coef_[0]
    if standardize:
        intercept, coef = optim_res.surrogate_handler._destandardize_coefs(coef, intercept, C_root_inv, mean)
    if print_coefs:
        print("Raw coefs:", coef)
    if norm_coef > 0:
        intercept, coef = optim_res.surrogate_handler._normalize_coefs(coef, intercept, norm_coef)
    if print_coefs:
        print("Normalized coefs:", coef)
    print("Score:", compute_score(coef, intercept, X_raw, y))
    return log

def multivariate_uniform(shape_X, widths = None, **kwargs):
    if widths is None:
        widths = [1] * shape_X[1]
    X = np.empty(shape_X)
    for i, width in enumerate(widths):
        x = np.random.uniform(low = -width, high=width,size=shape_X[0])
        X[:,i] = x
    return X

def multivariate_normal(shape_X, stds = None, **kwargs):
    if stds is None:
        stds = [1] * shape_X[1]
    X = np.empty(shape_X)
    mean = [0]*shape_X[1]
    cov = np.diag(stds)**2
    X = np.random.multivariate_normal(mean, cov, size=shape_X[0])
    return X


def normalize_data(X):
    mean = np.mean(X, axis = 0)
    cov = np.cov(X.T)    
    C_inv = np.linalg.inv(cov)
    C_root_inv = scipy.linalg.sqrtm(C_inv)
    X_normed = (X - mean) @ C_root_inv
    if np.iscomplexobj(X_normed.flatten()):
        raise Exception(f"Data is complex. mean: {mean}, C: {cov}, C_inv: {C_inv}, C_root_inv: {C_root_inv}")
    return X_normed, C_root_inv, mean


def destandardize_coefs(coefs, intercept, C_root_inv, mean):
    coef_tilde = C_root_inv @ coefs.reshape(-1) 
    intercept_tilde = intercept - mean.T.dot(C_root_inv).dot(coefs.reshape(-1))
    return intercept_tilde, coef_tilde

def normalize_coefs(coefs, intercept, target_norm):
    coef_norm_true = np.linalg.norm(coefs)
    coef_norm_true = 1e-16 if coef_norm_true < 1e-16 else coef_norm_true
    if not coef_norm_true:
        print(coef_norm_true)
        raise
    
    coef_normed = coefs*target_norm/coef_norm_true
    intercept_normed = intercept*target_norm/coef_norm_true
    return intercept_normed, coef_normed


def gen_sample_normal(shape_X: tuple, stds, mean = None):
    N, dim = shape_X
    if mean is None:
        mean = np.zeros(dim)
    cov = np.diag(stds)**2
    X = np.random.multivariate_normal(mean, cov, size=shape_X[0])
    y = (X[:,0] > 0).flatten().astype(int)
    return X, y

def get_cosines(arr1, arr2):
    assert arr1.shape == arr2.shape, "arr1 and arr2 need to be of same shape"
    dot_prod = np.sum(arr1 * arr2, axis = -1)
    norm_prod = np.linalg.norm(arr1, axis = -1)*np.linalg.norm(arr2, axis = -1)
    return dot_prod/norm_prod

def compute_cosine(coef):
    dim = len(coef)
    true_coefs = np.array([1]+[0]*(dim-1))
    return get_cosines(coef, true_coefs)


def fit_logistic(X, y, coef_norm = -1):
    X_tilde, C_root_inv, mean = normalize_data(X)
    log = LogisticRegression(penalty=None, max_iter=int(1e10))
    log.fit(X_tilde, y)
    coefs, intercept = log.coef_, log.intercept_
    intercept, coefs = destandardize_coefs(coefs, intercept, C_root_inv, mean)
    if coef_norm > 0:
        intercept, coefs = normalize_coefs(coefs, intercept, coef_norm)
    return intercept, coefs

def plot_sample(X, **hist_kwargs):
    N, dim = X.shape
    nrows, ncols, figsize = get_subplot_grid(dim)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = axs.reshape(-1)
    for i in range(dim):
        axs[i].hist(X[:,i], **hist_kwargs)
        axs[i].set_title(f"Dim {i}")
    plt.show()


def projection_v_on_u(v, u):
    assert u.shape == v.shape, "u and v need to be of same shape"
    if np.linalg.norm(u) > 1+1e-2:
        print("Warning norm(u) > 1+1e-2")
    numerator = u.dot(v)
    norm_u = u.dot(u)
    return numerator*u/norm_u

def gramschmidt(V):
    """Constructs an orthogonal matrix from a matrix V. The first column will be stay untouched, the others are normalized"""
    n, k = V.shape
    U = np.zeros((n, k))
    u1 = V[:,0] / np.linalg.norm(V[:,0])
    U[:,0] = u1
    for i in range(1, k):
        U[:,i] = V[:,i]
        for j in range(i):
            U[:,i] = U[:,i] - projection_v_on_u(U[:,i], U[:,j])
        U[:,i] = U[:,i]/np.linalg.norm(U[:,i])
    return U

def construct_V(v1):
    """Constructs a random matrix with v1 in it as the first column"""
    v1 = v1.squeeze()
    d = len(v1)
    V = np.random.random((d, d-1))
    V = np.c_[v1, V]
    return V


def construct_cov_mat(grad_g, var_g_dir, var_other_dirs = 1):
    grad_g = grad_g.squeeze()
    dim = grad_g.shape[0]
    V = construct_V(grad_g)
    #print("V",V, sep="\n")
    U = gramschmidt(V)
    #print("U",U, sep="\n")
    #print("U.T",U.T, sep="\n")
    #print("U.T @ U:", U.T @ U, sep="\n")
    L = np.diag([var_g_dir] + [var_other_dirs]*(dim-1))
    #print("L",L, sep="\n")
    return U @ L @ U.T, U

def construct_basis(v1, normalized = False):
    """Constructs an orthogonal matrix with v1 (normalized) as the first vector"""
    if normalized:
        v1 = v1/np.linalg.norm(v1)
    random_matrix = construct_V(v1)
    orthogonal_basis = gramschmidt(random_matrix)
    return orthogonal_basis



def get_projection_point_on_plane(x, normal_vec_plane, alpha = 0):
    norm_n = np.linalg.norm(normal_vec_plane)
    a_num = x.dot(normal_vec_plane) + alpha
    a_denum = norm_n**2
    projection = x - normal_vec_plane*a_num/a_denum
    return projection


def transpose_inhomogenous_list(l):
    return np.array(list((zip_longest(*l, fillvalue=np.nan))))

def get_median_path(result_list):
    list_transposed_filled_na = transpose_inhomogenous_list(result_list)
    median_path = np.nanmedian(list_transposed_filled_na, axis = 1)
    return median_path

def normalize_vector(vec: np.ndarray):
    norm = np.linalg.norm(vec)
    return vec / norm

def get_ranks(arr: np.ndarray, increasing = True):
    """Returns ranks in increasing order (from index of smallest element to index of largest index). If `increasing = False` the order is inverted."""
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    ranks = arr.argsort()

    if increasing:
        return ranks
    else:
        return ranks[::-1]

def get_mahalanobis_distance(v1, v2, C, C_is_inv = False):
    if not C_is_inv:
        C_inv = np.linalg.inv(C)
    else:
        C_inv = C
    v1 = v1.squeeze()
    v2 = v2.squeeze()
    res = np.sqrt((v1 - v2).T @ C_inv @ (v1 - v2))
    return res.squeeze()


def as_arr(obj):
    """Converts object to `np.ndarray` if needed and possible"""
    if isinstance(obj, np.ndarray):
        return obj
    else:
        return np.asarray(obj)


def mahalanobis_projection(x, C, beta, alpha = 0):
    """Takes point x and projects it onto the hyperplane given by normal (eucledian sense) vector beta by minimizing the mahalanobis distance"""
    # C_inv = np.linalg.inv(C)
    x = x.reshape(-1, 1)
    beta = beta.reshape(-1, 1)
    beta_hat = C @ beta
    num = x.T @ beta + alpha
    denum = beta.T @ beta_hat
    constant = num/denum
    projection =  x - constant * beta_hat
    if np.any(np.isnan(projection)):
        warnings.warn("Projection is nan", stacklevel=1)
    dist = get_mahalanobis_distance(x, projection, C)
    return projection.squeeze(), dist #projection.squeeze(), dist

def sign(x):
    return -1 if x <= 0 else 1


def get_null_space(A):
    u, s, vh = scipy.linalg.svd(A, full_matrices=True)
    Q = vh[-1:,:].T.conj()
    return Q


def find_feasible_x0(x0, constraint):
    def constraint_aggregated(x):
        return np.max(constraint(x))
    xfeasible, es = cma.fmin2(constraint_aggregated, x0, sigma0=1, options={'ftarget': 0, 'verbose': -9})
    if 'ftarget' not in es.result.stop.keys():
        raise Exception('No feasible x0 found')
    return xfeasible