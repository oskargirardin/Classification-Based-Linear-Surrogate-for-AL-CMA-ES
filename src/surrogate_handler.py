import warnings
from collections import deque

import cma
import numpy as np
from scipy.linalg import null_space

from .linesearch import (
    get_bounds_opposite_feasibity,
    interval_halving,
    interval_halving2,
)
from .utils import (
    as_arr,
    binarize,
    get_mahalanobis_distance,
    get_null_space,
    get_projection_point_on_plane,
    mahalanobis_projection,
    sign,
)


class Hyperplane(object):
    def __init__(self, dim) -> None:
        self.dim = dim
        self.rcond = None
        self.coef_ = None
        self.intercept_ = None
        self.old_point_array = None
        self._data_array = np.empty((self.dim, self.dim + 2))  # for each point we have two distances we store
        self._data_array[:] = np.nan

    @property
    def n_points(self):
        return self.dim - np.sum(np.sum(np.isnan(self.points_array), axis=1) > 0)

    @property
    def is_full(self):
        return self.n_points >= self.dim

    @property
    def points_array(self):
        return self._data_array[:, :self.dim]

    @property
    def euclidean_distances(self):
        return self._data_array[:, self.dim]

    @property
    def mahalanobis_distances(self):
        return self._data_array[:, self.dim+1]

    @property
    def mean_points(self):
        return np.mean(self.points_array, axis=0)

    @property
    def parameters(self):
        return self.coef_, self.intercept_

    @property
    def full_rank(self):
        if not self.is_full:
            warnings.warn("Point matrix is not full, therefore rank < dim", stacklevel=1)
        return np.linalg.matrix_rank(self.points_array) == self.dim

    @property
    def diversity_measure(self):
        """
        Ratio between the smallest and largest singular value of X.
        If this is close to zero, the points in the hyperplane lie on an underlying n-2 dimensional space
        """
        if not self.is_full:
            return np.nan
        _, s, _ = np.linalg.svd(self.points_array - self.mean_points)
        return s[-2] / s[0]

    @property
    def largest_singular_value(self):
        """
        largest singular value of X
        """
        if not self.is_full:
            return np.nan
        _, s, _ = np.linalg.svd(self.points_array - self.mean_points)
        return s[0]

    @property
    def singular_values(self):
        """singular values of X - x-bar"""
        if not self.is_full:
            return [np.nan] * self.dim
        _, s, _ = np.linalg.svd(self.points_array - self.mean_points)
        return s

    @property
    def stds_in_coordinates(self):
        """variance of points in the coordinate axes"""
        if not self.is_full:
            return [np.nan] * self.dim
        cov = np.cov(self.points_array, rowvar=False)
        return np.sqrt(np.diag(cov))

    @property
    def empirical_covariance(self):
        X = self.points_array - self.mean_points
        return X @ X.T


    def set_parameters(self, coef, intercept):
        self.coef_ = np.asarray(coef)
        if isinstance(intercept, np.ndarray):
            intercept = intercept.item()
        self.intercept_ = intercept


    def add_point(self, new_point, euclidean_dist=None, mahalanobis_dist=None, check_distance=False):
        self.old_point_array = self.points_array.copy()
        if check_distance:
            dist = np.linalg.norm(new_point - self.points_array, axis = 1)
            if np.any(dist < 1e-14):
                raise Exception(f"Point to add {new_point} is too close to another point")

        # self.points_on_hyperplane.append(new_point.squeeze())
        self._data_array[:-1] = self._data_array[1:]
        self._data_array[-1] = np.r_[new_point.squeeze(), euclidean_dist, mahalanobis_dist]

        assert self._data_array.shape[0] == self.dim, "self._data_array.shape[0] should be dim"


    def fit(self, norm = None):
        # TODO: change that if new point causes dim of nullspace to be 2, reject that point
        X = self.points_array
        # print(X)
        assert np.sum(np.isnan(X)) == 0, "No nan values are allowed in hyperplane"
        # trunk-ignore(bandit/B101)
        assert X.ndim == 2
        if X.shape[0] != X.shape[1]:
            raise ValueError

        mean_hyperplane = np.mean(X, axis = 0)
        coef_ = get_null_space(X - mean_hyperplane)

        if coef_.size != self.dim:
            raise Exception(f"Null space of dimension ({coef_.size // self.dim})")


        self.coef_ = coef_.reshape(-1)
        self.intercept_ = - np.dot(mean_hyperplane, self.coef_)

        if norm:
            self.coef_ /= np.linalg.norm(self.coef_) / norm
            self.intercept_ /= np.linalg.norm(self.coef_) / norm

        return self.coef_, self.intercept_

    def euclidean_projection_onto(self, x):
        coef, intercept = self.parameters
        return get_projection_point_on_plane(x, coef, intercept)

    def mahalanobis_projection_onto(self, x, C, C_is_inv = False):
        coef, intercept = self.parameters
        return mahalanobis_projection(x, C, coef, intercept)[0]




class SurrogateSettings(cma.utilities.utils.DefaultSettings):
    max_iters_interval_halving = 6
    epsilon_interval_halving = 1e-11
    max_interval_length = 1e-14
    max_iters_good_bounds = 200
    norm_nabla_ghat = 1
    keep_point_if_linesearch_fail = 'mean'
    n_iters_delay_update = np.inf
    update_method = 'always'
    epsilon_zero = 1e-2
    linesearch_termination_criterion = 4
    add_gb_iters_for_ih = True
    initial_linesearch_point_factor = 2

    def _checking(self):
        if self.keep_point_if_linesearch_fail not in ['mean', 'projection']:
            raise NotImplementedError
        if self.update_method not in ['feasible_mean_correct', 'always', 'model_correct']:
            raise NotImplementedError
        if self.norm_nabla_ghat <= 0:
            raise ValueError('norm_nabla_ghat must be positive')
        return self

class MahalanobisMeanSurrogate(object):

    def __init__(self,
                 dim,
                 constraint,
                 constraint_id,
                 es,
                 optim,
                 logging,
                 max_iters_interval_halving=None,
                 epsilon_interval_halving=None,
                 max_interval_length=None,
                 max_iters_good_bounds=None,
                 norm_nabla_ghat=None,
                 keep_point_if_linesearch_fail=None,
                 n_iters_delay_update=None,
                 update_method=None,
                 epsilon_zero=None,
                 linesearch_termination_criterion=None,
                 add_gb_iters_for_ih=None,
                 initial_linesearch_point_factor=None
                 ):

        self.dim = dim
        self.constraint_id = constraint_id
        self.binary_constraint = lambda x: sign(constraint(x, id=self.constraint_id))
        self.binary_constraint_no_count = lambda x: sign(constraint(x, id=self.constraint_id, count=False))
        self.es = es
        self.optim = optim
        self.length_point_archive = 4*self.dim
        self.settings = SurrogateSettings(locals(), 12, self)._checking()

        self.hyperplane = Hyperplane(dim)
        self.nabla_ghat0 = np.random.random(dim)
        self.intercept0 = np.random.random()
        self.nabla_ghat = self.nabla_ghat0.copy()
        self.intercept = self.intercept0
        self.n_updates = 0
        self._initialized = False
        self.feasible_archive = deque(maxlen=self.length_point_archive)
        self.infeasible_archive = deque(maxlen=self.length_point_archive)
        self.iteration_info = {}
        self.ls_iterations = 0
        self.iterations_good_bounds = 0

        self.history_horizon = 200
        self.nabla_ghat_history = deque(maxlen=self.history_horizon)
        self.intercept_history = deque(maxlen=self.history_horizon)
        self.mean_history = deque(maxlen=2)
        self.mean_history.append(self.es.mean)
        self.ls_iterations_history = []
        self.ls0_history = []
        self.count_iters_no_update = 0


    def save_iteration_info(self, es, X):
        """Reads a cma.EvolutionStrategy instance and extracts relevant information from it"""
        mean_pheno = es.gp.pheno(es.mean.copy())
        mean_g_val = self.binary_constraint(mean_pheno)
        self.iteration_info = {
            'mean': mean_pheno,
            'C': es.C.copy(),
            'C_half': es.sm.to_linear_transformation(),
            'transform_inverse': lambda x: es.sm.transform_inverse(es.gp.geno(x) / es.sigma_vec.scaling.copy()),
            'mahalanobis_norm': lambda x: es.mahalanobis_norm(x),
            'sigma': es.sigma.copy(),
            'mean_g_val': mean_g_val,
            'mean_feasible': mean_g_val <= 0,
        }
        self.add_to_archive(self.iteration_info['mean'], self.iteration_info['mean_g_val'])
        self.mean_history.append(mean_pheno)

    def reset_iteration_info(self):
        self.iteration_info = {}

    def __call__(self, x):
        if not self._initialized:
            return np.nan

        return np.dot(x, self.nabla_ghat) + self.intercept

    @property
    def initialized(self):
        return self._initialized

    @property
    def func(self):
        return lambda x: self(x)

    @property
    def state(self):
        attributes_in_state = [
            'nabla_ghat',
            'intercept',
            'n_updates'
        ]
        state = {
            attribute: getattr(self, attribute) for attribute in attributes_in_state
        }
        return state

    @property
    def n_feasible_in_history(self):
        return len(self.feasible_archive)

    @property
    def n_infeasible_in_history(self):
        return len(self.infeasible_archive)

    @property
    def n_points_in_history(self):
        return self.n_feasible_in_history + self.n_infeasible_in_history

    @property
    def _two_classes_in_history(self):
        return self.n_feasible_in_history*self.n_infeasible_in_history > 0

    @property
    def previous_model_coefficients(self):
        if len(self.nabla_ghat_history) < 2:
            return np.array([np.nan]*self.dim), np.nan
        return self.nabla_ghat_history[-2], self.intercept_history[-2]


    def predict_feasibility(self, x):
        return 1 if self(x) > 0 else -1

    def get_sorted_history(self, reference_point: np.ndarray, feasible: bool=True, distance: str='euclidean'):
        history_to_sort = self.feasible_archive if feasible else self.infeasible_archive
        if distance == 'euclidean':
            res = sorted(history_to_sort, key=lambda y: np.dot(y - reference_point, y - reference_point))
        elif distance == 'mahalanobis':
            C = self.iteration_info['C']
            res = sorted(history_to_sort, key=lambda y: get_mahalanobis_distance(reference_point, y, C))
        else:
            raise NotImplementedError
        return res


    def prediction_mean_is_correct(self):
        mean = self.iteration_info['mean']
        prediction_mean = self.predict_feasibility(mean)
        true_feasibility = self.iteration_info['mean_g_val']
        return prediction_mean * true_feasibility > 0


    def _should_update(self):
        if self.settings.update_method == 'always':
            return True
        elif self.settings.update_method == 'feasible_mean_correct':
            if self.iteration_info['mean_feasible'] and self.prediction_mean_is_correct():
                return False
        elif self.settings.update_method == 'model_correct':
            if self.prediction_mean_is_correct():
                return False
        else:
            raise NotImplementedError

        return True

    def get_initial_points_linesearch(self, mean, mean_projection, eps):
        # eps = max(eps, 1e-15)
        norm_function = np.linalg.norm
        v = (mean_projection - mean) / norm_function(mean_projection - mean)
        fac = self.settings.initial_linesearch_point_factor
        return mean_projection - fac * eps * v, mean_projection + fac * eps * v

    def _compute_distances(self, a, b):
        return np.linalg.norm(a - b), self.iteration_info['mahalanobis_norm'](a - b)

    def _get_initial_distance_linesearch(self):
        last_ls_euclidean_dist = self.hyperplane.euclidean_distances[-1]
        second_last_eucledian_dist = np.max(self.hyperplane.euclidean_distances[1:])
        if np.any(np.isnan(self.hyperplane.euclidean_distances)):
            last_ls_euclidean_dist = 1e0

        return second_last_eucledian_dist


    def update(self, X=None, g=None):
        mean = self.iteration_info['mean']
        if not self._should_update():
            self.ls_iterations = 0
            self.ls_iterations_history.append(0)
            self.iterations_good_bounds = 0
            self._update_coefficients_history(self.nabla_ghat, self.intercept)
            return

        mean_projection, _ = mahalanobis_projection(mean, self.iteration_info['C'], self.nabla_ghat, self.intercept)
        if self.n_updates >= self.dim: # TODO: creates spike in GB iterations in the first iteration where the condition is verified
            last_ls_euclidean_dist = self._get_initial_distance_linesearch()
            a_linesearch, b_linesearch = self.get_initial_points_linesearch(mean, mean_projection, eps=last_ls_euclidean_dist)

        else:
            a_linesearch, b_linesearch = mean, mean_projection
        # Line search
        try:
            point_on_boundary, res_interval_halving, iters_good_bounds = self._linesearch(a_linesearch, b_linesearch)
        except RuntimeError:
            self._update_coefficients_history(self.nabla_ghat, self.intercept)
            self.ls_iterations = 0
            self.iterations_good_bounds = self.settings.max_iters_good_bounds
            self.ls_iterations_history.append(self.ls_iterations)
            self.n_updates += 1
            self.count_iters_no_update = 0
            return

        euclidean_dist, mahal_dist = self._compute_distances(res_interval_halving['a'], res_interval_halving['b'])
        self.hyperplane.add_point(point_on_boundary, euclidean_dist=euclidean_dist, mahalanobis_dist=mahal_dist)
        nabla_ghat, intercept = self.hyperplane.fit(norm=self.settings.norm_nabla_ghat)
        self.nabla_ghat, self.intercept = self._correct_sign(nabla_ghat, intercept)
        self._update_coefficients_history(self.nabla_ghat, self.intercept)
        self.ls_iterations = res_interval_halving['niter']
        self.iterations_good_bounds = iters_good_bounds
        self.ls_iterations_history.append(self.ls_iterations)
        self.n_updates += 1
        self.count_iters_no_update = 0


    def get_mahalanobis_dist_to_mean(self, vec):
        """For the current sampling distribution, compute the mahalanobis distance to the mean"""
        sigma, mean = self.iteration_info['sigma'], self.iteration_info['mean']
        C_minus_half = self.iteration_info['C_minus_half']
        dist = np.sqrt(np.dot(C_minus_half, vec - mean).dot(np.dot(C_minus_half, vec - mean))) / sigma
        return dist


    def get_norm_in_sampling_distribution(self, vec):
        """For the current sampling distribution, compute the following norm:
            sqrt( vec.T @ C_minus_one @ vec ) / sigma
        """
        sigma = self.iteration_info['sigma']
        transform_inverse = self.iteration_info['transform_inverse'] # es.sm.to_linear_transf_inv()
        vec_inv_transformed = transform_inverse(vec)
        return np.sqrt(np.dot(vec_inv_transformed, vec_inv_transformed)) / sigma



    def _interval_halving_termination_callback(self, a, b) -> bool:
        ls_crit = self.settings.linesearch_termination_criterion
        mahalanobis_norm = self.iteration_info['mahalanobis_norm']
        C_sigma_half = self.iteration_info['C_half'] * self.iteration_info['sigma']
        interval_norm_mahal = mahalanobis_norm(a - b)
        sample_std_ls_dir = np.linalg.norm(C_sigma_half @ (a - b)) / np.linalg.norm(a - b)
        if ls_crit == 0:
            return interval_norm_mahal <= self.settings.epsilon_interval_halving
        elif ls_crit == 1:
            return interval_norm_mahal <= self.settings.epsilon_zero * sample_std_ls_dir
        elif ls_crit == 2:
            return np.linalg.norm(a - b) <= self.settings.epsilon_zero * sample_std_ls_dir
        elif ls_crit == 3:
            return np.linalg.norm(a - b) <= self.settings.epsilon_zero * sample_std_ls_dir ** 2
        elif ls_crit == 4:
            return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b)) <= 1e-15
        elif ls_crit == 5:
            return np.linalg.norm(a - b) <= 1e-15
        else:
            return False


    def _linesearch(self, x, y, feasibility_x: int = None, feasibility_y: int = None):
        # g-calls = 1 (mean) + 2 (a, b) + 1 (IH assert) + 1 (midpoint) + 1 (if iters_GB > 0)
        pre_ls = self.optim.constraints.n_calls
        if not feasibility_x:
            feasibility_x = self.binary_constraint(x)
        if not feasibility_y:
            feasibility_y = self.binary_constraint(y)

        iters_search_opp_feas = 0
        if feasibility_x*feasibility_y > 0: # same feasibility
            a, b, iters_search_opp_feas = get_bounds_opposite_feasibity(x, y, self.binary_constraint, feasibility_a0=feasibility_x, feasibility_b0=feasibility_y, max_iters=self.settings.max_iters_good_bounds)
        else:
            a, b = x.copy(), y.copy()

        self.ls0_history.append([a.copy(), b.copy()])

        max_iters_interval_halving = self.settings.max_iters_interval_halving
        if self.settings.add_gb_iters_for_ih:
            max_iters_interval_halving += iters_search_opp_feas

        res_interval_halving = interval_halving2(a,
                                                 b,
                                                 self.binary_constraint,
                                                 target_interval_length=self.settings.epsilon_interval_halving,
                                                 max_iters=max_iters_interval_halving,
                                                 a0_feasibility=feasibility_x if iters_search_opp_feas == 0 else None,
                                                 norm_func=self.iteration_info['mahalanobis_norm'],
                                                 termination_callback=self._interval_halving_termination_callback)


        a_boundary, feasible_a = res_interval_halving['a'], res_interval_halving['feasible_a']
        b_boundary, feasible_b = res_interval_halving['b'], res_interval_halving['feasible_b']
        assert feasible_a * feasible_b < 0, "interval halving produced points of equal feasibility"


        midpoint = (a_boundary + b_boundary)/2
        feasibility_midpoint = self.binary_constraint(midpoint)
        self.add_to_archive(midpoint, feasibility_midpoint)

        post_ls = self.optim.constraints.n_calls

        return midpoint, res_interval_halving, iters_search_opp_feas


    def _correct_sign(self, nabla_ghat, intercept):
        """Flips the sign of the coefficients by making the prediction of the mean perfect

        Args:
            nabla_ghat (nd.array): coefficients that need to be flipped
            intercept (float): intercept that needs to be flipped

        Returns:
            tuple: nabla_ghat, intercept with correct sign
        """
        nabla_ghat_old, intercept_old = nabla_ghat.copy(), intercept
        mean = self.iteration_info['mean'] # alias
        feasible_mean = self.iteration_info['mean_feasible']
        raw_pred = np.dot(nabla_ghat, mean) + intercept
        predicted_feasible = raw_pred <= 0
        if feasible_mean != predicted_feasible: # wrong sign -> flip it
            nabla_ghat, intercept = -nabla_ghat, -intercept

        # nabla_ghat_old, intercept_old = nabla_ghat.copy(), intercept
        # predictions_feas_archive = np.asarray([self.predict_feasibility(x) for x in self.feasible_archive])
        # predictions_infeas_archive = np.asarray([self.predict_feasibility(x) for x in self.infeasible_archive])
        # num_correct_predictions = np.sum(predictions_feas_archive <= 0) + np.sum(predictions_infeas_archive > 0)
        # threshold = self.n_points_in_history // 2
        # if nabla_ghat[0] < 0:
        #     raise
        # if num_correct_predictions < threshold:
        #     raise
        #     return -nabla_ghat_old, -intercept_old
        # else:
        #     return nabla_ghat_old, intercept_old

        if nabla_ghat[1] < 0:
            if 11 < 3:
                print("*"*30)
                print('Iteration', self.es.countiter)
                print("Initial nabla_ghat:", nabla_ghat_old)
                print("Initial intercept:", intercept_old)
                print("mean:", mean)
                print("raw_pred", raw_pred)
                print('Prediction model:', predicted_feasible)
                print('Truth:', feasible_mean)
                print("Final nabla_ghat:", nabla_ghat)
                print("Final intercept:", intercept)
                raise
        return nabla_ghat, intercept

    def add_to_archive(self, x, g=None):
        if g is None:
            g = self.binary_constraint(x)
        if g > 0:
            for y in self.infeasible_archive:
                dist = np.linalg.norm(x - y)
                if dist < 1e-14:
                    raise Exception(f"Point to add {x} is too close to {y}")
            self.infeasible_archive.append(x)
        else:
            for y in self.feasible_archive:
                dist = np.linalg.norm(x - y)
                if dist < 1e-14:
                    raise Exception(f"Point to add {x} is too close to {y}")

            self.feasible_archive.append(x)

    def update_archives(self, X, g):
        for x, feasibility in zip(X, g):
            self.add_to_archive(x, feasibility)

    def _num_false_predictions(self, X: np.ndarray, g: np.ndarray):
        feasibility = binarize(np.asarray(g), class_two_val=0)
        feasibility_predicted = binarize(np.asarray([self(x) for x in X]), class_two_val=0)
        false_predictions = np.logical_xor(feasibility, feasibility_predicted).astype(np.int64)
        return np.sum(false_predictions)

    def get_percentage_correct_predictions(self, X: np.ndarray, use_old_model=False):
        N = len(X)
        if use_old_model:
            nghat_old, intercept_old = self.previous_model_coefficients
            def old_model(x):
                return np.dot(nghat_old, x) + intercept_old
            predicted_feasibility = np.asarray([sign(old_model(x)) for x in X])
        else:
            predicted_feasibility = np.asarray([self.predict_feasibility(x) for x in X])

        true_feasibility = np.asarray([sign(self.binary_constraint_no_count(x)) for x in X]) # falsifies g-calls
        # true_feasibility = np.asarray([np.nan for x in X])
        n_correct_pred = np.sum((predicted_feasibility * true_feasibility) > 0)
        return n_correct_pred / N

    def can_initialize(self):
        return not self.iteration_info['mean_feasible']

    def initialize(self):
        mean, mean_feasible = self.iteration_info['mean'], self.iteration_info['mean_feasible']
        assert not mean_feasible
        prev_mean = self.mean_history[-2]
        self.nabla_ghat = mean - prev_mean
        boundary_point, res, iters = self._linesearch(mean,
                                                      prev_mean,
                                                      feasibility_x=1,
                                                      feasibility_y=-1)

        self.intercept = - np.dot(self.nabla_ghat, boundary_point)
        self.hyperplane.set_parameters(self.nabla_ghat, self.intercept)
        self._update_coefficients_history(self.nabla_ghat, self.intercept)
        self._initialized = True

        while self.hyperplane.n_points < self.dim - 1:
            x = self.es.ask(1)[0]
            projection = self.hyperplane.mahalanobis_projection_onto(x, self.iteration_info['C'])
            self.hyperplane.add_point(projection, check_distance=True)

        self.hyperplane.add_point(boundary_point)



    def _fill_archive_current_distribution(self, n_points_target: int):
        n_points_to_sample = n_points_target - self.n_points_in_history
        if n_points_to_sample < 1:
            return
        X = np.asarray(self.es.ask(n_points_to_sample))
        assert len(X) == n_points_to_sample # to be sure
        for x in X:
            self.add_to_archive(x)


    def _update_coefficients_history(self, nabla_ghat, intercept):
        self.nabla_ghat_history.append(nabla_ghat)
        self.intercept_history.append(intercept)
