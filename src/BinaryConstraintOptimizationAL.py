import warnings
import numpy as np
import cma
from .utils import find_feasible_x0
from .surrogate_handler import MahalanobisMeanSurrogate




class BinaryConstraintOptimizationAL(object):
    def __init__(
        self,
        fun,
        constraints,
        x0,
        sigma0,
        surrogate_opts=None,
        cma_opts=None,
        use_model=True,
        verbose=True,
        logging=False,
        logging_path = 'outcmaes/',
        archives=(),
        injecting_lam_duffosse=False
    ):


        self.fun = fun
        self.constraints = constraints
        self.logging = logging
        self.logging_path = logging_path
        self.sigma0 = sigma0
        self.verbose = verbose
        self.cma_opts = (
            cma.CMAOptions() if cma_opts is None else cma.CMAOptions(cma_opts)
        )
        if not verbose:
            self.cma_opts.set({"verbose": -9})

        self.x0 = x0
        self.dim = len(self.x0)
        self.cfun = cma.ConstrainedFitnessAL(
            self.fun,
            constraints=self.constraints,
            which='mean',
            logging=self.logging,
            archives=archives,
        )
        self.es = cma.CMAEvolutionStrategy(x0, sigma0, self.cma_opts)
        self.surrogate_opts = surrogate_opts
        self.n_constraints = len(self.constraints(x0))
        self.surrogate_opts = {} if surrogate_opts is None else surrogate_opts
        self.surrogates = []
        self.injecting_lam_duffosse = injecting_lam_duffosse

        if use_model:
            for i in range(self.n_constraints):
                self.surrogates.append(MahalanobisMeanSurrogate(self.dim,
                                                                self.constraints,
                                                                i,
                                                                self.es,
                                                                self,
                                                                logging,
                                                                **self.surrogate_opts))

        if self.logging:
            logged_attributes = [

            ]

            logged_callables = [
                lambda optimizer: int(optimizer.es.countiter),
                lambda optimizer: optimizer.constraints.n_calls_k,
                lambda optimizer: [surrogate.initialized for surrogate in optimizer.surrogates],
                lambda optimizer: [surrogate.intercept for surrogate in optimizer.surrogates],
                lambda optimizer: np.asarray([surrogate.nabla_ghat for surrogate in self.surrogates]).reshape(-1),
                lambda optimizer: [surrogate.get_percentage_correct_predictions(optimizer.cma_pop, use_old_model=True) for surrogate in optimizer.surrogates],
                lambda optimizer: [surrogate.get_percentage_correct_predictions(optimizer.cma_pop, use_old_model=False) for surrogate in optimizer.surrogates],
                lambda optimizer: [surrogate.hyperplane.diversity_measure for surrogate in optimizer.surrogates],
                lambda optimizer: [surrogate.ls_iterations for surrogate in optimizer.surrogates],
                lambda optimizer: [surrogate.iterations_good_bounds for surrogate in optimizer.surrogates],
                lambda optimizer: optimizer.cfun.al.lam if hasattr(optimizer.cfun, 'al') else [0]*self.n_constraints,
                lambda optimizer: optimizer.cfun.al.mu if hasattr(optimizer.cfun, 'al') else [0]*self.n_constraints,
                lambda optimizer: np.all([surrogate.iteration_info['mean_feasible'] for surrogate in optimizer.surrogates]),
                lambda optimizer: [surrogate.hyperplane.largest_singular_value for surrogate in optimizer.surrogates],
                lambda optimizer: [surrogate(optimizer.es.mean) for surrogate in optimizer.surrogates],
                lambda optimizer: optimizer.constraints(optimizer.es.mean, count=False),
                lambda optimizer: np.asarray([surrogate.hyperplane.singular_values[:-1] for surrogate in optimizer.surrogates]).reshape(-1),
                lambda optimizer: np.asarray([surrogate.hyperplane.stds_in_coordinates for surrogate in optimizer.surrogates]).reshape(-1),
                lambda optimizer: np.asarray([surrogate.hyperplane.euclidean_distances[-1] for surrogate in optimizer.surrogates]).reshape(-1),
            ]

            labels = [
                'iter',
                'gevals',
                'initialized',
                'intercept',
                'nabla_ghat',
                'ratio_correct_predictions_old_model',
                'ratio_correct_predictions_current_model',
                'diversity_measure',
                'ls_iterations',
                'iterations_good_bounds',
                'lam',
                'mu',
                'mean_feasible',
                'largest_singular_value_hyperplane',
                'ghat_mean',
                'g_mean',
                'singular_values',
                'stds_hyperplane',
                'final_ls_dist'
            ]
            self.surrogate_logger = cma.logger.Logger(self,
                                                    logged_attributes,
                                                    logged_callables,
                                                    name='surrogates',
                                                    path=self.logging_path,
                                                    labels=labels)

            es_attributes = [
                'mean',
                'sigma',
                'countevals'
            ]

            es_callables = [
                lambda es: np.max(es.sm.D)/np.min(es.sm.D),
                lambda es: es.sigma*np.min(es.sm.D),
                lambda es: es.sigma*np.max(es.sm.D)
            ]
            es_labels = [
                'mean',
                'sigma',
                'f_evals',
                'axis_ratio',
                'min_std',
                'max_std'
            ]

            self.es_logger = cma.logger.Logger(self.es,
                                es_attributes,
                                es_callables,
                                name='es',
                                path=self.logging_path,
                                labels=es_labels)

        # DEBUGGING
        self.last_mean = np.zeros(self.dim)
        self.mean_step = np.zeros(self.dim)


    @property
    def surrogates_is_initialized(self) -> list[bool]:
        return [surrogate.initialized for surrogate in self.surrogates]

    @property
    def idx_surrogates_initialized(self):
        return np.argwhere(self.surrogates_is_initialized).reshape(-1).tolist()

    @property
    def idx_surrogates_not_initialized(self):
        return np.argwhere(np.bitwise_not(self.surrogates_is_initialized)).reshape(-1).tolist()

    @property
    def split_surrogates_initialization(self) -> tuple[list]:
        return self.surrogates[self.surrogates_is_initialized], self.surrogates[self.surrogate_not_initialized]


    def push_custom_loggers(self):
        if self.surrogates:
            pass
        self.surrogate_logger.push()
        self.es_logger.push()

    @property
    def constraints_func_mixed(self):
        """Returns a function that returns the value of the surrogate or the true constraint (`if not surrogate.initialized`)"""
        def func(x):
            res = [np.nan] * self.n_constraints
            for i, surrogate in enumerate(self.surrogates):
                if surrogate.initialized:
                    res[i] = surrogate(x)
                else:
                    res[i] = 0 # this ensures that the penalty is 0 -> Al(x) = f(x) + 0
            return res
        return func

    @property
    def al_func_mixed(self):
        """Returns a function that returns the value of the surrogate or the true constraint (`if not surrogate.initialized`)"""
        def func(x):
            res = [np.nan] * self.n_constraints
            for i, surrogate in enumerate(self.surrogates):
                if surrogate.initialized:
                    res[i] = surrogate(x)
                else:
                    res[i] = 0 # this ensures that the penalty is 0 -> Al(x) = f(x) + 0
            return res
        return func

    def _update_surrogates(self, X):
        """Update all surrogates and initialize if necessary"""
        for i, surrogate in enumerate(self.surrogates):
            surrogate.save_iteration_info(self.es, X) # g-eval on mean

            if surrogate.initialized:
                surrogate.update()
                continue

            # If not initialized we do the following
            if surrogate.can_initialize():
                surrogate.initialize()


    def run(self):
        """Main optimization loop"""
        iteration = 0
        self._ensure_x0_feasible()
        while not self.es.stop():
            iteration += 1
            X = self.es.ask()
            self.cma_pop = X
            popsize = len(X)

            if self.surrogates:
                def constraints(x):
                    res = []
                    for surrogate in self.surrogates:
                        if surrogate.initialized:
                            res.append(surrogate(x))
                        else:
                            res.append(0)
                    return res
                self.cfun.constraints = constraints

            al_vals = [self.cfun(x) for x in X]
            self.es.tell(X, al_vals)

            if self.surrogates:
                self._update_surrogates(X)

            self.cfun.update(self.es) # Note: this does a g-evaluation in AL._fg_values()

            if self.injecting_lam_duffosse:
                if iteration < 2:
                    print("Injecting lam")
                self.cfun.al.lam = 2 * np.diag(self.fun.A)[:self.n_constraints]
                self.cfun.al.mu = np.zeros(self.n_constraints)

            if self.verbose:
                self.es.disp()

            if self.logging:
                self.push_custom_loggers()
                self.es.logger.add()

        if self.verbose:
            self._final_display()


        return self.es, self.cfun

    def _ensure_x0_feasible(self):
        """Checks that x0 is feasible"""
        g_x0 = self.constraints(self.x0)
        violated_constraints = np.asarray(g_x0) > 0
        violated_constraints_idxs = np.argwhere(violated_constraints).reshape(-1)
        if np.max(g_x0) > 0:
            warnings.warn(f"x0 is infeasible for constraint with idx: {[i for i in violated_constraints_idxs]}", stacklevel=1)
            self.x0 = find_feasible_x0(self.x0, self.constraints)
            self.es.x0 = self.x0


    def _set_al_components(self, X, constraint_idx: int | list[int] | None = None):
        """
        Here we set al components based on the following formula
        (mu is heavily inspired by cma.AugmentedLagrangian.set_coefficients()):
            lam = iqr(F) / n_constraints
            mu = 2 * iqr(F) / (5 * dim * (expected_range(G) + expected_range(G)))
        For expected_range(G) we use the following upper bound (jensen's inequality):
            expected_range(G) <= 2 * sigma * sqrt(2 * beta.T @ C @ beta * popsize)
        """
        raise NotImplementedError
        if constraint_idx is None:
            constraint_idx = list(range(self.n_constraints))
        try:
            F = [self.fun(x) for x in X]
            G = np.asarray([self.surrogates[constraint_idx](x) for x in X])
            iqr = cma.utilities.math.Mh.iqr
            self.cfun.al.lam[constraint_idx] = iqr(F) / (self.dim * iqr(G))
            self.cfun.al.mu[constraint_idx] = 2 * iqr(F) / (5 * self.dim * (iqr(G) + iqr(G**2)))

        except AttributeError: # No al component yet
            pass
        except TypeError: # al.lam is None
            pass
        except IndexError:
            if self.cfun.al._initialized:
                raise


    def _final_display(self):
        # CMA-output
        self.es.result_pretty()
        # Surrogate infos
        surrogate_state_arr = [surrogate.state for surrogate in self.surrogates] # list of dicts
        n_elements_nablaghat = 8
        for i, state_dict in enumerate(surrogate_state_arr):
            state_dict_copy = state_dict
            print(f'SURROGATE {i}')
            s = 'gradient = %s' % (str(state_dict.pop('nabla_ghat')[:n_elements_nablaghat])[:-1])
            s += ' ...]' if self.dim > n_elements_nablaghat else ' ]'
            print(s)
            for key, value in state_dict_copy.items():
                print(f'{key} = {value}')

            try:
                print('g-calls: ' + str(self.constraints.n_calls_k[i]))
            except AttributeError:
                pass

