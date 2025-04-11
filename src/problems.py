import numpy as np
from cma.fitness_transformations import Rotation
from .utils import construct_basis, binarize, sign


class BaseProblem:
    def get(self):
        return self.objective, self.constraint

    def __str__(self) -> str:
        return f"Objective: {self.objective}, Constraint: {self.constraint}"

class ObjectiveFunction:
    def __init__(self) -> None:
        pass

class SphereFun(ObjectiveFunction):
    def __init__(self, center = 1, f_offset = 0):
        self.center = center
        self.f_offset = f_offset
        self.f = lambda x: sum((x-self.center)**2) + self.f_offset

    def __call__(self, x):
        return self.f(x)

    def __str__(self) -> str:
        return f"Sphere function (center: {self.center})"


class Constraint:
    def __init__(self, type_constraint) -> None:
        self.type_constraint = type_constraint
        self.constraint = self._set_constraint()


    def _set_constraint(self):
        if self.type_constraint == "quadratic":
            a = -0.3
            assert a >= -0.5, "a must be larger than -0.5 for the optimum to be in 0"
            return lambda x: [a*x[0]**2+x[0]+x[1]]
        elif self.type_constraint == "linear":
            return lambda x: [sum(x)]

    def __call__(self, x):
        return self.constraint(x)

    def __str__(self) -> str:
        return self.type_constraint


class SphereSingleConstraint(BaseProblem):
    def __init__(self, type_constraint, center = 1, f_offset = 0):
        self.objective = SphereFun(center, f_offset)
        self.constraint = Constraint(type_constraint)




class Dufosse2021_MM(BaseProblem):
    def __init__(self, dim = 2, case = None, m = 1):
        self.center = np.array([1]*m + [0]*(dim-m))
        self.objective = lambda x: sum((x-self.center)**2)
        if case == 1:
            self.constraint = lambda x: [(1 + 1*(x[0]<= 0))*x[0]]
        if case == 2:
            self.constraint = lambda x: [(1*(x[0]>0))*x[0]]
        else:
            self.constraint = lambda x: [x[k] for k in range(m)]

class Dufosse2021_Base(BaseProblem):
    def __init__(self, dim = 2, m = 1):
        self.dim = dim
        self.m = m
        self.center = np.array([1]*self.m + [0]*(self.dim-self.m))
        self.constraint = lambda x: [x[k] for k in range(self.m)]
        self.objective = lambda x: sum((x-self.center)**2)
        self.f_c_star = self.m


class Dufosse2021Constr:
    def __init__(self, m: int, dim: int, binary: bool, easy_constraint: bool, linear_step: bool, rotation_func, relu: bool) -> None:
        self.m = m
        self.dim = dim
        self.binary = binary
        self.linear_step = linear_step
        self.easy_constraint = easy_constraint
        self.rotation_func = rotation_func
        self.relu = relu

    def __call__(self, x, count=True, id=None):
        if self.rotation_func:
            x = self.rotation_func(x)

        idxs = np.arange(self.m)
        if self.easy_constraint:
            idxs = idxs[::-1] + (self.dim - self.m)

        def transform(x):
            if self.binary:
                return sign(x)
            elif self.linear_step:
                step = 1
                return x + step if x > 0 else x
            elif self.relu:
                return x if x > 0 else 0
            else:
                return x

        return [transform(x[i]) for i in idxs]

    def __iter__(self):
        self.iteration = 0
        return self

    def __next__(self):
        if self.iteration < self.m:
            self.iteration += 1
            return lambda x: int(x[self.iteration - 1] > 0)
        raise StopIteration

    def __str__(self):
        return f"Linear constraints: {self.m}, Type: Dufosse2021"

class Dufosse2021ScipyConstraint:
    def __init__(self, m) -> None:
        self.m = m
        self.n_calls = 0
    def __call__(self, x):
        self.n_calls += 1
        return [int(x[k] > 0) for k in range(self.m)]


class Dufosse2021Obj:
    def __init__(self, dim, m, condition_number, easy_constraints, rotation_func):
        self.dim = dim
        self.condition_number = condition_number
        self.m = m
        self.easy_constraints = easy_constraints
        self.rotation_func = rotation_func
        if self.easy_constraints:
            self.center = np.array([0]*(self.dim-self.m)+ [1]*self.m)
        else:
            self.center = np.array([1]*self.m + [0]*(self.dim-self.m))
        self.A = np.diag([self.condition_number**((self.dim-i)/(self.dim-1)) for i in range(1,self.dim+1)])
        self.f_c_star = self([0]*self.dim)

    def __call__(self, x):
        if self.rotation_func:
            x = self.rotation_func(x)
        return np.transpose(x - self.center) @ self.A @ (x - self.center)

    def __str__(self) -> str:
        return f"Ellipsoid (cond. num = {self.condition_number}) with {self.m} linear constraints. Constrainted minimum: x-star: {[0]*self.dim}, f-star: {self.f_c_star}"


class Dufosse2021(BaseProblem):
    def __init__(self, dim=2, m=1, condition_number=1e6, rotation=None, binary=False, linear_step=False, easy_constraints=False, relu=False, seed=None) -> None:
        self.dim = dim
        self.m = m
        self.condition_number = condition_number
        self.rotation = rotation
        self.binary = binary
        self.linear_step = linear_step
        self.easy_constraints = easy_constraints
        self.relu = relu
        self.seed = seed
        assert m <= dim
        
        self.reset(self.seed)

        self.xstar = np.zeros(dim)
        self.f_c_star = self.objective(self.xstar)

    def __str__(self):
        return f"Dufosse2021"

    def get_gradient_obj(self, x):
        x = np.asarray(x)
        A = self.objective.A
        e = self.objective.center
        assert x.shape == e.shape, f"x should be of shape {e.shape}"
        return 2 * np.diag(A) * (x-e)
    
    def reset(self, seed=None):
        self.seed = seed
        self.rotation_func = Rotation(self.seed) if self.rotation else None
        self.objective = Dufosse2021Obj(self.dim, self.m, self.condition_number, self.easy_constraints, self.rotation_func)
        self._constraint = Dufosse2021Constr(self.m, self.dim, self.binary, self.easy_constraints, self.linear_step, self.rotation_func, self.relu)
        self.constraint = ConstraintWrapper(self._constraint, self.m)


class Constraints:
    def __init__(self, gfunc, make_binary = True) -> None:
        self.gfunc = gfunc
        self.n_calls = 0
        self.make_binary = make_binary

    def __call__(self, x):
        self.n_calls += 1
        if self.make_binary:
            return [binarize(res, class_two_val=-1) for res in self.gfunc(x)]
        else:
            return self.gfunc(x)

class RotatedEllipsoidSingleLinearConstraint(BaseProblem):
    def __init__(self, dim = 4, condition_number = 1e6, binary_constraint = True) -> None:
        self.dim = dim
        self.condition_number = condition_number
        self.center = np.ones(dim)
        self.D = np.diag([self.condition_number**((self.dim-1-i)/(self.dim-1)) for i in range(self.dim)])
        self.Q = construct_basis(self.center)
        self.A = self.Q @ self.D @ self.Q.T
        self.objective = lambda x: np.transpose(x-self.center) @ self.A @ (x-self.center)
        self.constraint = ConstraintWrapper(Constraints(lambda x: [np.sum(x)], make_binary=binary_constraint), m=1)
        self.f_c_star = self.objective(np.zeros(self.dim))


class SphereQuadraticConstraint(BaseProblem):
    def __init__(self, dim, condition_number = 1e0, a_constraint = 1, binary_constraint = True):
        self.dim = dim
        self.condition_number = condition_number
        self.a_constraint = a_constraint
        self.center = np.array([1] + [0]*(self.dim-1))
        self.A = np.diag([self.condition_number**((self.dim-i)/(self.dim-1)) for i in range(1,self.dim+1)])
        self.objective = lambda x: np.transpose(x - self.center) @ self.A @ (x - self.center)
        self.constraint = ConstraintWrapper(Constraints(lambda x: [x[0] + self.a_constraint*x[1]**2], make_binary=binary_constraint), m=1)


class ConstraintWrapper:
    def __init__(self, constraint, m, archive=False) -> None:
        self._constraint = constraint
        self.m = m
        self.n_calls = 0
        self.n_calls_k = [0] * m
        self.archive = archive
        if self.archive:
            self.archive_solutions = []

    def reset(self):
        self.n_calls = 0
        self.n_calls_k = [0] * self.m

    def __call__(self, x, id=None, count=True):
        if count:
            self.n_calls += 1
            if id is not None:
                self.n_calls_k[id] += 1
        if self.archive:
            self.archive_solutions.append(x)
        return self._constraint(x) if id is None else self._constraint(x)[id]


class OffAxisElli:
    def __init__(self, dim, cond = 1e6) -> None:
        self.dim = dim
        self.cond = cond
        self.A = np.diag([10**((i+1)*np.log10(cond)/dim) for i in range(dim)])
        self.beta = -2*np.diag(self.A)
        self.alpha = 2*np.trace(self.A)

    def get_beta(self, normalized = True):
        if normalized:
            return self.beta / np.linalg.norm(self.beta)

        return self.beta

    def get_alpha(self, normalized = True):
        if normalized:
            return self.alpha / np.linalg.norm(self.beta)

        return self.alpha

    def f(self, x):
        x_arr = np.asarray(x)
        res = x_arr.T @ self.A @ x_arr
        return res.squeeze(-1)

    def constraint(self, x):
        return [self.beta.dot(x) + self.alpha]

    def get(self):
        return self.f, ConstraintWrapper(self.constraint, m=1)