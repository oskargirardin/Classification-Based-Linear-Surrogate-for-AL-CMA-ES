import numpy as np
from functools import partial
from .utils import get_projection_point_on_plane
import warnings

def eucledian_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))


def interval_halving(a0, b0, gfunc, max_iters, max_interval_length, a_feasibility=None, distance_func = None):
    if distance_func is None:
        distance_func = eucledian_distance

    if a_feasibility is None:
        a_feasibility = gfunc(a0)
    # assert gfunc(a0)*gfunc(b0) < 0, "a and b must be of different feasibility"
    niter = 1
    interval_length = distance_func(a0, b0)
    a = a0.copy()
    b = b0.copy()
    while niter < max_iters and interval_length > max_interval_length:
        midpoint = (a + b)/2
        feasibility_midpoint = gfunc(midpoint)
        if feasibility_midpoint * a_feasibility > 0:
            a = midpoint
        else:
            b = midpoint
        interval_length = distance_func(a, b)
        niter += 1

    res = {
        'a0': a0,
        'b0': b0,
        'a': a,
        'b': b,
        'feasible_a': a_feasibility,
        'feasible_b': - a_feasibility,
        'niter': niter
    }
    return res


def interval_halving2(a0, b0, gfunc, target_interval_length, max_iters, a0_feasibility=None, norm_func=None, termination_callback=None):
    if a0_feasibility is None:
        a0_feasibility = gfunc(a0)
    assert a0_feasibility*gfunc(b0) < 0, "a and b must be of different feasibility"

    if termination_callback is None:
        def termination_callback(a, b):
            return norm_func(a - b) <= target_interval_length

    niter = 0
    a = a0.copy()
    b = b0.copy()
    for _ in range(max_iters):
        if termination_callback(a, b):
            break
        midpoint = (a + b)/2
        feasibility_midpoint = gfunc(midpoint)
        if feasibility_midpoint * a0_feasibility > 0:
            a = midpoint
        else:
            b = midpoint
        niter += 1

    res = {
        'a': a,
        'b': b,
        'feasible_a': a0_feasibility,
        'feasible_b': -a0_feasibility,
        'niter': niter
    }
    # print(f'initial: {norm_func(a0 - b0):.4e}, target: {target_interval_length:.4e}, final: {interval_length:.4e}, iters: {niter}, midpoint: {midpoint}')
    return res


def get_bounds_opposite_feasibit2(a0, b0, gfunc, feasibility_a0 = None, feasibility_b0 = None, max_iters=100):
    if not feasibility_a0:
        feasibility_a0 = gfunc(a0)
    if not feasibility_b0:
        feasibility_b0 = gfunc(b0)
    assert feasibility_a0 * feasibility_b0 > 0, "a and b must be of same feasibility"
    niter = 1
    intial_feasibility_a_and_b = feasibility_a0
    # a, b = min(a, b), max(a, b)
    a, b = a0, b0
    ab_arr = [a, b]
    idx = 1
    #search_direction = (a0-b0)/np.linalg.norm(a0 - b0)
    while niter < max_iters:
        # x_old = ab_arr[idx]
        a_prev = a
        b_prev = b
        a = 2*a - b_prev
        b = 2*b - a_prev
        # print(f"a: {a}\nb: {b}")
        feasibility_a = gfunc(a)
        feasibility_b = gfunc(b)

        niter += 1

        if feasibility_a != intial_feasibility_a_and_b:
            # print(f"[a, b]: [{a, a_prev}]\nAfter {num_iter} iterations")
            return a, a_prev, niter
        elif feasibility_b != intial_feasibility_a_and_b:
            # print(f"[a, b]: [{b, b_prev}]\nAfter {num_iter} iterations")
            return b, b_prev, niter

    raise RuntimeError(f"num_iters exceeded max_iters ({max_iters})")


def get_bounds_opposite_feasibity(a0, b0, gfunc, feasibility_a0 = None, feasibility_b0 = None, max_iters=100):
    if not feasibility_a0:
        feasibility_a0 = gfunc(a0)
        print("New feasibility_a0:", feasibility_a0)
    if not feasibility_b0:
        feasibility_b0 = gfunc(b0)
    assert feasibility_a0 * feasibility_b0 > 0, "a and b must be of same feasibility"
    niter = 1
    feasibility_a_and_b = feasibility_a0
    # a, b = min(a, b), max(a, b)
    ab_arr = [a0.copy(), b0.copy()]
    idx = 1
    #search_direction = (a0-b0)/np.linalg.norm(a0 - b0)
    expanding_factor = 1
    while niter < max_iters:
        x_old = ab_arr[idx]
        ab_arr[idx] = ab_arr[idx] + expanding_factor * (ab_arr[idx] - ab_arr[idx - 1])

        if gfunc(ab_arr[idx]) * feasibility_a_and_b < 0:
            return ab_arr[idx], x_old, niter

        niter += 1
        idx = 1 - idx

    raise RuntimeError(f"num_iters exceeded max_iters ({max_iters})")


def get_boundary_point(x,
                       gfunc,
                       nabla_ghat,
                       intercept,
                       max_iters_halving,
                       max_interval_length,
                       max_iters_bound_search
                       ):
    x_p = get_projection_point_on_plane(x, nabla_ghat, intercept)
    feas_x = gfunc(x)
    feas_x_p = gfunc(x_p)
    if feas_x != feas_x_p:
        a, b, niter = interval_halving(
            x, x_p, gfunc, max_iters_halving, max_interval_length, feasibility_a=feas_x)
    else:
        try:
            a, b, niter1 = get_bounds_opposite_feasibity(
                x, x_p, gfunc, max_iters_bound_search)
            a, b, niter2 = interval_halving(
                a, b, gfunc, max_iters_halving, max_interval_length)
            niter = niter1 + niter2
        except RuntimeError:
            niter = 0
            a, b = x, x_p

    midpoint = (a+b)/2
    return midpoint, gfunc(midpoint), niter
