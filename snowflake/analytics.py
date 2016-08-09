from __future__ import print_function

import ast
from collections import defaultdict, namedtuple
import itertools
from math import ceil, floor
from multiprocessing import cpu_count, Pool
import pickle
import multiprocess

from snowflake.nodes import StencilComponent, RectangularDomain, DomainUnion

import numpy as np
import sympy
from sympy.solvers import diophantine as dio
from sympy.solvers import inequalities as ineq
from snowflake.vector import Vector

__author__ = 'nzhang-dev'

class AnalysisError(Exception):
    pass


def get_shadow(stencil):
    shadows = defaultdict(set)
    for node in (n for n in ast.walk(stencil) if isinstance(n, StencilComponent)):
        arr_name = node.name
        weights = node.weights
        shadows[arr_name] |= set(weights.vectors)
    return shadows

def analyze_dependencies(vec, space1, space2):
    x, y = sympy.symbols("x y", integer=True)  # used for setting up diophantine equations
    # TODO: Make it work on superficially colliding domains (but in reality spatially separated)

    results = []
    for dim, (s1, o1, v_comp, s2, o2) in enumerate(zip(space1.stride, space1.lower, vec, space2.stride, space2.lower)):
        if isinstance(v_comp, sympy.Expr):
            v_comp = v_comp.subs({'index_{}'.format(dim): s1*x})
        results.append(
            dio.diophantine(s1 * x + o1 + v_comp - s2 * y - o2)
        )
    return results


def validate_stencil(stencil, _verbose=False):
    shadow = get_shadow(stencil)
    output = stencil.output
    if output not in shadow:
        if _verbose:
            print("Output does not coincide with input grids. Analysis Complete.")
        return True
    iteration_space = stencil.iteration_space
    if isinstance(iteration_space, RectangularDomain):
        iteration_space = DomainUnion([iteration_space])
    if _verbose:
        print("Analyzing stencil over {}".format(iteration_space))

    for vector in shadow[output]:
        for space1, space2 in itertools.combinations_with_replacement(iteration_space.domains, 2):
            if (not any(vector)) and space1 == space2:  # zero vector in the same space
                continue
            if _verbose:
                print("Analyzing {} over spaces {}\t{}".format(vector, space1, space2))
            analysis = analyze_dependencies(vector, space1, space2)
            if all(analysis):
                if _verbose:
                    print("FAILED")
                return False
    return True

StencilData = namedtuple("StencilData", ["iteration_space", "shadow", "output", "primary_mesh"])
ThinDomainUnion = namedtuple("ThinDomainUnion", ["domains"])
ThinRD = namedtuple("ThinRD", ["lower", "upper", "stride"])

def get_data(stencil):
    return StencilData(
        # ThinDomainUnion(
        #     [ThinRD(space.lower, space.upper, space.stride) for space in stencil.iteration_space.domains]
        # ),
        stencil.iteration_space,
        get_shadow(stencil),
        stencil.output,
        stencil.primary_mesh
    )

def _contains_integer(interval):
    if interval is sympy.EmptySet():
        return False
    if interval.is_left_unbounded or interval.is_right_unbounded:
        return True  # trivially true
    lower = interval.left if not interval.left_open else int(floor(interval.left + 1))
    upper = interval.right if not interval.right_open else int(ceil(interval.right - 1))
    return (upper - lower) >= 0

def _has_conflict(rd1, rd2, offset):
    """
    :param rd1: rectangular domain from Stencil1 (1D) <- (low, high, stride) (written domain)
    :param rd2: rectangular domain from Stencil2 (1D) <- (low, high, stride) (read domain)
    :param offset_vector: Offset from the center (relative to rd2), 1D projections
    :return: Whether OV from rectangular domain 2 reads from rectangular domain 1
    """

    n1 = sympy.Symbol('n1')
    n2 = sympy.Symbol('n2')

    # Diophantine equations:
    # offset(iterationspace1) + n1 * stride(iteration_space1) <- write vectors
    diophantine_eq1 = rd1[0] + n1 * rd1[2]
    # offset(iterationspace2) + n2 * stride(iteration_space2) + offset_vector  <- Read vectors
    diophantine_eq2 = rd2[0] + n2 * rd2[2] + offset
    # Since sympy solves eq = 0, we want dio_eq1 - dio_eq2 = 0.
    eqn = diophantine_eq1 - diophantine_eq2
    parameter = sympy.Symbol("t", integer=True)
    sat_param = sympy.Symbol("t_0", integer=True)
    if not eqn.free_symbols:
        return eqn == 0
    solutions = dio.diophantine(eqn, parameter) # default parameter is "t"
    #print("Sols:", solutions)
    for sol in solutions:
        if len(eqn.free_symbols) != 2:
            if n1 in eqn.free_symbols:
                parametric_n1 = sol[0]
                parametric_n2 = 0
            else:
                parametric_n1 = 0
                parametric_n2 = sol[0]
        else:
            (parametric_n1, parametric_n2) = sol
        # Solutions is a set of tuples, each containing either a number or parametric expression
        # which give conditions on satisfiability.

        # are these satisfiable on the original bounds?
        # substitute the parametric forms in
        substituted_1 = diophantine_eq1.subs({n1: parametric_n1})
        substituted_2 = rd2[0] + parametric_n2 * rd2[2]  # we ditch the offset because it's irrelevant since the bounds are based on the center of the stencil


        #print(substituted_1, "\t", substituted_2)
        #print(rd1[0], rd1[1], rd2[0], rd2[1])
        #now do they satisfy the bounds?
        satisfactory_interval = ineq.reduce_rational_inequalities(
            [[
                rd1[0] <= substituted_1,
                rd1[1] > substituted_1,
                rd2[0] <= substituted_2,
                rd2[1] > substituted_2
            ]],
            sat_param,
            relational=False
        )
        #print(satisfactory_interval)

        if _contains_integer(satisfactory_interval):
            return True
    return False



def stencil_conflict(stencil1, stencil2, shape_map):
    """
    :param stencil1: First stencil to be run
    :param stencil2: Second stencil to be run
    :param shape_map: Map of meshname : shape
    :return: whether stencil2 has a read where stencil1 writes to.
    """

    return _stencil_conflict(get_data(stencil1), get_data(stencil2), shape_map)

def _stencil_conflict(data1, data2, shape_map):

    # shadows = get_shadow(stencil2)  # these are all of the offset vectors for each array
    shadows = data2.shadow
    if data1.output not in shadows:
        return False  # not read/writing from same mesh

    shade = shadows[data1.output] # these are vectors that are used.

    shape = shape_map[data1.primary_mesh]

    if data1.output == data2.output: # if they write to the same array
        shade |= {Vector.zero_vector(len(shape))}



    # Perform exhaustive search over pairs of iteration spaces?

    # TODO: Probably should make this smarter
    for write_domain in data1.iteration_space.reify(shape).domains:
        for read_domain in data2.iteration_space.reify(shape).domains:
            # in order to save time, compute the projections for each vector onto each dimension.
            collisions = [{} for _ in shape]
            for vector in shade:
                for dim, val in enumerate(vector):
                    collisions[dim][val] = False

            # using 1d slices of each domain
            for wd_1d, rd_1d, offsets in zip(
                zip(write_domain.lower, write_domain.upper, write_domain.stride),
                zip(read_domain.lower, read_domain.upper, read_domain.stride),
                collisions
            ):
                for offset in offsets:
                    offsets[offset] = _has_conflict(wd_1d, rd_1d, offset)
            #print(collisions)
            for vector in shade:
                if all(collisions[dim][val] for dim, val in enumerate(vector)): # some vector has a collision on all dimensions
                    return True
    return False



def create_dependency_graph(stencil_group, shape_map):
    """
    returns a graph such that G[a][b] is True iff a depends on b.
    """
    graph = defaultdict(dict)
    minimal_stencil_list = []
    for a, b in itertools.product(stencil_group.body, repeat=2):
        if hash(a) not in graph:
            minimal_stencil_list.append(a)  # haven't seen this before
        graph[hash(a)][hash(b)] = None

    for a, b in itertools.product(minimal_stencil_list, repeat=2):
        graph[hash(a)][hash(b)] = stencil_conflict(b, a, shape_map)
    #print(graph)
    return graph


def conflict_wrapper((hash_a, hash_b, data_a, data_b, m)):
    #       b          a        s1, s2, map
    result = hash_b, hash_a, _stencil_conflict(data_b, data_a, m)
    return result

def create_parallel_graph(stencil_group, shape_map, num_threads=cpu_count()):
    """
    returns a graph such that G[a][b] is True iff a depends on b.
    """

    pool = Pool(1)

    graph = defaultdict(dict)
    minimal_stencil_list = []
    for a, b in itertools.product(stencil_group.body, repeat=2):
        if hash(a) not in graph:
            minimal_stencil_list.append(a)  # haven't seen this before
        graph[hash(a)][hash(b)] = None
    args = [(hash(b), hash(a), get_data(a), get_data(b), shape_map)
                                      for a, b in itertools.product(minimal_stencil_list, repeat=2)]
    results = pool.map(conflict_wrapper, args)
    for b, a, result in results:
        graph[a][b] = result
    # for a, b in itertools.product(minimal_stencil_list, repeat=2):
    #     graph[hash(a)][hash(b)] = stencil_conflict(b, a, shape_map)

    return graph


def create_independent_groups(stencil_group, shape_map):
    """
    :param stencil_group: a given stencil group
    :return: a list of groups that can be run in parallel (i.e. things within a group don't interfere)
    """
    groups = []
    current_group = []
    graph = create_dependency_graph(stencil_group, shape_map)
    for index, stencil in enumerate(stencil_group.body):
        for other_index, other_stencil in current_group:
            if graph[index][other_index]:
                # if we have a conflict with a stencil already within the group
                groups.append(current_group)
                current_group = []
                break
        current_group.append((index, stencil))
    if current_group:
        groups.append(current_group)
    return [
        [stencil for index, stencil in group]
        for group in groups
    ]



def _getdata(a):
    return a.__array_interface__['data'][0], a.nbytes

def array_collision(a, b):
    a_low, a_len = _getdata(a)
    b_low, b_len = _getdata(b)
    a_high = a_low + a_len
    b_high = b_low + b_len
    return a_low <= b_low <= a_high or a_low <= b_high <= a_high