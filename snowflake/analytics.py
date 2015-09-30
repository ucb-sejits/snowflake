from __future__ import print_function

import ast
from collections import defaultdict
import itertools

from snowflake.nodes import StencilComponent, RectangularDomain, DomainUnion

import numpy as np
import sympy
from sympy.solvers import diophantine as dio
from snowflake.vector import Vector

__author__ = 'nzhang-dev'


def get_shadow(stencil):
    shadows = defaultdict(set)
    for node in (n for n in ast.walk(stencil) if isinstance(n, StencilComponent)):
        arr_name = node.name
        weights = node.weights
        shadows[arr_name] |= set(weights.vectors)
    return shadows

def analyze_dependencies(vec, space1, space2):
    x, y = sympy.symbols("x y", integer=True)  # used for setting up diophantine equations
    # s1x + o1 + v = s2y + o2
    return [
        dio.diophantine(s1 * x + o1 + v_comp - s2 * y - o2) for s1, o1, v_comp, s2, o2 in
        zip(space1.stride, space1.lower, vec, space2.stride, space2.lower)
    ]

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
                return False
    return True


def _getdata(a):
    return a.__array_interface__['data'][0], a.nbytes

def array_collision(a, b):
    a_low, a_len = _getdata(a)
    b_low, b_len = _getdata(b)
    a_high = a_low + a_len
    b_high = b_low + b_len
    return a_low <= b_low <= a_high or a_low <= b_high <= a_high