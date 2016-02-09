from snowflake.nodes import Stencil, WeightArray, StencilComponent, DomainUnion, RectangularDomain
from snowflake.stencil_compiler import CCompiler

import numpy as np

__author__ = 'nzhang-dev'


def run_test():
    size = 10
    weight_array = WeightArray(
        np.ones((3, 3), dtype=np.float) / 9
    )
    component = StencilComponent(
        "input",
        weight_array
    )
    red = DomainUnion([
        RectangularDomain((
            (1, -1, 2),
            (1, -1, 2)
        )),
        RectangularDomain((
            (2, 0, 2),
            (2, 0, 2)
        ))
    ])
    stencil = Stencil(
        component,
        "output",
        red
    )
    ccompiler = CCompiler()
    kern = ccompiler.compile(stencil)
    arr = np.arange(size**2, dtype=np.float).reshape((size, size))
    out = np.zeros_like(arr)
    kern(out, arr)
    print(out)

if __name__ == "__main__":
    run_test()
