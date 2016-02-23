from __future__ import print_function
import itertools
from snowflake.analytics import stencil_conflict, validate_stencil

from snowflake.nodes import Stencil, StencilComponent, SparseWeightArray, DomainUnion, RectangularDomain
from snowflake.vector import Vector

__author__ = 'nzhang-dev'


def test_collision():
    """
    Test using Red Black. All of them should collide.
    """
    dimensions = 3
    # Red domains are formed by the origin plus all non-negative vectors with 1-norm 2.
    red_domain_starts = [Vector.zero_vector(dimensions)] + [i for i in Vector.von_neumann_vectors(dimensions, 2)
                                                            if all(coord >= 0 for coord in i)]
    red_domain = DomainUnion([
        RectangularDomain([(s, -1, 2) for s in start])
        for start in red_domain_starts
    ])


    # Black domains are formed by non-negative vectors with 1-norm 1.

    black_domain_starts = [i for i in Vector.von_neumann_vectors(dimensions, 1) if all(coord >= 0 for coord in i)]

    black_domain = DomainUnion([
        RectangularDomain([(s, -1, 2) for s in start])
        for start in black_domain_starts
    ])

    star_neighborhood = {Vector.unit_vector(d, dimensions): 1 for d in range(dimensions)}
    star_neighborhood.update(
        {-Vector.unit_vector(d, dimensions): 1 for d in range(dimensions)}
    )
    red = Stencil(
        StencilComponent(
            "mesh",
            SparseWeightArray(
                star_neighborhood
            )
        ),
        "mesh",
        red_domain
    )

    black = Stencil(
        StencilComponent(
            "mesh",
            SparseWeightArray(
                star_neighborhood
            )
        ),
        "mesh",
        black_domain
    )

    red_oop = Stencil(
        StencilComponent(
            "mesh",
            SparseWeightArray(
                star_neighborhood
            )
        ),
        "output",
        red_domain
    )

    black_oop = Stencil(
        StencilComponent(
            "mesh",
            SparseWeightArray(
                star_neighborhood
            )
        ),
        "output",
        black_domain
    )

    print("Testing with simulated 32^dimensions mesh")
    print("WRITE-READ", "has collision")
    print("RED - RED", stencil_conflict(red, red, {"mesh": (32,)*dimensions}))
    print("BLACK - BLACK", stencil_conflict(black, black, {"mesh": (32,)*dimensions}))
    print("RED - BLACK", stencil_conflict(red, black, {"mesh": (32,)*dimensions}))
    print("BLACK - RED", stencil_conflict(black, red, {"mesh": (32,)*dimensions}))
    print("REDOOP - BLACKOOP", stencil_conflict(red_oop, black_oop, {"mesh": (32,)*dimensions}))
    print("BLACKOOP - REDOOP", stencil_conflict(black_oop, red_oop, {"mesh": (32,)*dimensions}))
    print("INTRA-red is valid", validate_stencil(red))
    print("INTRA-black is valid", validate_stencil(black))

def test_boundaries():
    """
    Tests if boundaries can be parallelized (or not).
    """
    dimensions = 3

    def reference_vector(vec):
        first_nonzero_position = next(i for i in range(len(vec)) if vec[i])
        return -vec * Vector.unit_vector(first_nonzero_position, len(vec))

    def get_stencil(boundary):  # boundaries of 1-norm n depend on boundaries of 1-norm n-1 by looking in on the first non-zero element.
        read_vector = reference_vector(boundary)
        component = StencilComponent(
            'mesh',
            SparseWeightArray(
                {
                    read_vector: 1
                }
            )
        )
        return Stencil(component, 'mesh', [
            (-1, 0, 1) if bound == 1 else (0, 1, 1) if bound == -1 else (1, -1, 1)
            for bound in boundary
        ])

    stencils = [
        (boundary, get_stencil(boundary)) for boundary in Vector.moore_vectors(dimensions)
    ]

    for (b1, s1), (b2, s2) in itertools.product(stencils, repeat=2):
        has_conflict = b2+reference_vector(b2) == b1 or b1 == b2
        reported = stencil_conflict(s1, s2, shape_map={"mesh": (32,)*dimensions})
        print(b1, b2, reported, has_conflict, "*" * 5 * (has_conflict != reported))

if __name__ == "__main__":
    # test_collision()
    test_boundaries()