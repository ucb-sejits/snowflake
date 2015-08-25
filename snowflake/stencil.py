__author__ = 'nzhang-dev'

###############################################################################################################
# A component is defined by a weight array, which may or may not recursively contain components.
###############################################################################################################

from nodes import Stencil, StencilConstant, StencilComponent, WeightArray, SparseWeightArray, Vector

__all__ = ['Stencil', 'StencilConstant', 'StencilComponent', 'WeightArray', 'SparseWeightArray', 'Vector']