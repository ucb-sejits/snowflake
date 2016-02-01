import ast
from snowflake.optimizations import OptimizationLevels
from snowflake.optimizations.base import CompilerOptimization

__author__ = 'nzhang-dev'

class TilingOptimization(CompilerOptimization):
    """Applies a tiling by tile_sizes to iteration space nodes. Requires Filled Iteration Spaces"""

    optimization_level = OptimizationLevels.REIFIED


