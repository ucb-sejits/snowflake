import ast
import copy
from ctypes import c_long
import ctree
import itertools
from ctree.c.nodes import For, SymbolRef, Assign, Constant, Lt, AugAssign, AddAssign, Add, MultiNode, Pragma, And
from snowflake.optimizations import OptimizationLevels
from snowflake.optimizations.base import Optimization
from ctree.frontend import dump

__author__ = 'nzhang-dev'

class TilingOptimization(Optimization):
    """Applies a tiling by tile_sizes to iteration space nodes. Requires Filled Iteration Spaces"""

    optimization_level = OptimizationLevels.PRECOMPILED

    class _ForBlocker(ast.NodeTransformer):
        def __init__(self, blocking_sizes):
            self.blocking_sizes = blocking_sizes
            self.tagged = {}

        def _is_nested_for(self, node, n): # at least len(blocking_sizes) dimensions
            if n == 0:
                return True
            return isinstance(node, For) and node.body and self._is_nested_for(node.body[0], n-1)

        def visit_Pragma(self, node):
            """Attach pragmas to nodes"""
            if not node.body:
                return node
            if len(node.body) > 1:
                node.body = [MultiNode(node.body)]
            self.tagged[node.body[0]] = node
            return self.visit(node.body[0])  #basically skip this pragma node for now

        def visit_For(self, node):
            if not self._is_nested_for(node, len(self.blocking_sizes)):
                return self.generic_visit(node)
            cur_node = node
            last_for = node
            assignments = []  # for things like index_0 = index_0_outer + index_0_inner
            blocks = []
            for block_size in self.blocking_sizes:
                low = cur_node.init.right.value
                high = cur_node.test.right.value
                stride = cur_node.incr.value.value
                if block_size and block_size < (high - low): # if 0 means don't block.

                    if block_size % stride:
                        raise ValueError(
                            "Block size must be a multiple of the stride. Block size: {}, stride: {}".format(
                                block_size, stride
                            ))
                    # if (high-low) % block_size:
                    #     raise ValueError("Block size must divide high-low. Block size: {}, high: {}, low: {}".format(
                    #                      block_size, high, low))
                    index_name = cur_node.init.left.name

                    # create new loops
                    # modify current loop in place to save nodes
                    # create inner loop
                    # for (int i_inner = 0; i_inner < block_size; i_inner += stride)

                    cur_node.init.left = SymbolRef(name=index_name+"_inner", sym_type=cur_node.init.left.type)
                    cur_node.init.right = Constant(0)
                    # cur_node.test.left = SymbolRef(name=index_name+"_inner")
                    cur_node.test = And(
                        Lt(SymbolRef(name=index_name+"_inner"), Constant(block_size)),
                        Lt(
                            Add(
                                SymbolRef(name=index_name+"_inner"),
                                SymbolRef(name=index_name+"_outer")
                            ),
                            Constant(high)
                        )
                    )
                    # cur_node.test.right = Constant(block_size)
                    cur_node.incr.target = SymbolRef(name=index_name+"_inner")

                    # create outer loop
                    # for (int i_outer = low; i_outer < high; i_outer += block_size)
                    outside = For(
                        init=Assign(
                            SymbolRef(index_name+"_outer", sym_type=cur_node.init.left.type),
                            Constant(low)
                        ),
                        test=Lt(
                            SymbolRef(index_name+"_outer"),
                            Constant(high)
                        ),
                        incr=AddAssign(
                            SymbolRef(index_name+"_outer"),
                            Constant(block_size)
                        )
                    )
                    assignments.append(
                        Assign(
                            SymbolRef(index_name, sym_type=c_long(0)),
                            Add(SymbolRef(index_name+"_inner"), SymbolRef(index_name+"_outer"))
                        )
                    )
                    last_for = cur_node
                    cur_node = cur_node.body[0]
                    blocks.append(outside)
                else:
                    cur_node = cur_node.body[0]
            # at this point we've finished tiling, so we insert the assignments
            # cur_node.body = assignments + cur_node.body
            last_for.body = assignments + last_for.body
            if blocks:
                block_iter = iter(blocks)
                ret_node = cur = next(block_iter)
                for block_node in block_iter:
                    cur.body = [block_node]  # nesting blocks
                    cur = block_node
                cur.body = [node]
                return ret_node
            return node

    class RePragmaizer(ast.NodeTransformer):
        def __init__(self, tags):
            self.tags = tags

        def visit(self, node):
            """Visit a node."""
            node = self.generic_visit(node)
            if node in self.tags:
                tag = self.tags[node]
                tag.body = [node]
                return tag
            return node

    def optimize(self, tree, params):
        blocker = self._ForBlocker(params)
        tree = blocker.visit(tree)
        reprag = self.RePragmaizer(blocker.tagged)
        return reprag.visit(tree)


"""
Transform loops like

for (int i = low; i < high; i+=stride)

into

for (int i_outer = low; i_outer < high; i_outer += block_size)
    for (int i_inner = 0; i_inner < block_size; i_inner += stride)
        i = i_outer + i_inner
"""
