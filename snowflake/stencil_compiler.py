from __future__ import division
import abc
import ast
from collections import Iterable, OrderedDict
import copy
import ctypes
import itertools
from ctree.c.nodes import String, FunctionCall, SymbolRef, FunctionDecl, For, Constant, Assign, Lt, AugAssign, AddAssign, \
    CFile, Sub, ArrayRef, MultiNode, Mul, Add, LtE, TernaryOp, And
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
from ctree.frontend import dump
import math
from ctree.types import get_ctype
from rebox.specializers.order import Ordering
from _compiler import StencilCompiler, find_names, ArrayOpRecognizer, OpSimplifier
from nodes import WeightArray, SparseWeightArray, Stencil, StencilGroup

import numpy as np

from rebox.specializers.rm.encode import MultiplyEncode

__author__ = 'nzhang-dev'

class Compiler(object):

    @staticmethod
    def get_ndim(node):
        class Visitor(ast.NodeVisitor):
            def __init__(self):
                self.ndim = 0
            def visit_WeightArray(self, node):
                self.ndim = len(node.center)
            def visit_SparseWeightArray(self, node):
                self.visit_WeightArray(node)

        v = Visitor()
        v.visit(node)
        return v.ndim

    def __init__(self, index_name="index"):
        self.index_name = index_name

    def _compile(self, node, index_name, **kwargs):
        ndim = self.get_ndim(node)
        # wrapping the snowflake into a block
        if not isinstance(node, StencilGroup):
            node = StencilGroup([node])

        node = copy.deepcopy(node)
        stack = [
            StencilCompiler(index_name, ndim),
            OpSimplifier(),
            ArrayOpRecognizer(index_name, ndim),
        ]
        output = node
        for transformer in stack:
            output = transformer.visit(output)
        return output

    def _post_process(self, original, compiled, index_name, **kwargs):
        raise NotImplementedError("PostProcessing isn't implemented")

    def compile(self, node, **kwargs):
        if not isinstance(node, StencilGroup):
            node = StencilGroup([node])
        compiled = self._compile(node, self.index_name, **kwargs)
        processed = self._post_process(node, compiled, self.index_name, **kwargs)
        return processed


class PythonCompiler(Compiler):

    class BasicIndexOpConverter(ast.NodeTransformer):
        # def visit(self, node):
        #     print(node, type(node))
        #     return super(PythonCompiler.BasicIndexOpConverter, self).visit(node)

        def visit_IndexOp(self, node):
            return ast.Tuple(
                elts=[
                    self.visit(elt) for elt in node.elts
                ],
                ctx=ast.Load()
            )

        def visit_ArrayIndex(self, node):
            #print("visited", node)
            return self.visit_IndexOp(node)

    class IterationSpaceConverter(ast.NodeTransformer):
        def __init__(self, index_name, reference_array_name):
            self.index_name = index_name
            self.reference_array_name = reference_array_name

        def get_range_args(self, iter_range, dim):
            range_info = [
                ast.Num(n=iter_range[0]) if iter_range[0] >= 0 else
                ast.BinOp(
                    left=ast.Subscript(
                        value=ast.Attribute(
                            value=ast.Name(id=self.reference_array_name, ctx=ast.Load()),
                            attr='shape',
                            ctx=ast.Load()
                        ),
                        slice=ast.Index(value=ast.Num(n=dim)),
                        ctx=ast.Load()
                    ),
                    right=ast.Num(n=-iter_range[0]),
                    op=ast.Sub()
                )
            ] + [
                ast.Num(n=iter_range[1]) if iter_range[1] > 0 else
                ast.BinOp(
                    left=ast.Subscript(
                        value=ast.Attribute(
                            value=ast.Name(id=self.reference_array_name, ctx=ast.Load()),
                            attr='shape',
                            ctx=ast.Load()
                        ),
                        slice=ast.Index(value=ast.Num(n=dim)),
                        ctx=ast.Load()
                    ),
                    right=ast.Num(n=-iter_range[1]),
                    op=ast.Sub()
                )
            ]
            if iter_range.stride is not None:
                stride = iter_range.stride
            elif 0 <= iter_range.low < iter_range.high or iter_range.high < 0 <= iter_range.low:
                stride = 1
            else:
                stride = -1
            range_info.append(ast.Num(n=stride))
            return range_info




        def visit_IterationSpace(self, node):
            self.generic_visit(node)
            nested = node.body
            for dim, iteration_range in reversed(tuple(enumerate(node.space))):
                nested = ast.For(
                    target=ast.Name(id="{}_{}".format(self.index_name, dim), ctx=ast.Store()),
                    iter=ast.Call(
                        func=ast.Name(id="range", ctx=ast.Load()),
                        args=self.get_range_args(iteration_range, dim),
                        keywords=[],
                        starargs=None,
                        kwargs=None
                    ),
                    body=[nested] if not isinstance(nested, list) else nested,
                    orelse=[]
                )
            return nested

    def _post_process(self, original, compiled, index_name, **kwargs):
        target_name = original.body[-1].output
        #nodes = [self.IterationSpaceConverter(self.index_name).visit(node) for node in nodes]
        array_names = find_names(original)
        function_name = 'kernel'
        ndim = self.get_ndim(original)

        args = [
            target_name,
        ]
        args.extend(sorted(array_names - {target_name}))
        tree = ast.FunctionDef(
            name=function_name,
            args=ast.arguments(
                args=[ast.Name(id=arg, ctx=ast.Param()) for arg in args],
                vararg=None,
                kwarg=None,
                defaults=[]
            ),
            body=compiled.body,
            decorator_list=[]
        )
        # print("hello")
        tree = ast.Module(body=[tree])
        tree = self.BasicIndexOpConverter().visit(tree)
        tree = self.IterationSpaceConverter(self.index_name, original.body[-1].output).visit(tree)
        tree = ast.fix_missing_locations(tree)
        code = compile(tree, '<string>', 'exec')
        exec code in globals(), locals()
        return locals()['kernel']



class CCompiler(Compiler):

    def __init__(self, *args):
        super(CCompiler, self).__init__(*args)
        self._lsk = None

    class BlockConverter(ast.NodeTransformer):
        def visit_Block(self, node):
            return MultiNode(node.body)

    class IndexOpToEncode(ast.NodeTransformer):
        def visit_IndexOp(self, node):
            return ast.Call(
                func=ast.Name(id="encode", ctx=ast.Load()),
                args=[
                    self.visit(elt) for elt in node.elts
                ],
                vararg=None,
                kwarg=None,
                starargs=None
            )

        def visit_ArrayIndex(self, node):
            return self.visit_IndexOp(node)

    class IterationSpaceExpander(ast.NodeTransformer):
        def __init__(self, index_name, reference_array_shape, block_size=(32, 32)):
            self.index_name = index_name
            self.reference_array_shape = reference_array_shape
            self.block_size = block_size

        def visit_IterationSpace(self, node):
            node = self.generic_visit(node)
            inside = node.body
            make_low = lambda low, dim: low if low >= 0 else self.reference_array_shape[dim] + low
            make_high = lambda high, dim: high if high > 0 else self.reference_array_shape[dim] + high
            insides = []
            outsides = []
            for dim, (iteration_range, block_size) in enumerate(itertools.izip_longest(node.space,
                                                                                       self.block_size, fillvalue=0)):
                stride = iteration_range.stride or 1
                low = make_low(iteration_range.low, dim)
                high = make_high(iteration_range.high, dim)
                if block_size:
                    num_iterations = int(math.ceil((high - low) / block_size))
                    outer = SymbolRef('{}_{}_outer'.format(self.index_name, dim))
                    for_loop = For(
                        init=Assign(outer.copy(), Constant(low)),
                        test=Lt(outer, Constant(high)),
                        incr=AddAssign(outer, Constant(block_size))
                    )
                    outsides.append(for_loop)
                    inner = SymbolRef('{}_{}'.format(self.index_name, dim))
                    inner_for = For(
                        init=Assign(inner.copy(), outer),
                        test=And(
                            Lt(inner, Add(outer, Constant(block_size))),
                            Lt(inner, Constant(high))
                        ),
                        incr=AddAssign(inner, Constant(stride)),
                        body=[]
                    )
                    insides.append(inner_for)
                else:
                    insides.append(
                        For(
                            init=Assign(SymbolRef("{}_{}".format(self.index_name, dim)), Constant(low)),
                            test=Lt(SymbolRef("{}_{}".format(self.index_name, dim)), Constant(high)),
                            incr=AddAssign(SymbolRef("{}_{}".format(self.index_name, dim)), Constant(stride)),
                            body=[]
                        )
                )
            all_loops = insides + outsides
            for loop in all_loops:
                loop.body = [inside]
                inside = loop
            return inside


    class ConcreteSpecializedKernel(ConcreteSpecializedFunction):
        def finalize(self, entry_point_name, project_node, entry_point_typesig):
        #print("SmoothCFunction Finalize", entry_point_name)
            self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
            self.entry_point_name = entry_point_name
            return self

        def __call__(self, *args, **kwargs):
            return self._c_function(*args)


    class LazySpecializedKernel(LazySpecializedFunction):
        def __init__(self, py_ast=None, sub_dir=None, backend_name="default", names=None, target_name='out', index_name='index'):
            #print(dump(py_ast))
            super(CCompiler.LazySpecializedKernel, self).__init__(py_ast, sub_dir, backend_name)
            self.names = names
            self.target_name = target_name
            self.index_name = index_name
            self.sub_dir = 'snowflake_' + self.sub_dir

        @property
        def arg_spec(self):
            return [self.target_name] + list(sorted(self.names - {self.target_name}))

        class Subconfig(OrderedDict):
            def __hash__(self):
                return hash(tuple((name, arg.shape, arg.dtype) for name, arg in sorted(self.items())))

        def args_to_subconfig(self, args):
            names_to_use = self.arg_spec
            subconf = self.Subconfig()
            for name, arg in zip(names_to_use, args):
                subconf[name] = arg
            return subconf

        def transform(self, tree, program_config):
            # print("NEW FILE")
            subconfig, tuning_config = program_config
            CCompiler.IndexOpToEncode().visit(tree)
            ndim = subconfig[self.target_name].ndim
            c_tree = PyBasicConversions().visit(tree)
            ordering = Ordering([MultiplyEncode()])
            bits_per_dim = min([math.log(i, 2) for i in subconfig[self.target_name].shape]) + 1
            encode_func = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)
            c_func = FunctionDecl(
                name=SymbolRef("kernel"),
                params=[
                    SymbolRef(name=arg_name, sym_type=get_ctype(
                        arg if not isinstance(arg, np.ndarray) else arg.ravel()
                    )) for arg_name, arg in subconfig.items()
                ],
                defn=[c_tree]
            )
            c_func = CCompiler.IterationSpaceExpander(self.index_name, subconfig[self.target_name].shape).visit(c_func)
            c_func = CCompiler.BlockConverter().visit(c_func)
            # print(c_func)
            # print(encode_func)
            out_file = CFile(body=[encode_func, c_func])
            return out_file


        def finalize(self, transform_result, program_config):
            proj = Project(files=transform_result)
            fn = CCompiler.ConcreteSpecializedKernel()
            func_types = [
                        np.ctypeslib.ndpointer(arg.dtype, arg.ndim, arg.shape) if isinstance(arg, np.ndarray) else type(arg)
                        for arg in program_config.args_subconfig.values()
                    ]
            return fn.finalize(
                entry_point_name='kernel',
                project_node=proj,
                entry_point_typesig=ctypes.CFUNCTYPE(
                    None, *func_types
                )
            )

    def _post_process(self, original, compiled, index_name, **kwargs):
        py_ast = self.IndexOpToEncode().visit(compiled)
        ast.fix_missing_locations(py_ast)
        return self.LazySpecializedKernel(
            py_ast=py_ast,
            names=find_names(original),
            index_name=index_name,
            target_name=original.body[-1].output
        )
