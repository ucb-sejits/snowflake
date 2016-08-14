from __future__ import division
import ast
import atexit
from collections import OrderedDict
import copy
import ctypes
import functools
import os
import re
import ctree
from ctree.c.nodes import SymbolRef, FunctionDecl, For, Constant, Assign, Lt, AddAssign, \
    CFile, MultiNode
from ctree.cpp.nodes import CppInclude
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
from ctree.transforms import DeclarationFiller
from ctree.types import get_ctype
import time
from _compiler import StencilCompiler, find_names, ArrayOpRecognizer, OpSimplifier
from nodes import StencilGroup

import numpy as np

from snowflake.compiler_utils import generate_encode_macro, fill_iteration_spaces
from snowflake.optimizations import OptimizationLevels
from snowflake.optimizations.tile import TilingOptimization

__author__ = 'nzhang-dev'

class OptimizationList(object):
    """Defines a list of optimizations to be run."""

    def __init__(self, optimization_list=()):
        if not optimization_list:
            pass
        if not all(a.optimization_level <= b.optimization_level for a,b in zip(optimization_list[:-1], optimization_list[1:])):
            raise TypeError("Optimizations must be in order")
        self.optimization_groups = {
            optimizationlevel: [opt for opt in optimization_list if opt.optimization_level == optimizationlevel]
            for optimizationlevel in OptimizationLevels.choices
        }


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

    def __init__(self, optimizations=(), record_time=False):
        self.index_name = "index"
        self.optimizations = OptimizationList(optimizations)
        self.record_time = record_time

    def _compile(self, node, index_name, **kwargs):
        ndim = self.get_ndim(node)
        # wrapping the snowflake into a block
        if not isinstance(node, StencilGroup):
            node = StencilGroup([node])

        node = copy.deepcopy(node)
        stack = [
            StencilCompiler(ndim),
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
        original = copy.deepcopy(node)
        h = hash(original)
        copied = copy.deepcopy(node)
        compiled = self._compile(original, self.index_name, **kwargs)
        processed = self._post_process(copied, compiled, self.index_name, **kwargs)
        call_method = processed.__call__
        if self.record_time:
            times = []
            func_name = kwargs.get("name", h)
            @functools.wraps(call_method)
            def timed_call(*args):
                t = time.time()
                res = call_method(*args)
                end = time.time()
                times.append(end-t)
                return res

            @atexit.register
            def print_result():
                if times:
                    print(func_name, len(times), sum(times)/len(times))

            processed.__call__ = timed_call
        return processed

    class SpecializationKernel(LazySpecializedFunction):
        def get_program_config(self, args, kwargs):
            # Don't break old specializers that don't support kwargs
            args_subconfig = self.args_to_subconfig(args)
            try:
                self._tuner.configs.send((args, args_subconfig))
            except TypeError:
                "Can't send into an unstarted generator"
                pass
            tuner_subconfig = next(self._tuner.configs)
            return self.ProgramConfig(args_subconfig, tuner_subconfig)

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
            if iter_range[2] is not None:
                stride = iter_range[2]
            elif 0 <= iter_range[0] < iter_range[1] or iter_range[1] < 0 <= iter_range[0]:
                stride = 1
            else:
                stride = -1
            range_info.append(ast.Num(n=stride))
            return range_info




        def visit_IterationSpace(self, node):
            self.generic_visit(node)
            parts = []
            for space in node.space.spaces:
                nested = node.body
                for dim, iteration_range in reversed(tuple(enumerate(zip(space.low, space.high, space.stride)))):
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
                parts.append(nested)
            return parts


    def _post_process(self, original, compiled, index_name, **kwargs):
        target_names = [part.primary_mesh for part in original.body]
        #nodes = [self.IterationSpaceConverter(self.index_name).visit(node) for node in nodes]
        array_names = find_names(original)
        function_name = 'kernel'
        ndim = self.get_ndim(original)
        body = []
        for output_name, space in zip(target_names, compiled.body):
            space = self.BasicIndexOpConverter().visit(space)
            parts = self.IterationSpaceConverter(self.index_name, output_name).visit(space)
            body.extend(parts)
        args = [part.output for part in original.body]
        args.extend(sorted(array_names - set(target_names)))
        tree = ast.FunctionDef(
            name=function_name,
            args=ast.arguments(
                args=[ast.Name(id=arg, ctx=ast.Param()) for arg in args],
                vararg=None,
                kwarg=None,
                defaults=[]
            ),
            body=body,
            decorator_list=[]
        )
        # print("hello")
        tree = ast.Module(body=[tree])
        # tree = self.BasicIndexOpConverter().visit(tree)
        # tree = self.IterationSpaceConverter(self.index_name, original.body[-1].output).visit(tree)
        tree = ast.fix_missing_locations(tree)
        code = compile(tree, '<string>', 'exec')
        exec code in globals(), locals()
        return locals()['kernel']



class CCompiler(Compiler):

    @staticmethod
    def _shape_to_str(shape):
        return '_'.join(map(str, shape))

    def __init__(self):
        super(CCompiler, self).__init__()
        self._lsk = None
        self.tile_size = (16, 16)

    class BlockConverter(ast.NodeTransformer):
        def visit_Block(self, node):
            return MultiNode(node.body)

    class IndexOpToEncode(ast.NodeTransformer):
        def __init__(self, name_shape_map):
            self.name_shape_map = name_shape_map
            self.shapes = []

        def visit_IndexOp(self, node):
            return ast.Call(
                func=ast.Name(id="encode"+CCompiler._shape_to_str(self.shapes[-1]), ctx=ast.Load()),
                args=[
                    self.visit(elt) for elt in node.elts
                ],
                vararg=None,
                kwarg=None,
                starargs=None
            )

        def visit_Subscript(self, node):
            node_name = node.value.id
            self.shapes.append(self.name_shape_map[node_name])
            node.slice = self.visit(node.slice)
            self.shapes.pop(-1)
            return node


        def visit_ArrayIndex(self, node):
            return self.visit_IndexOp(node)

    class IterationSpaceExpander(ast.NodeTransformer):
        def __init__(self, index_name, reference_array_shape):
            self.index_name = index_name
            self.reference_array_shape = reference_array_shape

        def visit_IterationSpace(self, node):
            node = self.generic_visit(node)

            make_low = lambda low, dim: Constant(low) if low >= 0 else Constant(self.reference_array_shape[dim] + low)
            make_high = lambda high, dim: Constant(high) if high > 0 else Constant(self.reference_array_shape[dim] + high)
            parts = []
            for space in node.space.spaces:
                inside = node.body
                for dim, iteration_range in reversed(list(enumerate(zip(*[space.low, space.high, space.stride])))):
                    inside = [
                        For(
                            init=Assign(SymbolRef("{}_{}".format(self.index_name, dim)), make_low(iteration_range[0], dim)),
                            test=Lt(SymbolRef("{}_{}".format(self.index_name, dim)), make_high(iteration_range[1], dim)),
                            incr=AddAssign(SymbolRef("{}_{}".format(self.index_name, dim)), Constant(iteration_range[2] or 1)),
                            body=inside
                        )
                    ]
                parts.extend(inside)
            return MultiNode(parts)

    class ConcreteSpecializedKernel(ConcreteSpecializedFunction):
        def finalize(self, entry_point_name, project_node, entry_point_typesig):
            self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
            self.entry_point_name = entry_point_name
            return self

        def __call__(self, *args, **kwargs):
            res = self._c_function(*args)
            return res

    class LazySpecializedKernel(Compiler.SpecializationKernel):
        def __init__(self, py_ast=None, names=None, target_names=('out',), index_name='index',
                     _hash=None, original=None):
            self.__hash = _hash if _hash is not None else hash(ast.dump(py_ast))
            self.names = names
            self.target_names = target_names
            self.index_name = index_name

            super(CCompiler.LazySpecializedKernel, self).__init__(py_ast, 'snowflake_' + hex(hash(self)))
            self.parent_cls = CCompiler
            self.original = original
            self.__original_hash = str(hash(ast.dump(self.original_tree)))

        def config_to_dirname(self, program_config):
            """
            Much stripped down version
            """
            regex_filter = re.compile(r"""[/\?%*:|"<>()'{} -]""")

            path_parts = [
                'snowflake',
                self.__original_hash,
                str(self._hash(program_config.args_subconfig)),
                str(self._hash(program_config.tuner_subconfig)),
                str(self.parent_cls.__name__),
                ]
            path_parts = [re.sub(regex_filter, '_', part) for part in path_parts]
            compile_path = str(ctree.CONFIG.get('jit', 'COMPILE_PATH'))
            path = os.path.join(compile_path, *path_parts)
            final = re.sub('_+', '_', path)
            return final

        @property
        def arg_spec(self):
            return tuple(list(set(self.target_names)) + list(sorted(set(self.names) - set(self.target_names))))

        def __hash__(self):
            return self.__hash + hash((tuple(self.target_names), self.index_name))

        class Subconfig(OrderedDict):
            def __hash__(self):
                return hash(tuple(sorted((name, arg.shape, str(arg.dtype)) for name, arg in sorted(self.items()))))

        def args_to_subconfig(self, args):
            names_to_use = self.arg_spec
            subconf = self.Subconfig()
            for name, arg in zip(names_to_use, args):
                subconf[name] = arg
            return subconf

        def transform(self, tree, program_config):
            subconfig, tuning_config = program_config
            name_shape_map = {name: arg.shape for name, arg in subconfig.items()}
            shapes = set(name_shape_map.values())
            self.parent_cls.IndexOpToEncode(name_shape_map).visit(tree)
            encode_funcs = []
            c_tree = PyBasicConversions().visit(tree)
            # print(dump(c_tree))
            for shape in shapes:
                encode_funcs.append(generate_encode_macro('encode'+CCompiler._shape_to_str(shape), shape))
            components = []
            for target, ispace in zip(self.target_names, c_tree.body):
                shape = subconfig[target].shape
                sub = fill_iteration_spaces(ispace, shape)
                sub = self.parent_cls.IterationSpaceExpander(self.index_name, shape).visit(sub)
                sub = self.parent_cls.BlockConverter().visit(sub)
                components.append(sub)

            c_func = FunctionDecl(
                name=SymbolRef("kernel"),
                params=[
                    SymbolRef(name=arg_name, sym_type=get_ctype(
                        arg if not isinstance(arg, np.ndarray) else arg.ravel()
                    ), _restrict=True) for arg_name, arg in subconfig.items()
                ],
                defn=components
            )
            TilingOptimization().optimize(c_func, self.tile_size)
            includes = [
                CppInclude("stdint.h")
            ]
            out_file = CFile(body=includes + encode_funcs + [c_func])
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
        kern = self.LazySpecializedKernel(
            py_ast=compiled,
            names=find_names(original),
            index_name=index_name,
            target_names=[stencil.primary_mesh for stencil in original.body if hasattr(stencil, "primary_mesh")],
            _hash=hash(original),
            original=original
        )
        kern.tile_size = self.tile_size
        return kern
