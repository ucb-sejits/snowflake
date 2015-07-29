import abc
import ast
from collections import Iterable
import copy
import ctypes
import itertools
from ctree.c.nodes import String, FunctionCall, SymbolRef, FunctionDecl, For, Constant, Assign, Lt, AugAssign, AddAssign
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.transformations import PyBasicConversions
from ctree.frontend import dump
import math
from rebox.specializers.order import Ordering
from _compiler import StencilCompiler, find_names, ArrayOpRecognizer, OpSimplifier
from nodes import WeightArray

from rebox.specializers.rm.encode import MultiplyEncode

__author__ = 'nzhang-dev'

class Compiler(object):

    @staticmethod
    def get_ndim(node):
        weight_array = next(n for n in ast.walk(node) if isinstance(n, WeightArray))
        return len(weight_array.center)

    def __init__(self, index_name="index"):
        self.index_name = index_name

    def _compile(self, node, index_name, **kwargs):
        ndim = self.get_ndim(node)
        stack = [
            StencilCompiler(index_name, ndim),
            OpSimplifier(),
            ArrayOpRecognizer(index_name, ndim),
        ]
        output = node
        for transformer in stack:
            #print(transformer)
            #print(dump(output))
            output = transformer.visit(output)
        #print(dump(output))
        return output

    def _post_process(self, original, compiled, index_name, **kwargs):
        raise NotImplementedError("PostProcessing isn't implemented")

    def compile(self, node, **kwargs):
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
        def __init__(self, index_name):
            self.index_name = index_name

        def visit_IterationSpace(self, node):
            self.generic_visit(node)
            nested = node.body
            for dim, iteration_range in reversed(tuple(enumerate(node.space))):
                nested = ast.For(
                    target=ast.Name(id="{}_{}".format(self.index_name, dim), ctx=ast.Store()),
                    iter=ast.Call(
                        func=ast.Name(id="range", ctx=ast.Load()),
                        args=[
                            ast.Num(n=i) if isinstance(i, int) else ast.Name(id=i, ctx=ast.Load())
                            for i in iteration_range
                        ],
                        keywords=[],
                        starargs=None,
                        kwargs=None
                    ),
                    body=[nested] if not isinstance(nested, list) else nested,
                    orelse=[]
                )
            return nested

    def _post_process(self, original, compiled, index_name):
        #print(compiled)
        target_name = original.output
        nodes = list(compiled) if isinstance(compiled, (list, tuple)) else [compiled]
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
            body=nodes,
            decorator_list=[]
        )
        # print("hello")
        tree = ast.Module(body=[tree])
        tree = self.BasicIndexOpConverter().visit(tree)
        tree = self.IterationSpaceConverter(self.index_name).visit(tree)
        # for p in ast.walk(tree):
        #     print(p, type(p))
        # print(dump(tree))
        tree = ast.fix_missing_locations(tree)
        code = compile(tree, '<string>', 'exec')
        exec code in globals(), locals()
        return locals()['kernel']



class CCompiler(Compiler):

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
        def __init__(self, index_name):
            self.index_name = index_name

        def visit_IterationSpace(self, node):
            node = self.generic_visit(node)
            inside = node.body
            for dim, iteration_range in reversed(list(enumerate(node.space))):
                inside = [
                    For(
                        init=Assign(SymbolRef("{}_{}".format(self.index_name, dim)), Constant(iteration_range.low)),
                        test=Lt(SymbolRef("{}_{}".format(self.index_name, dim)), Constant(iteration_range.high)),
                        incr=AddAssign(SymbolRef("{}_{}".format(self.index_name, dim)), Constant(iteration_range.stride)),
                        body=inside
                    )
                ]
            return inside[0]


    class ConcreteSpecializedKernel(ConcreteSpecializedFunction):
        pass

    class LazySpecializedKernel(LazySpecializedFunction):
        def __init__(self, py_ast=None, sub_dir=None, backend_name="default", names=None, target_name='out', index_name='index'):
            #print(dump(py_ast))
            super(CCompiler.LazySpecializedKernel, self).__init__(py_ast, sub_dir, backend_name)
            self.names = names
            self.target_name = target_name
            self.index_name = index_name

        def args_to_subconfig(self, args):
            names_to_use = [self.target_name] + list(sorted(self.names - {self.target_name}))
            return {
                name: arg for name, arg in zip(names_to_use, args)
            }

        def transform(self, tree, program_config):

            subconfig, tuning_config = program_config
            CCompiler.IndexOpToEncode().visit(tree)
            ndim = subconfig[self.target_name].ndim
            c_tree = PyBasicConversions().visit(tree)
            ordering = Ordering([MultiplyEncode()])
            bits_per_dim = min([math.log(i, 2) for i in subconfig[self.target_name].shape]) + 1
            encode_func = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)
            c_func = FunctionDecl(
                name=SymbolRef("kernel"),
                params=[],
                defn=[c_tree]
            )
            c_func = CCompiler.IterationSpaceExpander(self.index_name).visit(c_func)
            print(c_func)
            print(encode_func)


        def finalize(self, transform_result, program_config):
            pass

    def _post_process(self, original, compiled, index_name, **kwargs):
        lsk = self.LazySpecializedKernel(
            py_ast=self.IndexOpToEncode().visit(compiled),
            names=find_names(original),
            index_name=index_name,
            target_name=original.output
        )
        return lsk
        raise Exception()