import ast
from collections import Iterable
import itertools
from _compiler import StencilCompiler, find_names

__author__ = 'nzhang-dev'

class Compiler(object):

    def __init__(self, index_name="index"):
        self.index_name = index_name

    def _compile(self, node, index_name):
        raise NotImplementedError("Compile is not implemented")

    def _post_process(self, compiled):
        raise NotImplementedError("PostProcessing isn't implemented")

    def compile(self, node):
        compiled = self._compile(node, self.index_name)
        processed = self._post_process(node, compiled, self.index_name)
        return processed


class PythonCompiler(Compiler):

    def _compile(self, node, index_name):
        #print("compile")
        return StencilCompiler(index_name).visit(node)

    def _post_process(self, original, compiled, index_name):
        #print(compiled)
        nodes = list(compiled) if isinstance(compiled, Iterable) else [compiled]
        if all(not isinstance(n, ast.Return) for n in itertools.chain(ast.walk(node) for node in nodes)):
            nodes[-1] = ast.Return(nodes[-1])

        array_names = find_names(original)
        print(array_names)

        function_name = 'kernel'
        args = [
            index_name,
        ]
        args.extend(sorted(array_names))
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
        tree = ast.Module(body=[tree])
        tree = ast.fix_missing_locations(tree)
        code = compile(tree, '<string>', 'exec')
        exec code in globals(), locals()
        return locals()['kernel']