import ast
import functools
import operator

__author__ = 'nzhang-dev'


class StencilCompiler(ast.NodeVisitor):

    operator_to_ast_map = {
        operator.add: ast.Add(),
        operator.sub: ast.Sub(),
        operator.mul: ast.Mult(),
        operator.div: ast.Div()
    }

    def __init__(self, index_name='index'):
        self.index_name = index_name

    def visit_StencilConstant(self, node):
        return ast.Num(n=node.value, ctx=ast.Load())

    def visit_StencilComponent(self, node):
        weights = [self.visit(weight) for weight in node.weights.weights]
        components = [
            ast.BinOp(
                left=ast.Subscript(
                    value=ast.Name(
                        id=node.name,
                        ctx=ast.Load()
                    ),
                    slice=ast.Index(
                        value=ast.BinOp(
                            left=ast.Name(id=self.index_name, ctx=ast.Load()),
                            op=ast.Add(),
                            right=ast.Tuple(
                                elts=[ast.Num(n=i) for i in vector],
                                ctx=ast.Load()
                            )
                        )
                    )
                ),
                op=ast.Mult(),
                right=weight
            ) for weight, vector in zip(weights, node.weights.vectors)
        ]
        return functools.reduce(
            lambda x, y: ast.BinOp(left=x, right=y, op=ast.Add()),
            components
        )

    def visit_StencilOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = self.operator_to_ast_map[node.op]
        return ast.BinOp(left=left, op=op, right=right)


