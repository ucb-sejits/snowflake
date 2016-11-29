from snowflake_opencl.compiler import OpenCLCompiler

from snowflake.stencil_compiler import Compiler
import numpy as np


class ComparisonCompiler(Compiler):
    def __init__(self, comparisons=[]):
        super(ComparisonCompiler, self).__init__()
        self.comparisons = comparisons

    def _post_process(self, original, compiled, index_name, **kwargs):
        kernels = [comp._post_process(original, compiled, index_name, **kwargs) for comp in self.comparisons]

        def callable(*args):
            outputs = []
            for kernel, compiler in zip(kernels, self.comparisons):
                copied = [arg.copy() for arg in args]
                kernel(*copied)
                if isinstance(compiler, OpenCLCompiler):
                    for arg in copied:
                        arg.gpu_to_device(wait=True, force=True)
                outputs.append(copied)
            for out in outputs[1:]:
                for out0, out1 in zip(outputs[0], out):
                    if not np.allclose(out1[1:-1, 1:-1, 1:-1], out0[1:-1, 1:-1, 1:-1], rtol=0.01):
                        raise Exception("Mismatch")

            kernels[0](*args)
        return callable