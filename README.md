# stencil
new eDSL for stencils in python

Simple stencils can be defined by simply creating a coefficient array and naming the corresponding variable name.

```python
weights = WeightArray([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
s = StencilComponent("a", weights)
simple_stencil = Stencil(s)
kernel = simple_stencil.get_kernel()
```

More complicated Variable Coefficient Stencils can be created by putting `StencilComponents` in as weights to a `WeightArray`.

```python
variable_coefficient_w1 = WeightArray(np.arange(9).reshape((3, 3)).tolist())
variable_coefficient_w2 = WeightArray([[3]])
variable_coefficient_w1[1][1] += StencilComponent("b", variable_coefficient_w2)
vc_stencil = Stencil(StencilComponent("a", variable_coefficient_w1))
vc_kernel = vc_stencil.get_kernel()
```
