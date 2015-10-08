# Snowflake
new eDSL for stencils in python

## What is a stencil?
Stencils are neighborhood operations defined over a grid. For example, a 1-D neighborhood average might look like $\text{Out}[x] = \frac{1}{4} \text{In}[x-1] + \frac{1}{2} \text{In}[x] + \frac{1}{4} \text{In}[x+1]$. Traditionally, these constructs have been expressed through the use of loops.

```py
def linear_avg(arr):
	output = np.zeros_like(arr)
	for i in range(1, len(arr)-1):
		output[i] = 0.25*arr[i-1] + 0.5*arr[i] + 0.25*arr[i+1]
	return arr
```

This model works perfectly well until the problem begins to scale into higher dimensions (like 6-D space), or have variable coefficients, or even complex operations at each point.

##Why Snowflake?
Snowflake is a lightweight DSL designed for representing things as they are. Complex stencils can be composed as combinations of simpler ones, and switching platforms (i.e. Python vs. C vs. OpenCL) can be done by simply switching the compiler.

##Snowflake usage

Simple stencils can be defined by simply creating a coefficient array and naming the corresponding variable name.

```python
weights = WeightArray([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
s = StencilComponent("a", weights)
simple_stencil = Stencil(s, 'out', ((1, -1, 1),)) # from 1 to -1 step by 1
kernel = ccompiler.compile(simple_stencil)
```

More complicated Variable Coefficient Stencils can be created by putting `StencilComponents` in as weights to a `WeightArray`.

```python
variable_coefficient_w1 = WeightArray(np.arange(9).reshape((3, 3)).tolist())
variable_coefficient_w2 = WeightArray([[3]])
variable_coefficient_w1[1][1] += StencilComponent("b", variable_coefficient_w2)
vc_stencil = Stencil(StencilComponent("a", variable_coefficient_w1))
vc_kernel = ccompiler.compile(vc_stencil, 'out', ((1, -1, 1),))
```
