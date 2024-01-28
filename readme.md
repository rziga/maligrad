![logo](https://github.com/rziga/maligrad/assets/102856773/11f60388-14b5-4a8d-9b9e-3d3a5ab34e5a)

### [maligrad](https://maps.app.goo.gl/kZGt592YTsWSrcWA9) is a small numpy-array-based autograd engine with an included NN library

It can do majority of things numpy can:
```py

a = Variable(2 * np.ones((3, 3)), requires_grad=True)
b = Variable(3 * np.ones((3, 3)))

# addition
print(a + b)
>>> Variable(
    data=
    [[5. 5.]
     [5. 5.]]
    )

# multiplication
print(a * b)
>>> Variable(
    data=
    [[6. 6.]
     [6. 6.]]
    )

# matrix multiplication
print(a @ b[0])
>>> Variable(
    data=
    [12. 12.]
    )

# and much more
```
while also enabling reverse mode automatic differentiation:
```py
(a[0] * b).sum().backward()
print(a.grad)
>>> [[6., 6.],
     [0., 0.]]
```

# Info

The project was inspired by the great [micrograd](https://github.com/karpathy/micrograd).
If you watched the video tutorial or implemented micrograd yourself, you know that that engine is based on scalars.
While being really simple, it is still powerful enough to define pretty much any neural net, that approach is not as efficient.

So, I wondered how much more work would be needed to make a similar automatic differentiation engine, but based on tensors / $N$-dimensional arrays, like it's done in [pytorch](https://github.com/pytorch/pytorch), for example.
I thought that things like broadcasting, indexing and all the other special array operations would complicate things a great deal.

It turns out that not that much is different.
For comparison, micrograd's engine is around 70 lines of code, while maligrad has roughly 330 lines under the hood.
And that includes all the extra ops that are not included in micrograd.
At the end the broadcasting shenanigans sorted themselves out on their own (I attribute this to my great planning and look-ahead. No blind luck was involved.).

# TODOs

I will probably never come around to do any of these :)

1) Inplace modifications are not supported and never will be. But some operations, like padding in convolution, would require a similiar operation.

2) Currently, maligrad is pretty memory inefficient, because the compute graphs are longer than they need to be, since only a few basic ops have `Function` implementations. For example: division is implemented as `a / b = a * b ** -1`, which produces 2 nodes in DAG. This is not a lone example, so operator overrides at least should have an associated `Function`.

3) RNNs and other nn stuff.