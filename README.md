This code supports the preprint [preprintlink], in particular section [sectionnumber]. This code checks the derived formal gradient of a tokenwise mean squared error loss backpropagated through one multihead linear self-attention block against numerical approximations based on finite differences and forward automatic differentiation. 

The results can be replicated by first installing [Julia](https://julialang.org/downloads/) if necessary, and the necessary packages
```console
julia -e 'using Pkg; Pkg.add.(["Random", "ForwardDiff", "LinearAlgebra"])'
```
and launching the single script
```console
cd Formal-MHSA-gradient-numerics-main/
julia numerics.jl
```
