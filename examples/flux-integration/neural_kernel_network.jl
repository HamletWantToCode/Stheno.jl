#
# A baseline implementation of the Neural Kernel Network in conjunction with Flux and Zygote
#

# Set up the project. Quite restrictive version control has been employed, so bit-rot should
# be limited. If you find you are unable to run this example for any reason, please raise
# and issue on the Stheno.jl repo.
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Stheno, Zygote, Flux



#
# Implement some layers for the NKN. I've not implemented support for complex numbers or the
# multiplication layers, this being a prototype, but they're conceptually similar to the
# linear layers. This section could be moved inside Stheno. It's also very much a prototype
# to show what could be done, as opposed to being anything like a polished implementation.
# For example, it's basically untested.
#

# Linear layers. Assume positivity has been handled, although this isn't strictly necessary.
const Linear = Dense{<:typeof(identity)}

# Activation layers. Only got the exp for now.
struct ExpActivation end
(::ExpActivation)(X::AbstractMatrix{<:Real}) = exp.(X)



# Implement the standard Stheno kernel interface. See documentation and src/kernel.jl for
# examples.
import Stheno: ew, pw, Kernel

"""
    NKN{Tprimitives<:AbstractVector{<:Kernel}, Tchain<:Chain} <: Kernel

The Neural Kernel Network (NKN). <insert reference to paper here>
"""
struct NKN{Tprimitives<:AbstractVector{<:Kernel}, Tchain<:Chain} <: Kernel
    primitives::Tprimitives
    chain::Tchain
end

# Helper function.
function _apply_chain(nkn::NKN, Ks::Vector{<:AbstractMatrix})
    ks = map(K->reshape(K, :), Ks)
    K = transpose(hcat(ks...))
    return reshape(nkn.chain(K), size(first(Ks)))
end

# Convenience definition.
const AV = AbstractVector

# Unary methods
ew(nkn::NKN, x::AV) = _apply_chain(nkn, map(k->ew(k, x), nkn.primitives))
pw(nkn::NKN, x::AV) = _apply_chain(nkn, map(k->pw(k, x), nkn.primitives))

# Binary methods
ew(nkn::NKN, x::AV, x′::AV) = _apply_chain(nkn, map(k->ew(k, x, x′), nkn.primitives))
pw(nkn::NKN, x::AV, x′::AV) = _apply_chain(nkn, map(k->pw(k, x, x′), nkn.primitives))





#
# Specify a GP with the NKN.
#

# ls are length scales, one for each primitive kernel, and g is an MLP.
# As before, we have this slight clash of paradigms, so have to pass in Stheno parameters.
# Should definitely be possible to resolve this with only a little work.
function build_model(ls, g)
    return GP(NKN(stretch.(Ref(eq()), 1 ./ ls), g), GPC())
end

# Initialise length-scales and the MLP
ls = [0.1, 1.3, 0.7]
g = Chain(
    Dense(softplus.(randn(5, 3)), softplus.(randn(5))),
    ExpActivation(),
    Dense(softplus.(randn(1, 5)), softplus.(randn(1))),
)
f = build_model(ls, g)

# Draw some samples and compute their log marginal likelihood.
x = range(0.0, 10.0; length=13)
y = rand(f(x, 0.1))

# Compute the logpdf (just because we can)
logpdf(f(x, 0.1), y)

# Compute the gradient w.r.t. everything.
dls, dg = Zygote.gradient(
    function(ls, g)
        f = build_model(ls, g)
        return logpdf(f(x, 0.1), y)
    end,
    ls, g,
)
