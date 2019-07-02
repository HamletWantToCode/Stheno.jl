using Turing, Bijectors
import Bijectors: link, invlink, logpdf_with_trans



"""
    TuringGPC{Tlml<:Real} <: AbstractGPC

`TuringGPC` := "Turing GP Coordinator". This is a book-keeping object that ensures that

"""
mutable struct TuringGPC <: AbstractGPC
    n::Int
    logp_obs::Vector{Observation}
    rand_obs::Vector{Observation}
    TuringGPC() = new(0, Vector{Observation}(undef, 0), Vector{Observation}(undef, 0))
end

# A Turing FiniteGP is just a FiniteGP with a TuringGPC, as opposed to another AbstractGPC.
const TuringFiniteGP{Tm<:MeanFunction, Tk<:CrossKernel} = FiniteGP{<:GP{Tm, Tk, <:TuringGPC}}

function rand(rng::AbstractRNG, fx::TuringFiniteGP, N::Int)
    gpc = fx.f.gpc
    if isempty(gpc.rand_obs)
        y = _rand(rng, fx, N)
    else
        y = _rand(rng, (fx.f | gpc.rand_obs)(fx.x, fx.Σy), N)
    end
    gpc.rand_obs = vcat(gpc.rand_obs, fx ← y)
    return y
end

function rand(rng::AbstractRNG, fx::TuringFiniteGP)
    gpc = fx.f.gpc
    if isempty(gpc.rand_obs)
        y = _rand(rng, fx)
    else
        y = _rand(rng, (fx.f | gpc.rand_obs)(fx.x, fx.Σy))
    end
    gpc.rand_obs = vcat(gpc.rand_obs, fx ← y)
    return y
end

function logpdf(fx::TuringFiniteGP, y::AbstractVector{<:Real})
    gpc = fx.f.gpc
    if isempty(gpc.logp_obs)
        l = _logpdf(fx, y)
    else
        l = _logpdf((fx.f | gpc.logp_obs)(fx.x, fx.Σy), y)
    end
    gpc.logp_obs = vcat(gpc.logp_obs, fx ← y)
    return l
end



#
# Define transformations for efficient sampling
#

function link(fx::TuringFiniteGP, y::AbstractVector{<:Real})
    return cholesky(cov(fx)).U' \ (y - mean(fx))
end

function invlink(fx::TuringFiniteGP, ε::AbstractVector{<:Real})
    return mean(fx) + cholesky(cov(fx)).U' * ε
end

function logpdf_with_trans(fx::TuringFiniteGP, ε::AbstractVector{<:Real}, transform::Bool)
    if transform
        return -(length(ε) * log(2π) + logdet(cholesky(cov(fx))) + sum(abs2, ε)) / 2
    else
        return _logpdf(fx, ε)
    end
end



#
# Ensure that users use the correct type of `AbstractGPC` object.
#

function Turing.assume(
    ::Turing.AbstractSampler,
    ::GP{<:MeanFunction, <:CrossKernel, <:GPC},
    ::Any,
    ::Turing.VarInfo,
)
    error("GP with `GPC` found, expected GP with `TuringGPC`.")
end

function Turing.observe(
    ::Turing.AbstractSampler,
    ::GP{<:MeanFunction, <:CrossKernel, <:GPC},
    ::Any,
    ::Turing.VarInfo,
)
    error("GP with `GPC` found, expected GP with `TuringGPC`.")
end



#
# Disallow the usual Turing operations with infinite-dimensional objects.
#

function Turing.assume(::Turing.AbstractSampler, ::GP, ::Any, ::Turing.VarInfo)
    error("You cannot sample from a `GP`. Construct a `FiniteGP` (i.e. `f(x)`) to sample.")
end

function Turing.observe(::Turing.AbstractSampler, ::GP, ::Any, ::Turing.VarInfo)
    error("You cannot observe an a `GP`. Construct a `FiniteGP` (i.e. `f(x)`) to observe.")
end
