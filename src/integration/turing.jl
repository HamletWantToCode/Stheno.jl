using Turing

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

function rand(rng::AbstractRNG, fx::TuringFiniteGP)
    gpc = fx.f.gpc
    if isempty(gpc.rand_obs)
        y = _rand(rng, fx)
    else
        y = _rand(rng, (fx.f | gpc.rand_obs)(fx.x, fx.Σ))
    end
    gpc.rand_obs = vcat(gpc.rand_obs, fx ← y)
    return y
end

function logpdf(fx::TuringFiniteGP, y::AbstractVector{<:Real})
    gpc = fx.f.gpc
    if isempty(gpc.logp_obs)
        l = _logpdf(fx, y)
    else
        l = _logpdf((fx.f | gpc.logp_obs)(fx.x, fx.Σ), y)
    end
    gpc.logp_obs = vcat(gpc.logp_obs, fx ← y)
    return l
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
