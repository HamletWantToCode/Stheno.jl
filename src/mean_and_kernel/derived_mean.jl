import Stheno: MeanFunction, ew, AV, mean, kernel
using Stheno: ZeroMean, OneMean, EQ, ZeroKernel



#
# Derived Mean
#

struct DerivedMean{Tmodel, Targs} <: MeanFunction
    model::Tmodel
    args::Targs
end

ew(m::DerivedMean, x::AV) = ew_derived_mean(m.model, m.args..., x)

function ew_derived_mean(model, ::typeof(+), f1::Val, f2::Val, x::AV)
    return ew(mean(model, f1), x) + ew(mean(model, f2), x)
end



#
# Derived Kernel
#

struct DerivedKernel{Tmodel, Targs}
    model::Tmodel
    args::Targs
end

ew(k::DerivedKernel, a::AV, b::AV) = ew_derived_kernel(k.model, k.args..., a, b)

function ew_derived_kernel(model, ::typeof(+), f1::Val, f2::Val, a::AV, b::AV)
    return ew(kernel(model, f1), a, b) + ew(kernel(model, f2), a, b)
end

const AddArgs = Tuple{typeof(+), Val, Val}

function ew_derived_kernel(model, (_, fa, fb)::AddArgs, fq::Val, x, x′)
    return ew(cross_kernel(model, fa, fq), x, x′) + ew(cross_kernel(model, fb, fq), x, x′)
end

function ew_derived_kernel(model, fq::Val, (_, fa, fb)::AddArgs, x, x′)
    return ew(cross_kernel(model, fq, fa), x, x′) + ew(cross_kernel(model, fq, fb), x, x′)
end



#
# New type of GPs that work with the above means and kernels.
#

struct GP{Tmodel, Tindex}
    model::Tmodel
    index::Tindex
end

mean(f::GP) = mean(f.model, f.index)
mean(model, index) = mean(model, index, is_primitive(model, index))
mean(model, index, ::Val{true}) = first(args(model, index))
mean(model, index, ::Val{false}) = DerivedMean(model, args(model, index))


kernel(f::GP) = kernel(f.model, f.index)
kernel(model, index) = kernel(model, index, is_primitive(model, index))
kernel(model, index, ::Val{true}) = last(args(model, index))
kernel(model, index, ::Val{false}) = DerivedKernel(model, args(model, index))

kernel(fa::GP{Tm}, fb::GP{Tm}) where {Tm} = cross_kernel(fa.model, fa.index, fb.index)
kernel(fa::GP, fb::GP) = error("processes not from same programme")

cross_kernel(model, fa::Val{N}, fb::Val{N}) where {N} = kernel(model, fa)

function cross_kernel(model, fa::Val, fb::Val)
    return cross_kernel(model, fa, fb, is_primitive(model, fa), is_primitive(model, fb))
end

cross_kernel(model, fa::Val, fb::Val, va::Val{true}, vb::Val{true}) = ZeroKernel()

function cross_kernel(model, fa::Val{Na}, fb::Val{Nb}, ::Val, ::Val) where {Na, Nb}
    if Na > Nb
        return DerivedKernel(model, (args(model, fa), fb))
    else
        return DerivedKernel(model, (fa, args(model, fb)))
    end
end

# Fallback method to prevent process from different programmes interacting incorrectly

# abstract type AbstractGP{Tmodel, Tindex} end

# kernel(fa::AbstractGP{Tm}, fb::AbstractGP{Tm}) = kernel(fa.model, fa, fb)

# # A Gaussian process whose mean and kernel are known.
# struct PrimitiveGP{Tmodel, Tindex, Tm, Tk} <: AbstractGP{Tmodel, Tindex}
#     model::Tmodel
#     index::Tindex
#     m::Tm
#     k::Tk
# end

# mean(f::PrimitiveGP) = f.m
# kernel(f::PrimitiveGP) = f.k
# kernel(fa::PrimitiveGP{Tm}, fb::PrimitiveGP{Tm}) where {Tm} = ZeroKernel()


# # A Gaussian process whose mean and kernel are specified in terms of other GPs.
# struct DerivedGP{Tmodel, Tindex, Targs} <: AbstractGP{Tmodel, Tindex}
#     model::Tmodel
#     index::Tindex
# end

# mean(f::DerivedGP) = DerivedMean(f.model, f.args)
# kernel(f::DerivedGP) = DerivedKernel(f.model, f.args)





#
# A dummy model that will eventually be generated automatically from an input programme
#

# dummy_model(::Val{1}) = PrimitiveGP(dummy_model, Val(1), ZeroMean(), EQ())
# dummy_model(::Val{2}) = PrimitiveGP(dummy_model, Val(2), OneMean(), EQ())
# dummy_model(::Val{3}) = DerivedGP(dummy_model, Val(3), (+, Val(1), Val(2)))

dummy_model(::Val{N}) where {N} = GP(dummy_model, Val(N))

args(::typeof(dummy_model), ::Val{1}) = (ZeroMean(), EQ())
args(::typeof(dummy_model), ::Val{2}) = (OneMean(), EQ())
args(::typeof(dummy_model), ::Val{3}) = (+, Val(1), Val(2))

args(f::GP) = args(f.model, f.index)

is_primitive(::typeof(dummy_model), ::Val{1}) = Val(true)
is_primitive(::typeof(dummy_model), ::Val{2}) = Val(true)
is_primitive(::typeof(dummy_model), ::Val{3}) = Val(false)


# mean(::typeof(dummy_model), f::Val) = mean(dummy_model(f))
# kernel(::typeof(dummy_model), f::Val) = kernel(dummy_model(f))

# Get processes
f1 = dummy_model(Val(1))
f2 = dummy_model(Val(2))
f3 = dummy_model(Val(3))

# Get means
m1 = mean(f1)
m2 = mean(f2)
m3 = mean(f3)

x = collect(range(0.0, 1.0; length=5));

# Compute means at x
μ1 = ew(m1, x)
μ2 = ew(m2, x)
μ3 = ew(m3, x)


# Get kernels
k1 = kernel(f1)
k2 = kernel(f2)
k3 = kernel(f3)

# Compute ew on kernels at x
ew(k1, x, x)
ew(k2, x, x)
ew(k3, x, x)


#
# Computing cross-kernels (maybe the tricky bit?)
#

kernel(f1, f1)
kernel(f1, f2)
kernel(f1, f3)

kernel(f2, f1)
kernel(f2, f2)
kernel(f2, f3)

kernel(f3, f1)
kernel(f3, f2)
kernel(f3, f3)

# need another derived process to complete this.


