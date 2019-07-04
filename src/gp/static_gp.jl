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



#
# A dummy model that will eventually be generated automatically from an input programme
#

dummy_model(::Val{N}) where {N} = GP(dummy_model, Val(N))

args(::typeof(dummy_model), ::Val{1}) = (ZeroMean(), EQ())
args(::typeof(dummy_model), ::Val{2}) = (OneMean(), EQ())
args(::typeof(dummy_model), ::Val{3}) = (+, Val(1), Val(2))



is_primitive(::typeof(dummy_model), ::Val{1}) = Val(true)
is_primitive(::typeof(dummy_model), ::Val{2}) = Val(true)
is_primitive(::typeof(dummy_model), ::Val{3}) = Val(false)

args(f::GP) = args(f.model, f.index)
is_primitive(f::GP) = is_primitive(f.model, f.index)

#
# Test stuff out
#

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

#
# A static model, from which we derive stuff automatically
#

@static_model function foo(θ)
    f1 = GP(m1(θ), k1(θ))
    f2 = GP(m2(θ), k2(θ))
    f3 = +(f1, f2)
end

# becomes

struct StaticModel{Tmodel, Targs}
    model::Tmodel
    args::Targs
end

foo(args...) = StaticModel(foo, args)

(s::StaticModel{typeof(foo)})(::Val{N}) where {N} = GP(foo, Val(N))

function args(s::StaticModel{typeof(foo)}, ::Val{1})
    θ, = args
    return m1(θ), k1(θ)
end

function args(s::StaticModel{typeof(foo)}, ::Val{2})
    θ, = args
    return m2(θ), k2(θ)
end

function args(s::StaticModel{typeof(foo)}, ::Val{3})
    θ, = args
    return (+, Val(:f1), Val(:f2))
    # Need to assume unique naming on the left for this to work :(
end

# Functions mapping from symbolic names to Val-names.
_map_name(::Val{:f1}) = Val(1)
_map_name(::Val{:f2}) = Val(2)
_map_name(::Val{:f3}) = Val(3)


#
# Some operations on the resulting process
#

using MacroTools

macro static_model(code)

    # Define StaticModel constructor for the model in code
    code_dict = copy(splitdef(code))
    static_model_dict = copy(code_dict)
    static_model_dict[:body] = Expr(
        :call,
        :StaticModel,
        code_dict[:name],
        Expr(:tuple, namify.(code_dict[:args])...),
    )

    # Define GP generator
    name = code_dict[:name]
    gp_gen = :((s::StaticModel{typeof($name)})(::Val{N}) where {N} = GP($name, Val(N)))

    # Define name maps
    lines = rmlines(code_dict[:body]).args
    process_names = [first(line.args) for line in lines]
    gen_name_map(n) = :(_map_name(::typeof($name), ::Val{$(process_names[n])}) = Val($n))
    name_maps = gen_name_map.(eachindex(lines))

    # Define is_primtive
    gen_is_primitive(n, bool) = :(is_primitive(::typeof($name), ::Val($n)) = Val($bool))
    is_gp(line) = line.args[2].args[1] == :GP
    is_primitives = map(is_gp, lines)
    is_primitive_exprs = gen_is_primitive.(eachindex(lines), is_primitives)

    # Define args
    # MODIFY THIS CODE TO REPLACE THE ARGUMENTS WITH THE VAL-ED VERSIONS.
    static_model_name = gensym("static_model")
    function gen_args_expr(n)

        # Produce first line
        args_lhs = Expr(:tuple, namify.(code_dict[:args])...)
        args_rhs = :($static_model_name.args)
        first_line = Expr(:(=), args_lhs, args_rhs)

        # Produce second line
        

        return nothing
    end
    args_exprs = gen_args_expr.(eachindex(lines))

    return esc(:nothing)
end

@static_model function bar(θ, b::T, c::A) where {T, A}
    f1 = GP(m1(θ), k1(θ))
    f2 = GP(m2(θ), k2(θ))
    f3 = +(f1, f2)
end

model = foo(θ)










