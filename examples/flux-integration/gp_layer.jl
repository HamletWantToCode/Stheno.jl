#
# Some very toy examples explaining how you might integrate a Stheno.jl model inside Flux.jl
# This is basically just sampling from the prior of a GP. The real challenge is figuring out
# how to include inference in this API.
#

# Set up the project. Quite restrictive version control has been employed, so bit-rot should
# be limited. If you find you are unable to run this example for any reason, please raise
# and issue on the Stheno.jl repo.
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Stheno, Zygote, Flux
using Stheno: AbstractGP, Stretched, GP



#
# Implement a layer that can be used inside a Flux Chain.
#

# Add some Flux + Stheno compat stuff. Could be internalised.
Flux.@functor Stretched
Flux.@functor GP

# Specify the GP layer.
struct GPLayer{Tf<:AbstractGP}
    f::Tf
    odims::Int
end

Flux.@functor GPLayer

(gpl::GPLayer)(x) = transpose(rand(gpl.f(ColVecs(x), 1e-9), gpl.odims))



#
# Compose a GP and a couple of Flux layers.
#

# Make a Stheno.jl GP
f = GP(stretch(matern32(), [1.0]), GPC())

# Build a Flux.jl model.
model = Chain(Dense(3, 10), GPLayer(f, 2), Dense(2, 1))

# Sample some random input locations.
x = randn(3, 11)

# Model forwards-pass.
model(x)

# Model reverse-pass.
gs = Zygote.gradient(()->sum(model(x)), params(model))



#
# Compose a couple of GPs to make a Deep GP.
#

deep_gp = Chain(
    GPLayer(GP(stretch(matern32(), [1.0]), GPC()), 10),
    GPLayer(GP(stretch(matern52(), [0.5]), GPC()), 1),
)

# Forwards-pass.
deep_gp(x)

# Model reverse-pass.
gs = Zygote.gradient(()->sum(deep_gp(x)), params(deep_gp))
