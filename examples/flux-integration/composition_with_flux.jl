#
# Some very toy examples explaining how you might integrate a Flux model within Stheno.
# At the bottom of the file is the baseline.
#

# Set up the project. Quite restrictive version control has been employed, so bit-rot should
# be limited. If you find you are unable to run this example for any reason, please raise
# and issue on the Stheno.jl repo.
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Stheno, Zygote, Flux



# Stheno and Flux have slightly different conventions regarding their data. In particular
# - Flux assumes that you'll provide a `Matrix` of data, in which each column is an
#   observation.
# - Stheno assumes that you'll provide an `AbstractVector`, where each element is an
#   observation. To handle multi-dimensional inputs we have the `ColVecs` object, that
#   literally just wraps a `Matrix` and tells Stheno to pretend is a vector of vectors. This
#   is helpful to remove some ambiguities that arrise if you don't do this.
#
# This function just bridges the interface between the two packages. It pulls the `Matrix`
# out from a `ColVecs`, passes it through the MLP, and re-wraps it so that Stheno knows
# what to do. We'll need to add the other Flux components to this list.
Broadcast.broadcasted(model::Chain, x::ColVecs) = ColVecs(model(x.X))



#
# The most basic example usage.
#

D_in = 5
D_out = 2

# Specify a simple MLP.
g = Chain(
    Dense(D_in, 10, relu),
    Dense(10, D_out),
)

# Specify a simple GP.
f = GP(eq(), GPC())

# Compose the GP with the MLP.
f_nn = f ∘ g

# Generate some noisy data from it.
x = ColVecs(randn(D_in, 100))
fx = f_nn(x, 0.1)
y = rand(fx)

# Compute its logpdf, because we can.
logpdf(fx, y)



#
# Marginally more complicated example. We're going to use Zygote to compute the gradient
# of the log marginal likelihood of some data w.r.t GP hypers and an MLP. Once you've done
# this, you can learn the parameters etc.
#

# Function to construct a GP. Stheno goes with a re-build your model each time approach at
# the minute. I'm beginning to come around the the Flux / Zygote way of thinking on this
# though, so we need to think about how this could work without screwing up what's already
# there. Presumably I need to do something with the functor interface, but I don't really
# know my way around it at the minute.
function build_model(σ²::Real, mlp::Chain)
    f = GP(eq(), GPC())
    return σ² * (f ∘ mlp)
end

D_in = 5
D_out = 2

# Specify another simple MLP.
g = Chain(
    Dense(D_in, 10, σ),
    Dense(10, D_out),
)

# Generate some data.
σ² = 2.3
f = build_model(σ², g)
x = ColVecs(randn(D_in, 100))
fx = f(x, 0.1)
y = rand(fx)

# Compute gradient of log marginal likelihood w.r.t. our parameters.
dσ², dg = Zygote.gradient(
    function(σ², g)
        f = build_model(σ², g)
        fx = f(x, 0.1)
        return logpdf(fx, y)
    end,
    σ², g,
)

dσ² # gradient w.r.t. process variance
dg  # gradient w.r.t. MLP


#
# Below is the code that you would write if Stheno weren't at all aware of Flux's existence.
#

dσ², dg = Zygote.gradient(
    function(σ², g)

        # Manually transform data
        gx = ColVecs(g(x.X))

        # Construct GP and compute marginal likelihood using transformed data
        f = σ² * GP(eq(), GPC())
        fx = f(gx, 0.1)
        return logpdf(fx, y)
    end,
    σ², g,
)
