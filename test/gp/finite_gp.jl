using Stheno: FiniteGP, GPC, pw, ConstMean, OuterKernel
using Statistics
using Distributions: MvNormal, PDMat

@testset "finite_gp" begin

    @testset "statistics" begin
        rng, N, N′ = MersenneTwister(123456), 1, 9
        x, x′ = randn(rng, N), randn(rng, N′)
        f = GP(sin, EQ(), GPC())
        fx, fx′ = FiniteGP(f, x, 0), FiniteGP(f, x′, 0)

        @test mean(fx) == map(mean(f), x)
        @test cov(fx) == pw(kernel(f), x)
        @test cov(fx, fx′) == pw(kernel(f), x, x′)
        @test mean.(marginals(fx)) == mean(f(x))
        @test var.(marginals(fx)) == map(kernel(f), x)
        @test std.(marginals(fx)) == sqrt.(map(kernel(f), x))
    end

    @testset "rand (deterministic)" begin
        rng, N, D = MersenneTwister(123456), 10, 2
        X, x = ColsAreObs(randn(rng, D, N)), randn(rng, N)
        fX = FiniteGP(GP(1, EQ(), GPC()), X, 0)
        fx = FiniteGP(GP(1, EQ(), GPC()), x, 0)

        # Check that single-GP samples have the correct dimensions.
        @test length(rand(rng, fX)) == length(X)
        @test size(rand(rng, fX, 10)) == (length(X), 10)

        @test length(rand(rng, fx)) == length(x)
        @test size(rand(rng, fx, 10)) == (length(x), 10)
    end

    @testset "rand (statistical)" begin
        rng, N, D, μ0, S = MersenneTwister(123456), 10, 2, 1, 100_000
        X = ColsAreObs(randn(rng, D, N))
        f = FiniteGP(GP(1, EQ(), GPC()), X, 0)

        # Check mean + covariance estimates approximately converge for single-GP sampling.
        f̂ = rand(rng, f, S)
        @test maximum(abs.(mean(f̂; dims=2) - mean(f))) < 1e-2

        Σ′ = (f̂ .- mean(f)) * (f̂ .- mean(f))' ./ S
        @test mean(abs.(Σ′ - cov(f))) < 1e-2
    end

    @testset "rand (gradients)" begin
        rng, N, S = MersenneTwister(123456), 10, 3
        x = collect(range(-3.0, stop=3.0, length=N))

        # Check that the gradient w.r.t. the samples is correct.
        adjoint_test(
            x->rand(MersenneTwister(123456), FiniteGP(GP(sin, EQ(), GPC()), x, 0), S),
            randn(rng, N, S),
            x,
        )
    end

    @testset "logpdf / elbo" begin
        rng, N, σ, gpc = MersenneTwister(123456), 10, 1e-1, GPC()
        x = collect(range(-3.0, stop=3.0, length=N))
        k_noise = OuterKernel(ConstMean(σ), Noise())
        f_, y_ = GP(1, EQ(), gpc), GP(1, EQ() + k_noise, gpc)
        f, y = FiniteGP(f_, x, 0), FiniteGP(y_, x, 0)
        ŷ = rand(rng, y)

        # Check that logpdf returns the correct type and roughly agrees with Distributions.
        @test logpdf(y, ŷ) isa Real
        @test logpdf(y, ŷ) ≈ logpdf(MvNormal(Vector(mean(f)), cov(y)), ŷ)

        # Check gradient of logpdf at mean is zero for `f`.
        adjoint_test(ŷ->logpdf(f, ŷ), 1, ones(size(ŷ)))
        lp, back = Zygote.forward(ŷ->logpdf(f, ŷ), ones(size(ŷ)))
        @test back(randn(rng))[1] == zeros(size(ŷ)) 

        # Check that gradient of logpdf at mean is zero for `y`.
        adjoint_test(ŷ->logpdf(y, ŷ), 1, ones(size(ŷ)))
        lp, back = Zygote.forward(ŷ->logpdf(y, ŷ), ones(size(ŷ)))
        @test back(randn(rng))[1] == zeros(size(ŷ))

        # Check that gradient w.r.t. inputs is approximately correct for `f` and `y`.
        x, l̄ = randn(rng, N), randn(rng)
        adjoint_test(x->logpdf(FiniteGP(f_, x, 1e-3), ones(size(x))), l̄, collect(x))
        adjoint_test(x->logpdf(FiniteGP(y_, x, 0), ones(size(x))), l̄, collect(x))

        # Check that the gradient w.r.t. the noise is approximately correct for `f`.
        # adjoint_test(σ_->logpdf(FiniteGP(f_, x, )))

        # Ensure that the elbo is close to the logpdf when appropriate.
        @test elbo(f, ŷ, f, σ) isa Real
        @test abs(elbo(f, ŷ, f, σ) - logpdf(y, ŷ)) < 1e-6
    end
end
