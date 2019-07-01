using Turing, Stheno, BenchmarkTools
using Stheno: GPC, TuringGPC

x = collect(range(-5.0, 5.0; length=2500));
Turing.@model turing_foo(y) = begin
    f = GP(0, eq(), Stheno.TuringGPC())
    y ~ f(x, 0.1)
    return y
end

x1, x2 = collect(range(-5.0, 5.0; length=100)), collect(range(-4.5, 5.5; length=100))
Turing.@model turing_foo_two_procs(y1, y2) = begin
    f = GP(0, eq(), Stheno.TuringGPC())
    y1 ~ f(x1, 0.1)
    y2 ~ f(x2, 0.1)
end

y = turing_foo()();

stheno_foo() = GP(0, eq(), GPC())

@benchmark turing_foo()()
@benchmark rand(stheno_foo()($x, 0.1))

@testset "turing.jl" begin
    @testset "logpdf - single proc" begin
        vi = Turing.runmodel!(turing_foo(), Turing.VarInfo(), Turing.SampleFromPrior())
        y = Float64.(vi.metadata.vals)
        @test logpdf(stheno_foo()(x, 0.1), y) == vi.logp
    end
    @testset "logpdf - two procs" begin
        vi = Turing.runmodel!(turing_foo_two_procs(), Turing.VarInfo(), Turing.SampleFromPrior())
        ys = Float64.(vi.metadata.vals)
        y1, y2 = ys[1:100], ys[101:200]
        f = stheno_foo()
        @test logpdf(f(x1, 0.1) ← y1, f(x2, 0.1) ← y2)
    end

end
