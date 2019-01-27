"""
    (f::GP)(x::AbstractVector)

Construct a `FiniteGP` representing the projection of `f` at `x`.
"""
(f::GP)(x::AbstractVector, σ²::Union{Real, AbstractVector{<:Real}}) = FiniteGP(f, x, σ²)
(f::GP)(x::AbstractVector) = f(x, 0)

# """
#     (f_q::BlockGP)(X::BlockData)

# Index the `c`th component `AbstractGP` of `f_q` at `X[c]`.

#     (f_q::BlockGP)(X::AbstractVector)

# Index each `c`th component `AbstractGP` of `f_q` at `X`.
# """
# (f_q::BlockGP)(X::BlockData) = BlockGP(map((f, x)->f(x), f_q.fs, blocks(X)))
# (f_q::BlockGP)(X::AbstractVector) = BlockGP([f(X) for f in f_q.fs])
