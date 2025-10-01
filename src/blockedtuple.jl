# This file defines an abstract type AbstractBlockTuple and a concrete type BlockedTuple.
# These types allow to store a Tuple of heterogeneous Tuples with a BlockArrays.jl like
# interface.

using BlockArrays: Block, BlockArrays, BlockIndexRange, BlockRange, blockedrange

using TypeParameterAccessors: type_parameters, unspecify_type_parameters

#
# ==================================  AbstractBlockTuple  ==================================
#
# AbstractBlockTuple imposes BlockLength as first type parameter for easy dispatch
# it makes no assumption on storage type
abstract type AbstractBlockTuple{BlockLength} end

constructorof(type::Type{<:AbstractBlockTuple}) = unspecify_type_parameters(type)
widened_constructorof(type::Type{<:AbstractBlockTuple}) = constructorof(type)

# Like `BlockRange`.
function blockeachindex(bt::AbstractBlockTuple)
    return ntuple(i -> Block(i), blocklength(bt))
end

# Base interface
Base.axes(bt::AbstractBlockTuple) = (blockedrange([blocklengths(bt)...]),)
Base.axes(::AbstractBlockTuple{0}) = (blockedrange(Int[]),)

Base.deepcopy(bt::AbstractBlockTuple) = deepcopy.(bt)

Base.firstindex(::AbstractBlockTuple) = 1

Base.getindex(bt::AbstractBlockTuple, i::Integer) = Tuple(bt)[i]
Base.getindex(bt::AbstractBlockTuple, r::AbstractUnitRange) = Tuple(bt)[r]
Base.getindex(bt::AbstractBlockTuple, b::Block{1}) = blocks(bt)[Int(b)]
function Base.getindex(bt::AbstractBlockTuple, br::BlockRange{1})
    r = Int.(br)
    flat = Tuple(bt)[blockfirsts(bt)[first(r)]:blocklasts(bt)[last(r)]]
    return widened_constructorof(typeof(bt))(flat, blocklengths(bt)[r])
end
function Base.getindex(bt::AbstractBlockTuple, bi::BlockIndexRange{1})
    return bt[Block(bi)][only(bi.indices)]
end
# needed for nested broadcast in Julia < 1.11
Base.getindex(bt::AbstractBlockTuple, ci::CartesianIndex{1}) = bt[only(Tuple(ci))]

Base.iterate(bt::AbstractBlockTuple) = iterate(Tuple(bt))
Base.iterate(bt::AbstractBlockTuple, i::Int) = iterate(Tuple(bt), i)

Base.lastindex(bt::AbstractBlockTuple) = length(bt)

Base.length(bt::AbstractBlockTuple) = sum(blocklengths(bt); init = 0)

function Base.map(f, bt::AbstractBlockTuple)
    BL = blocklengths(bt)
    # use Val to preserve compile time knowledge of BL
    return widened_constructorof(typeof(bt))(map(f, Tuple(bt)), Val(BL))
end

function Base.show(io::IO, bt::AbstractBlockTuple)
    return print(io, nameof(typeof(bt)), blocks(bt))
end
function Base.show(io::IO, ::MIME"text/plain", bt::AbstractBlockTuple)
    println(io, typeof(bt))
    return print(io, blocks(bt))
end

# Broadcast interface
Base.broadcastable(bt::AbstractBlockTuple) = bt
struct AbstractBlockTupleBroadcastStyle{BlockLengths, BT} <: Broadcast.BroadcastStyle end
function Base.BroadcastStyle(T::Type{<:AbstractBlockTuple})
    return AbstractBlockTupleBroadcastStyle{blocklengths(T), unspecify_type_parameters(T)}()
end

# default
combine_types(::Type{<:AbstractBlockTuple}, ::Type{<:AbstractBlockTuple}) = BlockedTuple

# BroadcastStyle(::Style1, ::Style2) is not called when Style1 == Style2
# tuplemortar(((1,), (2,))) .== tuplemortar(((1,), (2,))) = tuplemortar(((true,), (true,)))
# tuplemortar(((1,), (2,))) .== tuplemortar(((1, 2),)) = tuplemortar(((true,), (true,)))
# tuplemortar(((1,), (2,))) .== tuplemortar(((1,), (2,), (3,))) = error DimensionMismatch
function Base.BroadcastStyle(
        s1::AbstractBlockTupleBroadcastStyle, s2::AbstractBlockTupleBroadcastStyle
    )
    blocklengths1 = type_parameters(s1, 1)
    blocklengths2 = type_parameters(s2, 1)
    sum(blocklengths1; init = 0) != sum(blocklengths2; init = 0) &&
        throw(DimensionMismatch("blocked tuples could not be broadcast to a common size"))
    new_blocklasts = static_mergesort(cumsum(blocklengths1), cumsum(blocklengths2))
    new_blocklengths = (
        first(new_blocklasts), Base.tail(new_blocklasts) .- Base.front(new_blocklasts)...,
    )
    BT = combine_types(type_parameters(s1, 2), type_parameters(s2, 2))
    return AbstractBlockTupleBroadcastStyle{new_blocklengths, BT}()
end

static_mergesort(::Tuple{}, ::Tuple{}) = ()
static_mergesort(a::Tuple, ::Tuple{}) = a
static_mergesort(::Tuple{}, b::Tuple) = b
function static_mergesort(a::Tuple, b::Tuple)
    if first(a) == first(b)
        return (first(a), static_mergesort(Base.tail(a), Base.tail(b))...)
    end
    if first(a) < first(b)
        return (first(a), static_mergesort(Base.tail(a), b)...)
    end
    return (first(b), static_mergesort(a, Base.tail(b))...)
end

# tuplemortar(((1,), (2,))) .== (1, 2) = (true, true)
function Base.BroadcastStyle(
        s::AbstractBlockTupleBroadcastStyle, ::Base.Broadcast.Style{Tuple}
    )
    return s
end

# tuplemortar(((1,), (2,))) .== 1 = (true, false)
function Base.BroadcastStyle(
        ::Base.Broadcast.DefaultArrayStyle{0}, s::AbstractBlockTupleBroadcastStyle
    )
    return s
end

# tuplemortar(((1,), (2,))) .== [1, 1] = BlockVector([true, false], [1, 1])
function Base.BroadcastStyle(
        a::Base.Broadcast.AbstractArrayStyle, ::AbstractBlockTupleBroadcastStyle
    )
    return a
end

function Base.copy(
        bc::Broadcast.Broadcasted{AbstractBlockTupleBroadcastStyle{BlockLengths, BT}}
    ) where {BlockLengths, BT}
    return widened_constructorof(BT)(bc.f.((Tuple.(bc.args))...), Val(BlockLengths))
end

Base.ndims(::Type{<:AbstractBlockTuple}) = 1  # needed in nested broadcast

# BlockArrays interface
BlockArrays.blockfirsts(::AbstractBlockTuple{0}) = ()
function BlockArrays.blockfirsts(bt::AbstractBlockTuple)
    return (0, cumsum(Base.front(blocklengths(bt)))...) .+ 1
end

function BlockArrays.blocklasts(bt::AbstractBlockTuple)
    return cumsum(blocklengths(bt))
end

BlockArrays.blocklength(::AbstractBlockTuple{BlockLength}) where {BlockLength} = BlockLength

BlockArrays.blocklengths(bt::AbstractBlockTuple) = blocklengths(typeof(bt))

function BlockArrays.blocks(bt::AbstractBlockTuple)
    bf = blockfirsts(bt)
    bl = blocklasts(bt)
    return ntuple(i -> Tuple(bt)[bf[i]:bl[i]], blocklength(bt))
end

# =====================================  BlockedTuple  =====================================
#
struct BlockedTuple{BlockLength, BlockLengths, Flat} <: AbstractBlockTuple{BlockLength}
    flat::Flat

    function BlockedTuple{BlockLength, BlockLengths}(
            flat::Tuple
        ) where {BlockLength, BlockLengths}
        length(BlockLengths) != BlockLength && throw(DimensionMismatch("Invalid blocklength"))
        length(flat) != sum(BlockLengths; init = 0) &&
            throw(DimensionMismatch("Invalid total length"))
        any(BlockLengths .< 0) && throw(DimensionMismatch("Invalid block length"))
        return new{BlockLength, BlockLengths, typeof(flat)}(flat)
    end
end

# TensorAlgebra Interface
function tuplemortar(tt::Tuple{Vararg{Tuple}})
    return BlockedTuple{length(tt), length.(tt)}(flatten_tuples(tt))
end
function BlockedTuple(flat::Tuple, BlockLengths::Tuple{Vararg{Int}})
    return BlockedTuple{length(BlockLengths), BlockLengths}(flat)
end
function BlockedTuple(flat::Tuple, ::Val{BlockLengths}) where {BlockLengths}
    # use Val to preserve compile time knowledge of BL
    return BlockedTuple{length(BlockLengths), BlockLengths}(flat)
end
function BlockedTuple(bt::AbstractBlockTuple)
    bl = blocklengths(bt)
    return BlockedTuple{length(bl), bl}(Tuple(bt))
end

# Base interface
Base.Tuple(bt::BlockedTuple) = bt.flat

# BlockArrays interface
function BlockArrays.blocklengths(
        ::Type{<:BlockedTuple{<:Any, BlockLengths}}
    ) where {BlockLengths}
    return BlockLengths
end
