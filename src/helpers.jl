using LinearAlgebra
"""
a_q(n_levels)

    Returns the bosonic annihilation operator of dimension n_levels for each qubit in the system.
    The operator is represented as a matrix of size n_levels x n_levels.

    Parameters
    ----------
    n_levels : Int
        Number of levels in the  basis.

    Returns
    -------
    a : Matrix{Float64}
        Bosonic annihilation operator in the basis.

"""
# bosonic annhilation operator 
function a_q(n_levels)
    a = zeros((n_levels,n_levels))
    for i ∈ 1:n_levels-1
        a[i,i+1] = √i            
    end
    return a
end

"""
a_fullspace(n_sites,n_levels,eig_basis)

    Returns the bosonic annihilation operator in the full Hilbert space of the system.
    The operator is represented as a matrix of size n_levels^n_sites x n_levels^n_sites.

    Parameters
    ----------
    n_sites : Int
        Number of sites in the system.
    n_levels : Int
        Number of levels in the basis.
    eig_basis : Bool
        If true, the operator is transformed to the eigenbasis.

    Returns
    -------
    a : Vector{Matrix{Float64}}
        Bosonic annihilation operator in the full Hilbert space.

"""

# bosonic annhilation operator in full hilbert space
function a_fullspace(n_sites,n_levels;eig_basis=nothing)
    a_k = a_q(n_levels) 
    a=Vector{Matrix{Float64}}(undef, n_sites)
    for k in 1:n_sites 
        A = ones(1,1)                   
        I = one(a_k)                
        for i ∈ 1:n_sites
             # `a_k` acts on sites `k`, `I` acts on other sites
            if i==k
                A= kron(A, a_k)
            else
                A= kron(A, I)
            end   
        end
        a[k]=A
    end 
    if !(eig_basis === nothing)
        for k in 1:n_sites
            a[k] = eigbasis'*a[k]*eigbasis
        end
        
    end                          
    return a
end


"""
    projector(n_sites::Integer, n_levels::Integer, n_levels0::Integer)

Project a Hilbert space of `n_sites` `n_levels0`-level qubits onto that of `n_sites` `n_levels`-level qubits

Returns an (`n_sites^n_levels`, `n_sites^n_levels`) shaped matrix `Π`.
To perform the projection on a vector, use ψ ← Πψ.
To perform the projection on a matrix, use A ← ΠAΠ'.

"""
function projector(n::Integer, m::Integer, m0::Integer)
    
    if m < m0
        return projector(n, m0, m)'
    end   

    z = Vector{Int}(undef, n)        # Stores base-`m` digit decomposition
    N  = m^n; N0 = m0^n             # Dimensions of larger/smaller spaces
    Id = Matrix{Bool}(I, N, N)      # Identity matrix for larger space
    Π = Matrix{Bool}(undef, N, N0)  # Projector matrix (N×N0)
    j = 1                           # Column index for Π

    for i ∈ 1:N                     # Iterate over columns of larger space
        digits!(z, i-1, base=m)     # Decompose index i-1 into base-`m` digits
        if any(z .>= m0); continue; end  # Skip invalid indices
        Π[:,j] .= Id[:,i]           # Copy valid column from identity
        j += 1                      # Move to next column in Π
    end
    
    return Π
end

function transform!(
    x::Vector{T}, A::AbstractMatrix{<:Number}, _x::Vector{T}
) where T <: Number
    x .= mul!(_x, A, x)
end
function kron_concat(
    ops::AbstractVector{Matrix{T}},
    O_::AbstractVector{Matrix{T}},
) where T <: Number
    O_[1] .= ops[1]
    for q ∈ 2:length(ops)
        kron!(O_[q], O_[q-1], ops[q])
    end
    return O_[end]
end