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