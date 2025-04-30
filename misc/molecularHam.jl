## This code is taken from Kyle's ctrlVQEJulia repository##
#= Provide an interface to PySCF calculations.

Input: geometry, or molecule+parameters
Output: h[p,q], h[p,q,r,s], E_HF, E_FCI

=#

import LinearAlgebra: I, Hermitian, norm, eigen

import TensorOperations: @tensor
import PyCall: pyimport

GeometryType = Vector{Pair{String, Tuple{Float64,Float64,Float64}}}

function H2_geometry(r::Real)::GeometryType
    return [
        "H" => (zero(typeof(r)), zero(typeof(r)), zero(typeof(r))),
        "H" => (zero(typeof(r)), zero(typeof(r)),             r  ),
    ]
end

struct Molecule
    geometry::GeometryType      # ATOMIC LABELS PAIRED WITH COORDINATES
    N::Int                      # SIZE OF BASIS SET, OR NUMBER OF *SPATIAL* ORBITALS
    n::Int                      # TOTAL NUMBER OF *SPIN* ORBITALS
    Ne::Int                     # TOTAL NUMBER OF ELECTRONS
    Nα::Int                     # NUMBER OF SPIN-UP   ELECTRONS
    Nβ::Int                     # NUMBER OF SPIN-DOWN ELECTRONS
    E_HF::Float64               # HARTREE-FOCK ENERGY
    E_FCI::Float64              # FULL CONFIGURATION INTERACTION ENERGY
    h0::Float64                 # NUCLEAR REPULSION ENERGY
    h1::Array{Float64,2}        # SINGLE-BODY TENSOR
    h2::Array{Float64,4}        # TWO-BODY TENSOR
    # H = h0 + ∑ h1[p,q] a†[p] a[q] + ∑ h2[p,q,r,s] a†[p] a†[q] a[r] a[s]
end

function Molecule(geometry::GeometryType; skip_FCI=false)
    # CREATE MOLECULE
    mol = pyimport("pyscf.gto").Mole()
    mol.atom = geometry     
    mol.basis = "sto-3g"
    mol.build()
        
    N = mol.nao_nr().tolist() 
    n = 2N
    Nα, Nβ = mol.nelec
    Ne = Nα + Nβ

    # RUN CALCULATIONS
    mf = pyimport("pyscf.scf").RHF(mol).run()
    E_HF = mf.e_tot
    println("E_HF = ", E_HF)

    if !skip_FCI
        fci = pyimport("pyscf.fci").FCI(mf).run()
        E_FCI = fci.e_tot
        println("E_FCI = ", E_FCI)
    end

    # PROCESS INTEGRALS INTO HAMILTONIAN TENSORS
    
    # CALCULATE ATOMIC-ORBITAL INTEGRALS
    T = mol.intor("int1e_kin_sph")      # KINETIC ENERGY
    U = mol.intor("int1e_nuc_sph")      # NUCLEAR REPULSION
    h = T + U                           # SINGLE-BODY INTEGRALS
    v = mol.intor("int2e_sph")          # TWO-BODY INTEGRALS

    # TRANSFORM INTO MOLECULAR-ORBITAL BASIS
    C = mf.mo_coeff                    # AO -> MO TRANSFORMATION MATRIX
    @tensor H[i,j] := C'[i,p] * h[p,q] * C[q,j]
    @tensor V[i,j,k,l] := v[p,q,r,s] * C[p,i] * C[q,j] * C[r,k] * C[s,l]

    # CONSTRUCT TENSORS
    h0 = mol.energy_nuc()               # NUCLEAR REPULSION

    h1 = zeros(Float64, n, n)           # ONE-BODY TENSOR
    for i in 1:N; for j in 1:N
        h1[2i-1,2j-1]     = H[i,j]      # SPIN-UP   TERMS
        h1[2i  ,2j  ] = H[i,j]          # SPIN-DOWN TERMS
    end; end

    h2 = zeros(Float64, n, n, n, n)     # TWO-BODY TENSOR
    for i in 1:N; for j in 1:N; for k in 1:N; for l in 1:N
        h2[2i-1, 2j-1, 2l-1, 2k-1] = V[i,k,j,l] / 2
        h2[2i-1, 2j  , 2l  , 2k-1] = V[i,k,j,l] / 2
        h2[2i  , 2j-1, 2l-1, 2k  ] = V[i,k,j,l] / 2
        h2[2i  , 2j  , 2l  , 2k  ] = V[i,k,j,l] / 2
    end; end; end; end

    return Molecule(geometry, N, n, Ne, Nα, Nβ, E_HF, E_FCI, h0, h1, h2)
end

function kron_concat(ops::Vector{<:AbstractMatrix})
    #= Concatenate a list of matrices into a single Kronecker product.
        This is a more efficient implementation than using the built-in kron function.
    =#
    result = ops[1]
    for i in 2:length(ops)
        result = kron(result, ops[i])
    end
    return result
end

function fermi_a(q,n)
    Iq = [1.0  0.0; 0.0  1.0]       # IDENTITY
    Zq = [1.0  0.0; 0.0 -1.0]       # PARITY
    aq = [0.0  1.0; 0.0  0.0]       # ANNIHILATOR

    ops = [ (p < q) ? Iq : (p > q) ? Zq : aq for p in 1:n]
    return kron_concat(ops)
end

function molecular_hamiltonian(mol::Molecule; m=nothing)
    #= m is number of levels to extend 2-level fermionic operator to.
        Only use this for calculating gradient,
        where we need to apply H to transmon state-vector in the middle of time evolution.
    =#
    
    n = mol.n           # NUMBER OF QUBITS == 2 * NUMBER OF SPATIAL ORBITALS
    N = 2^n             # SIZE OF HILBERT SPACE (NOTE: *NOT* THE SAME AS mol.N)

    H = mol.h0 * Matrix(I,N,N)                              # h0

    for i in 1:n; for j in 1:n                              # h1
        if abs(mol.h1[i,j]) < eps(Float64); continue; end
        H += mol.h1[i,j] * fermi_a(i,n)' * fermi_a(j,n)
    end; end

    for i in 1:n; for j in 1:n; for k in 1:n; for l in 1:n  # h2
        if abs(mol.h2[i,j,k,l]) < eps(Float64); continue; end
        H += mol.h2[i,j,k,l] * fermi_a(i,n)' * fermi_a(j,n)' * fermi_a(k,n) * fermi_a(l,n)
    end; end; end; end

    if m !== nothing
        H = extendoperator(H, n, m)
    end

    return Hermitian(H)
end

function measure_energy(H::Hermitian, ψ::Vector{ComplexF64})
    N = size(H,1)
    # IF NEEDED, PROJECT Ψ ONTO QUBIT SPACE
    if length(ψ) > N
        n = Int(ceil(log2(N)))              # NUMBER OF QUBITS
        ψ = qubitspace(ψ, n)          # PROJECTED STATEVECTOR
        ψ ./= norm(ψ)                       # RE-NORMALIZED
    end
    return real(ψ'*H*ψ)     # ENERGY CALCULATION
end


function qubitspace(
    ψ::Vector{T},                       # `m`-LEVEL STATEVECTOR
    n::Integer;                         # NUMBER OF QUBITS

    # INFERRED VALUES (relatively fast, but pass them in to minimize allocations)
    N = length(ψ),                      # SIZE OF STATEVECTOR
    m = round(Int, N^(1/n)),            # NUMBER OF LEVELS ON EACH QUBIT

    # PRE-ALLOCATIONS (for those that want every last drop of efficiency...)
    ψ2= Vector{T}(undef, 2^n),          # 2-LEVEL STATEVECTOR
    z = Vector{Bool}(undef, n),         # BITSTRING VECTOR
) where T <: Number
    # SELECT ELEMENTS OF ψ WITH ONLY 0, 1 VALUES
    for i2 in eachindex(ψ2)
        digits!(z, i2-1, base=2)                        # FILL z WITH i2's BITSTRING
        i = 1+foldr((a,b) -> muladd(m,b,a), z, init=0)  # PARSE z AS BASE `m` STRING
        ψ2[i2] = ψ[i]
    end

    return ψ2
end


function extendoperator(
    A::AbstractMatrix,      # OPERATOR
    n::Integer,             # NUMBER OF QUBITS
    m::Integer;             # TARGET NUMBER OF LEVELS ON EACH QUBIT

    # INFERRED VALUES (relatively fast, but pass them in to minimize allocations)
    N = m^n,                            # SIZE OF EXTENDED HILBERT SPACE
    N0 = size(A,1),                     # SIZE OF INITIAL HILBERT SPACE
    m0 = round(Int, N0^(1/n)),          # INITIAL NUMBER OF LEVELS ON EACH QUBIT

    # PRE-ALLOCATIONS (for those that want every last drop of efficiency...)
    B = zeros(eltype(A), (N,N)),        # EXTENDED OPERATOR (STARTS WITH ZEROS!)
    imap = nothing,                     # MAP FROM BASE-m0 INDICES TO BASE-m INDICES
)
    if imap === nothing
        z = Vector{Int}(undef, n)       # BITSTRING VECTOR
        imap = Vector{Int}(undef, N0)   # MAP FROM BASE-m0 INDICES TO BASE-m INDICES
        for i in 1:N0
            digits!(z, i-1, base=m0)                    # FILL z WITH i's BASE `m0` STRING
            imap[i] = 1+foldr((a,b)->muladd(m,b,a), z, init=0)# PARSE z AS BASE `m` STRING
        end
    end

    # COPY VALUES OF A INTO B
    for i in 1:N0; for j in 1:N0
        B[imap[i],imap[j]] = A[i,j]
    end; end

    return B
end


function BeH2_geometry(r::Real)::GeometryType
    return [
        "Be" => (zero(typeof(r)), zero(typeof(r)), zero(typeof(r))),
        "H"  => (zero(typeof(r)), zero(typeof(r)),             r  ),
        "H"  => (zero(typeof(r)), zero(typeof(r)),             -r  ),
    ]
end

##example 
mol = Molecule(H2_geometry(0.74))
H = molecular_hamiltonian(mol)
eigvalues, eigvectors = eigen(H)
println("Eignvalues of our static Hamiltonian")
display(eigvalues)
ψ = zeros(ComplexF64, 2^mol.n)
ψ[1] = 0.0
ψ[2] = 1.0

display(measure_energy(H, ψ))
mol = Molecule(H2_geometry(1.5))
H = molecular_hamiltonian(mol)

eigvalues, eigvectors = eigen(H)
println("Eignvalues of our static Hamiltonian")
display(eigvalues)
ψ = zeros(ComplexF64, 2^mol.n)
ψ[1] = 0.0
ψ[2] = 1.0

display(measure_energy(H, ψ))

# mol= Molecule(BeH2_geometry(1.33))
# H = molecular_hamiltonian(mol)
# ψ = zeros(ComplexF64, 2^mol.n)
# display(ψ)
# ψ[1] = 0.0
# ψ[2] = 1.0
# display(measure_energy(H, ψ))
