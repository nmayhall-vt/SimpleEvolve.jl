"""
    QubitCoupling(q1::Int, q2::Int)

A struct of a pairwise coupling between two qubits.

# Arguments
- `q1::Int`: Index of the first qubit.
- `q2::Int`: Index of the second qubit.

# Returns
- A `QubitCoupling` instance with the qubit indices ordered such that `qubit_1 ≤ qubit_2`.
"""
struct QubitCoupling
    qubit_1::Int
    qubit_2::Int
    function QubitCoupling(q1::Int, q2::Int)
        if q1 > q2
            return new(q2, q1)
        else
            return new(q1, q2)
        end
    end
end


"""
    Transmon(freq::Vector{Float64}, anharmonicities::Vector{Float64}, 
             coupling_map::Dict{QubitCoupling, Float64}=Dict(), 
             n_sites::Integer=length(freq))

A data structure representing a system of `n_sites` transmon qubits with specified frequencies, anharmonicities, and pairwise couplings.

# Arguments
- `freq::Vector{Float64}`: A vector specifying the bare transition frequencies (in GHz or appropriate units) of the transmon qubits. Must have length ≥ `n_sites`.
- `anharmonicities::Vector{Float64}`: A vector specifying the anharmonicity (typically negative) for each transmon qubit. Must have length ≥ `n_sites`.
- `coupling_map::Dict{QubitCoupling, Float64}` *(optional)*: A dictionary mapping qubit pairs (as `QubitCoupling`) to their coupling strength (in GHz or appropriate units). Only couplings between valid qubit indices (`1:n_sites`) are retained.
- `n_sites::Integer` *(optional)*: Number of qubits to model. Defaults to the length of `freq`.

# Returns
- A `Transmon` instance containing:
  - `n_sites`: Number of active transmon qubits.
  - `freq`: Vector of truncated frequencies for active sites.
  - `anharmonicities`: Vector of truncated anharmonicities for active sites.
  - `coupling_map`: Dictionary of valid pairwise couplings.
"""

struct Transmon 
    n_sites::Int
    freq::Vector{Float64}
    anharmonicities::Vector{Float64}
    coupling_map::Dict{QubitCoupling,Float64}  
    
    function Transmon(
        freq::Vector{Float64},
        anharmonicities::Vector{Float64},
        coupling_map::Dict{QubitCoupling,Float64}=Dict{QubitCoupling,Float64}(),
        n_sites::Integer=length(freq)
    )
        # Truncate arrays and filter couplings
        freq = freq[1:n_sites]
        anharmonicities = anharmonicities[1:n_sites]
        
        # Filter couplings to valid qubit indices
        valid_couplings = filter(pair -> 
            1 <= pair.first.qubit_1 <= n_sites &&
            1 <= pair.first.qubit_2 <= n_sites,
            coupling_map
        )
        
        new(n_sites, freq, anharmonicities, valid_couplings)
    end
end

"""
    choose_qubits(slice::AbstractVector{<:Integer}, device::Transmon) -> Transmon

Constructs a new `Transmon` subsystem by selecting a subset of qubits from an existing device.

# Arguments
- `slice::AbstractVector{<:Integer}`: A vector of qubit indices (1-based) specifying the subset of qubits to include in the new device. Must be within `1:device.n_sites`.
- `device::Transmon`: The original `Transmon` device from which a sub-device will be extracted.

# Returns
- A new `Transmon` object consisting only of the selected qubits, with:
  - Frequencies and anharmonicities truncated to the selected indices.
  - A filtered and reindexed `coupling_map` containing only the valid interactions between qubits in `slice`.
"""

function choose_qubits(slice::AbstractVector{<:Integer}, device::Transmon)
    # Validate slice indices
    @assert all(1 .<= slice .<= device.n_sites) "Invalid qubit indices in slice"
    
    permutation_vector = zeros(Int, device.n_sites)
    for k in eachindex(slice)
        permutation_vector[slice[k]] = k
    end

    # Create new coupling map with permuted indices
    gmap = Dict{QubitCoupling,Float64}()
    for (pair, g) in device.coupling_map
        new_q1 = permutation_vector[pair.qubit_1]
        new_q2 = permutation_vector[pair.qubit_2]
        if new_q1 != 0 && new_q2 != 0
            new_pair = QubitCoupling(new_q1, new_q2)
            gmap[new_pair] = g
        end
    end

    return Transmon(
        device.freq[slice],
        device.anharmonicities[slice],
        gmap,
        length(slice)
    )
end

"""
    static_hamiltonian(device::Transmon, n_levels::Int=2)

Constructs the transmon Hamiltonian
    ``∑_q ω_q a_q^† a_q
    - ∑_q \\frac{δ_q}{2} a_q^† a_q^† a_q a_q
    + ∑_{⟨pq⟩} g_{pq} (a_p^† a_q + a_q^† a_p)``.

"""

function static_hamiltonian(device::Transmon, n_levels::Integer=2)
    n = device.n_sites  # Number of transmon qubits
    N = n_levels^n  # Total Hilbert space dimension

    # Get annihilation operators for all qubits in full Hilbert space
    a_ = a_fullspace(n, n_levels)

    H = zeros(ComplexF64, N, N)

    # Resonance and anharmonic terms
    for q in 1:n
        aq = a_[q]
        H += device.freq[q] * (aq' * aq)  # ω_q a†a term
        H -= device.anharmonicities[q]/2 * (aq'^2 * aq^2)  # -δ/2 a_q^† a_q^† a_q a_q term
    end

    # Coupling terms (∑ g_{pq}(a_p†a_q + a_q†a_p))
    for (pair, g) in device.coupling_map
        q1 = pair.qubit_1
        q2 = pair.qubit_2
        term = g * (a_[q1]' * a_[q2])
        H += term + term'  # Add both a_p†a_q and its conjugate
    end

    return Hermitian(H)
end
