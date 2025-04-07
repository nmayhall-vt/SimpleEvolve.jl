struct QubitCoupling
    qubit_1::Int
    qubit_2::Int
    
    # Inner constructor to enforce ordering
    function QubitCoupling(q1::Int, q2::Int)
        if q1 > q2
            return new(q2, q1)
        else
            return new(q1, q2)
        end
    end
end


struct Transmon 
    n_sites::Int
    freq::Vector{Float64}
    anharmonicities::Vector{Float64}
    coupling_map::Dict{QubitCoupling,Float64}  # Fixed typo: QubitCouple → QubitCoupling
    
    function Transmon(
        freq::Vector{Float64},
        anharmonicities::Vector{Float64},
        coupling_map::Dict{QubitCoupling,Float64}=Dict{QubitCoupling,Float64}(),
        n_sites::Integer=length(freq)
    )
        # Truncate arrays and validate couplings
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
        H -= device.anharmonicities[q]/2 * (aq'^2 * aq^2)  # -δ/2 a_q^† a_q^† a_q a_qterm
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
