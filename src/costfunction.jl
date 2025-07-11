"""
costfunction_ode(ψ0, eigvals, signal, n_sites, drives, T, Cost_ham; tol_ode=1e-8)
    cost_function for the system using ODE solver
    args:
        ψ0     : Initial state
        eigvals: Eigenvalues of the Hamiltonian
        signal : Signal to be evolved
        n_sites: Number of sites in the system
        drives : External drives applied to the system # annhilation operators in case of qubits
        T      : Total time for evolution
        Cost_ham: Hamiltonian used for cost function calculation
        tol_ode: Tolerance for ODE solver (default is 1e-8)
    returns:
        cost   : Cost function value
        ψ_ode  : Evolved state

"""

function costfunction_ode(ψ0::Vector{ComplexF64},
                         eigvals::Vector{Float64},
                         signal, 
                         n_sites::Int, 
                         drives,
                         eigvectors::Matrix{ComplexF64}, 
                         T::Float64, 
                         Cost_ham;
                         basis = "eigenbasis",
                         tol_ode=1e-8)

    ψ_ode = evolve_ODE(ψ0, T, signal, n_sites, drives,eigvals,eigvectors;basis=basis,tol_ode=tol_ode)
    return real(ψ_ode'*Cost_ham*ψ_ode),  ψ_ode
end
"""
costfunction_direct_exponentiation(ψ0, 
                                   eigvals,
                                   signal, 
                                   n_sites, 
                                   drives, 
                                   T,  
                                   n_trotter_steps, 
                                   Cost_ham)

    cost_function for the system using direct exponentiation
    args:
        ψ0     : Initial state
        eigvals: Eigenvalues of the Hamiltonian
        signal : Signal to be evolved
        n_sites: Number of sites in the system
        drives : External drives applied to the system # annhilation operators in case of qubits
        T      : Total time for evolution
        n_trotter_steps: Number of Trotter steps for exponentiation
        Cost_ham: Hamiltonian used for cost function calculation
    returns:
        cost   : Cost function value
        ψ_direct: Evolved state


"""
function costfunction_direct_exponentiation(ψ0::Vector{ComplexF64}, 
                            eigvals::Vector{Float64},
                            eigvectors::Matrix{ComplexF64},
                            signal, 
                            n_sites::Int, 
                            drives,
                            Cost_ham, 
                            T::Float64;
                            basis = "eigenbasis",
                            n_trotter_steps=1000)

    ψ_direct = evolve_direct_exponentiation(ψ0, T, signal, n_sites, drives, eigvals,eigvectors;basis=basis, n_trotter_steps=n_trotter_steps)
    return real(ψ_direct'*Cost_ham*ψ_direct), ψ_direct
end


function costfunction_trotter(ψ0::Vector{ComplexF64}, 
                            eigvalues::Vector{Float64},
                            eigvectors::Matrix{ComplexF64},
                            signals, 
                            n_sites::Int,
                            n_levels::Int, 
                            a_q::Matrix{Float64},
                            Cost_ham,
                            T::Float64;
                            basis = "eigenbasis", 
                            n_trotter_steps=1000
                            )
    ψ_trotter= trotter_evolve(ψ0,T,signals,n_sites,n_levels,a_q,eigvalues,eigvectors;basis=basis,n_trotter_steps=n_trotter_steps)
    return real(ψ_trotter'*Cost_ham*ψ_trotter), ψ_trotter
end

"""
costfunction_ode_excited_states(ψ0, eigvals, signal, n_sites, drives, eigvectors, T, Cost_ham; basis="eigenbasis", tol_ode=1e-8)

    cost_function for the system using ODE solver for excited states
    args:
        ψ0     : Initial state
        eigvals: Eigenvalues of the Hamiltonian
        signal : Signal to be evolved
        n_sites: Number of sites in the system
        drives : External drives applied to the system # annhilation operators in case of qubits
        eigvectors: Eigenvectors of the Hamiltonian
        T      : Total time for evolution
        Cost_ham: Hamiltonian used for cost function calculation
        basis  : Basis in which the evolution is performed (default is "eigenbasis")
        tol_ode: Tolerance for ODE solver (default is 1e-8)
    returns:
        cost   : Cost function value
        ψ_ode  : Evolved state

cost_function=sum_i real(ψ_i' * Cost_ham * ψ_i)

"""
function costfunction_ode_ssvqe(
                            Ψ0::Matrix{ComplexF64},
                            eigvals::Vector{Float64},
                            signal,
                            n_sites::Int,
                            drives,
                            eigvectors::Matrix{ComplexF64},
                            T::Float64,
                            Cost_ham;
                            basis = "eigenbasis",
                            tol_ode = 1e-8,
                            weights = nothing,
                            weighted = false
                        )
    n_states = size(Ψ0, 2)
    if weights === nothing && weighted===true
        weights = reverse(collect(1.0:-0.1:1.0-0.1*(n_states-1)))
    end

    Ψ_ode = evolve_ODE_multiple_states(
        Ψ0, T, signal, n_sites, drives, eigvals, eigvectors;
        basis=basis, tol_ode=tol_ode
    )

    energies = [real(ψ' * Cost_ham * ψ) for ψ in eachcol(Ψ_ode)]
    if weighted == true
        cost = sum(energies .* weights[1:n_states])
    else
        cost = sum(energies .* ones(length(energies)))
    end
    return cost, Ψ_ode, energies
end


function costfunction_ode_vqd(
                        ψ0::Vector{ComplexF64},
                        eigvals::Vector{Float64},
                        signal, 
                        n_sites::Int, 
                        drives,
                        eigvectors::Matrix{ComplexF64}, 
                        T::Float64, 
                        Cost_ham,
                        previous_states::Vector{Vector{ComplexF64}},  # List of lower states
                        βs::Vector{Float64};                          # Penalty coefficients
                        basis = "eigenbasis",
                        tol_ode=1e-8
)
    ψ_ode = evolve_ODE(ψ0, T, signal, n_sites, drives, eigvals, eigvectors; basis=basis, tol_ode=tol_ode)
    energy = real(ψ_ode' * Cost_ham * ψ_ode)
    penalty = 0.0
    for (j, ψ_prev) in enumerate(previous_states)
        penalty += βs[j] * abs2(ψ_prev' * ψ_ode)
    end
    return energy + penalty, ψ_ode
end
