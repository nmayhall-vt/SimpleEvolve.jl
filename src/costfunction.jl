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

function costfunction_ode(ψ0,
                         eigvals,
                         signal, 
                         n_sites, 
                         drives,
                         eigvectors, 
                         T, 
                         Cost_ham;
                         tol_ode=1e-8)
    tmp_ψ = zeros(ComplexF64, length(ψ0))
    ψ_ode = evolve_ODE(ψ0, T, signal, n_sites, drives,eigvals,tol_ode=tol_ode)
    transform!(ψ_ode, eigvectors, tmp_ψ)           # transform the state to the eigenspace
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
function costfunction_direct_exponentiation(ψ0, 
                                            eigvals,
                                            signal, 
                                            n_sites, 
                                            drives, 
                                            T,  
                                            n_trotter_steps, 
                                            Cost_ham)

    ψ_direct = evolve_direct_exponentiation(ψ0, T, signal, n_sites, drives, eigvals, n_trotter_steps)
    tmp_ψ = zeros(ComplexF64, length(ψ0))
    transform!(ψ_direct, eigvectors, tmp_ψ)           # transform the state to the eigenspace
    return real(ψ_direct'*Cost_ham*ψ_direct), ψ_direct
end
