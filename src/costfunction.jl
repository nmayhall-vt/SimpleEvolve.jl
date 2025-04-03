

function costfunction_ode(ψ0, Hstatic,signal, n_sites, drives, T, Cost_ham)
    ψ_ode = evolve_ODE(ψ0, T, signal, n_sites, drives, Hstatic)
    return real(ψ_ode'*Cost_ham*ψ_ode),  ψ_ode
end

function costfunction_direct_exponentiation(ψ0, Hstatic,signal, n_sites, drives, T, δt, n_samples, Cost_ham)
    ψ_direct = evolve_direct_exponentiation(ψ0, T, signal, n_sites, drives, Hstatic, δt, n_samples)
    return real(ψ_direct'*Cost_ham*ψ_direct), ψ_direct
end
