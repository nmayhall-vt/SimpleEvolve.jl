function costfunction_ode(ψ0, Hstatic,signal, n_sites, drives, T)
    ψ_ode = evolve_ODE(ψ0, T, signal, n_sites, drives, Hstatic)
    return real(ψ_ode'*C*ψ_ode),  ψ_ode
end

function costfunction_direct_exponentiation(ψ0, Hstatic,signal, n_sites, drives, T, δt, n_samples)
    ψ_direct = evolve_direct_exponentiation(ψ0, T, signal, n_sites, drives, Hstatic, δt, n_samples)
    return real(ψ_direct'*C*ψ_direct), ψ_direct
end

function costfunction_trotter(ψ0, Hstatic,signal, n_sites, n_levels, drive_q_dbasis, T, δt, n_samples)
    ψ_trotter = evolve_trotter(ψ0, T, signal, n_sites, n_levels, drive_q_dbasis, Hstatic, δt, n_samples)
    return real(ψ_trotter'*C*ψ_trotter), ψ_trotter
end