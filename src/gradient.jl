using SimpleEvolve


function gradientsignal_ODE(ψ0,
                            T,
                            signal,
                            n_sites,
                            drives,
                            Hstatic,
                            cost_ham)
    eigvals, eigvecs = eigen(Hstatic)
    tmp_σ = zeros(ComplexF64, length(ψ0))
    ψ = copy(ψ0)
    σ = copy(ψ0)
    t_ = range(0,T,length=n_samples+1)
    δt = T/n_samples
    
    #evolve the sigma state with ODE in forward direction
    parameters = [signal, n_sites, drives, eigvals]
    prob = ODEProblem(dψdt!, σ, (0.0,T), parameters)
    sol = solve(prob, BS3(), abstol=1e-10, reltol=1e-11,save_everystep=false)
    σ .= sol.u[end]
    σ .= mul!(tmp_σ,cost_ham,σ)                     # calculate C|ψ⟩    
    
    # reverse time evolution of the sigma state
    prob_ = ODEProblem(dψdt!, σ, (T,0.0), parameters)
    sol_ = solve(prob_, BS3(), abstol=1e-10, reltol=1e-11,save_everystep=false)
    σ .= sol_.u[end]

    
    #calculating gradient by evolving both \psi and \sigma states
    for i ∈ (1:n_samples+1)
        if i==1
            t_i=0.0,t_f=δt/2,Δt=δt/2
        elseif i==n_samples+1
            t_i=T-δt/2,t_f=T,Δt=δt/2
        else
            t_i=t_[i]-δt/2,t_f=t_[i]+δt/2,Δt=δt
        end
        
        dΩ= gradient_timestep()
        parameters = [signal, n_sites, drives, eigvals]
        prob_ψ = ODEProblem(dψdt!, σ, (t_i, t_f), parameters)
        sol_ψ = solve(prob_ψ, BS3(), abstol=1e-10, reltol=1e-11,save_everystep=false)
        ψ .= sol_ψ.u[end]

        prob_σ = ODEProblem(dψdt!, σ, (t_i, t_f), parameters)
        sol_σ = solve(prob_σ, BS3(), abstol=1e-10, reltol=1e-11,save_everystep=false)
        σ .= sol_σ.u[end]  
    end

    return dΩ
end
function gradient_timestep()
    dΩ_dt = zeros(Float64, n_sites)
    return "not implemented yet"
end

function gradientsignal_direct_exponentiation()
    return "not implemented yet"
    
end
