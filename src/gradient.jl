using SimpleEvolve


function gradientsignal_ODE(ψ0,
                            T,
                            signals,
                            n_sites,
                            drives,
                            Hstatic,
                            cost_ham,
                            n_samples,
                            ∂Ω = Matrix{Float64}(undef, n_samples+1, n_sites))

    eigvals, eigvecs = eigen(Hstatic)
    tmp_σ = zeros(ComplexF64, length(ψ0))
    ψ = copy(ψ0)
    σ = copy(ψ0)
    t_ = range(0,T,length=n_samples+1)
    δt = T/n_samples

    #evolve the sigma state with ODE in forward direction
    parameters = [signals, n_sites, drives, eigvals]
    prob = ODEProblem(dψdt!, σ, (0.0,T), parameters)
    sol = solve(prob, BS3(), abstol=1e-8, reltol=1e-8,save_everystep=false)
    σ .= sol.u[end]
    σ .= mul!(tmp_σ,cost_ham,σ)                     # calculate C|ψ⟩    
    
    # reverse time evolution of the sigma state
    prob_ = ODEProblem(dψdt!, σ, (T,0.0), parameters)
    sol_ = solve(prob_, BS3(), abstol=1e-8, reltol=1e-8,save_everystep=false)
    σ .= sol_.u[end]

    
    #calculating gradient by evolving both \psi and \sigma states
    for i ∈ (1:n_samples+1)
        if i==1
            t_i=0.0
            t_f=δt/2
            Δt=δt/2
        elseif i==n_samples+1
            t_i=T-δt/2
            t_f=T
            Δt=δt/2
        else
            t_i=t_[i]-δt/2
            t_f=t_[i]+δt/2
            Δt=δt
        end
        
        gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,eigvals,t_i,i,Δt)
        parameters = [signals, n_sites, drives, eigvals]
        prob_ψ = ODEProblem(dψdt!, ψ, (t_i, t_f), parameters)
        sol_ψ = solve(prob_ψ, BS3(), abstol=1e-9, reltol=1e-9,save_everystep=false)
        ψ .= sol_ψ.u[end]

        prob_σ = ODEProblem(dψdt!, σ, (t_i, t_f), parameters)
        sol_σ = solve(prob_σ, BS3(), abstol=1e-9, reltol=1e-9,save_everystep=false)
        σ .= sol_σ.u[end]  
    end

    return ∂Ω
end




function gradient_eachtimestep!(∂Ω,
                                ψ,
                                σ,
                                multi_signal,
                                n_sites,
                                drives,
                                eigvals,
                                t,
                                time_index,
                                Δt)
    
    dim= length(ψ)
    dH_dΩ = zeros(ComplexF64, dim, dim)
    device_action = Vector{ComplexF64}(undef, dim)
    for k in 1:n_sites
        dH_dΩ .+= exp(im*frequency(multi_signal.channels[k], t)*t).*drives[k]
        dH_dΩ .+= dH_dΩ'

        # derivative of interaction picture Hamiltonian
        device_action .= exp.((im*t) .* eigvals)                  
        expD = Diagonal(device_action)                       
        lmul!(expD, dH_dΩ); rmul!(dH_dΩ, expD')               
        AΨ = dH_dΩ * ψ

        # calculate gradient ⟨σ|A|ψ⟩
        σAψ = -im*Δt * (σ' * AΨ)                       
        # ⟨σ|A|ψ⟩ + ⟨ψ|A|σ⟩ 
        ∂Ω[time_index,k] = σAψ + σAψ'                          
        dH_dΩ .= zeros(dim,dim) 
    end 
    
end

function gradientsignal_direct_exponentiation(ψ0,
                                        T,
                                        signals,
                                        n_sites,
                                        drives,
                                        Hstatic,
                                        n_trotter_steps,
                                        cost_ham,
                                        n_signals,
                                        ∂Ω = Matrix{Float64}(undef, n_signals+1, n_sites))
           
    eigvals, eigvecs = eigen(Hstatic)
    ψ = copy(ψ0)
    σ = copy(ψ0)
    t_series=range(0,T,n_trotter_steps+1)
    dt= T/n_trotter_steps
    Δt =T/n_signals
    tmp_σ = zeros(ComplexF64, length(ψ0))


    # time evolution with direct exponentiation
    σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvals, dt/2,t_series[1])
    for i in 2:n_trotter_steps
        σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvals, dt,t_series[i])
    end
    σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvals, dt/2,t_series[end])
    σ .= mul!(tmp_σ,cost_ham,σ) 



    #time evolution backward for sigma state
    σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvals, dt/2,t_series[end],true)
    for i in reverse(2:n_trotter_steps)
        σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvals, dt,t_series[i],true)
    end
    σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvals, dt/2,t_series[1],true)
    # 


    #calculating gradient by evolving both \psi and \sigma states
    gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,eigvals,t_series[1],1,Δt/2)
    σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvals, Δt/2,t_series[1])
    ψ .= single_trotter_exponentiation_step(ψ,signals, n_sites, drives, eigvals, Δt/2,t_series[1])
    
    for i in 2:n_signals

        gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,eigvals,t_series[i],i,Δt)
        ψ .= single_trotter_exponentiation_step(ψ,signals, n_sites, drives, eigvals, Δt,t_series[i])
        σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvals, Δt,t_series[i])
    
    end
    σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvals, Δt/2,t_series[end])
    ψ .= single_trotter_exponentiation_step(ψ,signals, n_sites, drives, eigvals, Δt/2,t_series[end])
    return ∂Ω

end
