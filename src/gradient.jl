using SimpleEvolve

"""
gradientsignal_ODE(ψ0, T, signals, n_sites, drives, eigvalues, device_action_independent_t, cost_ham, n_signals, ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites); tol_ode=1e-8)
    Function to compute the gradient of the energy 
    with respect to the amplitude of the signal using ODE
    args:
        ψ0       : Initial state vector
        T        : Total time for evolution
        signals   : Signals to be evolved
        n_sites   : Number of sites in the system
        drives    : External drives applied to the system
        eigvalues : Eigenvalues of the Hamiltonian
        device_action_independent_t: Device action independent time evolution
        cost_ham  : Cost Hamiltonian
        n_signals : Number of signals
        ∂Ω        : Gradient matrix (default is uninitialized)
        tol_ode   : Tolerance for ODE solver (default is 1e-8)
    returns:
        ∂Ω       : Gradient matrix

"""

function gradientsignal_ODE(ψ0,
                            T,
                            signals,
                            n_sites,
                            drives,
                            eigvalues,
                            device_action_independent_t,
                            cost_ham,
                            n_signals,
                            ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites);
                            tol_ode=1e-8) 

    # eigvalues, eigvecs = eigen(Hstatic)
    tmp_σ = zeros(ComplexF64, length(ψ0))
    ψ     = copy(ψ0)
    σ     = copy(ψ0)
    t_    = range(0,T,length=n_signals+1)
    δt    = T/n_signals

    #evolve the sigma state with ODE in forward direction
    parameters = [signals, n_sites, drives,eigvalues]
    prob = ODEProblem(dψdt!, σ, (0.0,T), parameters)
    sol  = solve(prob, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    σ .= sol.u[end]
    σ .= mul!(tmp_σ,cost_ham,σ)                     # calculate C|ψ⟩    
    
    # reverse time evolution of the sigma state
    prob_ = ODEProblem(dψdt!, σ, (T,0.0), parameters)
    sol_  = solve(prob_, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    σ .= sol_.u[end]

    
    #calculating gradient by evolving both \psi and \sigma states
    gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,device_action_independent_t,t_[1],1,t_[1]/δt)
    parameters = [signals, n_sites, drives, eigvalues]
    prob_ψ = ODEProblem(dψdt!, ψ, (0.0,δt/2), parameters)
    sol_ψ  = solve(prob_ψ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    ψ .= sol_ψ.u[end]

    prob_σ = ODEProblem(dψdt!, σ, (0.0, δt/2), parameters)
    sol_σ  = solve(prob_σ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    σ .= sol_σ.u[end]
    for i ∈ (2:n_signals)
        t_i = t_[i]-δt/2
        t_f = t_[i]+δt/2
        gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,device_action_independent_t,t_i,i,t_i/δt)
        parameters = [signals, n_sites, drives, eigvalues]
        prob_ψ = ODEProblem(dψdt!, ψ, (t_i, t_f), parameters)
        sol_ψ = solve(prob_ψ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
        ψ .= sol_ψ.u[end]

        prob_σ = ODEProblem(dψdt!, σ, (t_i, t_f), parameters)
        sol_σ  = solve(prob_σ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
        σ .= sol_σ.u[end]  
    end
    gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,device_action_independent_t,(t_[end]-δt/2),n_signals+1,(t_[end]-δt/2)/δt)
    parameters = [signals, n_sites, drives, eigvalues]
    prob_ψ = ODEProblem(dψdt!, ψ, (T-δt/2, T), parameters)
    sol_ψ  = solve(prob_ψ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    ψ .= sol_ψ.u[end]

    prob_σ = ODEProblem(dψdt!, σ, (T-δt/2, T), parameters)
    sol_σ  = solve(prob_σ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    σ .= sol_σ.u[end]

    return ∂Ω
end

"""
gradient_eachtimestep(δΩ, ψ, σ, multi_signal, n_sites, drives, device_action_independent_t, t, time_index, time_factor)
    Function to compute the gradient of the energy 
    with respect to the amplitude of the signal at each time step
    args:
        δΩ                : Gradient matrix
        ψ                 : Evolved state vector
        σ                 : Evolved state vector
        multi_signal      : Multi-channel signal
        n_sites           : Number of sites in the system
        drives            : External drives applied to the system
        device_action_independent_t: Device action independent time evolution
        t                 : Current time
        time_index        : Index of the current time step
        time_factor       : Time factor for exponentiation on device_action_independent_t
         to get the interaction Hamiltonian at time t
    returns:
        δΩ               : Updated gradient matrix at the current time step

"""


function gradient_eachtimestep!(∂Ω,
                                ψ,
                                σ,
                                multi_signal,
                                n_sites,
                                drives,
                                device_action_independent_t,
                                t,
                                time_index,
                                time_factor)
    
    dim= length(ψ)
    dH_dΩ = zeros(ComplexF64, dim, dim)
    device_action = Vector{ComplexF64}(undef, dim)
    for k in 1:n_sites
        dH_dΩ .+= exp(im*frequency(multi_signal.channels[k], t)*t).*drives[k]
        dH_dΩ .+= dH_dΩ'

        # derivative of interaction picture Hamiltonian
        #device_action_independent_t=exp.((im*τ) .* eigvalues)
        device_action .= (device_action_independent_t).^time_factor
        # device_action .= exp.((im*t) .* eigvalues)                  
        expD = Diagonal(device_action)                       
        lmul!(expD, dH_dΩ); rmul!(dH_dΩ, expD')               
        AΨ = dH_dΩ * ψ
        # calculate gradient ⟨σ|A|ψ⟩
        σAψ = -im * (σ' * AΨ)
        # σAψ = -im * (σ' * AΨ)*    multi_signal.channels[k].δt  

        # ⟨σ|A|ψ⟩ + ⟨ψ|A|σ⟩ 
        ∂Ω[time_index,k] = σAψ + σAψ'                      
        dH_dΩ .= zeros(dim,dim) 
    end 
    
end

"""
gradientsignal_direct_exponentiation(ψ0, T, signals, n_sites, drives, eigvalues, device_action_independent_t, n_trotter_steps, cost_ham, n_signals, ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites))
    Function to compute the gradient of the energy 
    with respect to the amplitude of the signal using direct exponentiation
    args:
        ψ0       : Initial state vector
        T        : Total time for evolution
        signals   : Signals to be evolved
        n_sites   : Number of sites in the system
        drives    : External drives applied to the system
        eigvalues : Eigenvalues of the Hamiltonian
        device_action_independent_t: Device action independent time evolution
        n_trotter_steps: Number of Trotter steps for exponentiation
        cost_ham  : Cost Hamiltonian
        n_signals : Number of signals
        ∂Ω       : Gradient matrix (default is uninitialized)
    returns:
        ∂Ω       : Gradient matrix


"""

function gradientsignal_direct_exponentiation(ψ0,
                                        T,
                                        signals,
                                        n_sites,
                                        drives,
                                        eigvalues,
                                        device_action_independent_t,
                                        n_trotter_steps,
                                        cost_ham,
                                        n_signals,
                                        ∂Ω = Matrix{Float64}(undef, n_signals+1, n_sites))
           
    # eigvalues, eigvecs = eigen(Hstatic)
    ψ = copy(ψ0)
    σ = copy(ψ0)
    t_series=range(0,T,n_trotter_steps+1)
    dt= T/n_trotter_steps
    Δt =T/n_signals
    tmp_σ = zeros(ComplexF64, length(ψ0))


    # time evolution with direct exponentiation
    σ .= single_trotter_exponentiation_step(ψ,signals, n_sites, drives, eigvalues, dt/2,t_series[1])
    for i in 2:n_trotter_steps
        σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues, dt,t_series[i])
    end
    σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues, dt/2,t_series[end])
    σ .= mul!(tmp_σ,cost_ham,σ) 



    #time evolution backward for sigma state
    σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives,eigvalues, dt/2,t_series[end],true)
    for i in reverse(2:n_trotter_steps)
        σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues, dt,t_series[i],true)
    end
    σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues, dt/2,t_series[1],true)
    # 


    #calculating gradient by evolving both \psi and \sigma states
    gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,device_action_independent_t,t_series[1],1,t_series[1]/Δt)
    σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues, Δt/2,t_series[1])
    ψ .= single_trotter_exponentiation_step(ψ,signals, n_sites, drives, eigvalues, Δt/2,t_series[1])
    
    for i in 2:n_signals
        gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,device_action_independent_t,t_series[i],i,t_series[i]/Δt)
        ψ .= single_trotter_exponentiation_step(ψ,signals, n_sites, drives, eigvalues, Δt,t_series[i])
        σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues, Δt,t_series[i])
    
    end
    gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,device_action_independent_t,t_series[end],n_signals+1,(t_series[end]-Δt/2)/Δt)
    σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues, Δt/2,t_series[end])
    ψ .= single_trotter_exponentiation_step(ψ,signals, n_sites, drives, eigvalues, Δt/2,t_series[end])
    return ∂Ω

end


