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
                            eigvectors,
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
    tmp_σ = zeros(ComplexF64, length(ψ0))
    tmp_ψ = zeros(ComplexF64, length(ψ0))
    repeated_device_action = eigvectors*Diagonal(exp.((-im*δt ) * eigvalues)) *eigvectors'

    #evolve the sigma state with ODE in forward direction
    parameters = [signals, n_sites, drives,eigvalues,false]
    prob = ODEProblem(dψdt!, σ, (0.0,T), parameters)
    sol  = solve(prob, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    σ .= sol.u[end]

    transform!(σ, eigvectors',tmp_σ)           # transform the state to the device space
    σ .*=exp.((im*T)*eigvalues)                # rotate phases for final exp(iHDT)
    transform!(σ, eigvectors, tmp_σ)           # transform the state to the eigenspace

    σ .= mul!(tmp_σ,cost_ham,σ)                # calculate C|ψ⟩    
    
    # # reverse time evolution of the sigma state
    transform!(σ,eigvectors',tmp_σ)            # transform the state to the device space
    σ .*=exp.((im*T)*eigvalues)                # rotate phases for final exp(iHDT)
    transform!(σ, eigvectors, tmp_σ)           # transform the state to the eigenspace

    parameters = [signals, n_sites, drives,eigvalues,true] #should it be true? 
    # or the time already taking care of it in solve function, I could not find any solid proof/evidence of it
    prob_ = ODEProblem(dψdt!, σ, (T,0.0), parameters)
    sol_  = solve(prob_, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    σ .= sol_.u[end]

    
    
    #calculating gradient by evolving both ψ and σ states
    gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,eigvalues,eigvectors,t_[1],1)
    
    # fixed_dt=δt/100
    # parameters = [signals, n_sites, drives, eigvalues,false,fixed_dt] # dψ_dt_grad!
    parameters = [signals, n_sites, drives, eigvalues,false]
    prob_ψ = ODEProblem(dψdt!, ψ, (0.0,δt/2), parameters)
    # sol_ψ  = solve(prob_ψ,dt=fixed_dt,adaptive=false, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    sol_ψ  = solve(prob_ψ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    ψ .= sol_ψ.u[end]


    prob_σ = ODEProblem(dψdt!, σ, (0.0, δt/2), parameters)
    # sol_σ  = solve(prob_σ, dt=fixed_dt,adaptive=false,abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    sol_σ  = solve(prob_σ,abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    σ .= sol_σ.u[end]

    #transform!(σ, V, tmpV)=> σ=mul!(tmpV, V, σ) 
    # transform!(σ, repeated_device_action', tmp_σ)
    # transform!(ψ, repeated_device_action', tmp_ψ)
    
    transform!(σ, eigvectors, tmp_σ )           # transform the state to the eigenspace  #rotating the state to the eigenspace
    transform!(ψ, eigvectors, tmp_ψ)           # transform the state to the eigenspace  #rotating the state to the eigenspace


    for i ∈ (2:n_signals)
        t_i = t_[i]-δt/2
        t_f = t_[i]+δt/2
        gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,eigvalues,eigvectors,t_i,i)
        
        # fixed_dt=δt/100
        # parameters = [signals, n_sites, drives, eigvalues,false,fixed_dt] # dψ_dt_grad!
        parameters = [signals, n_sites, drives, eigvalues,false]
        prob_ψ = ODEProblem(dψdt!, ψ, (t_i, t_f), parameters)
        # sol_ψ = solve(prob_ψ, dt=fixed_dt,adaptive=false,abstol=tol_ode, reltol=tol_ode,save_everystep=false)
        sol_ψ = solve(prob_ψ,abstol=tol_ode, reltol=tol_ode,save_everystep=false)
        ψ .= sol_ψ.u[end]

        prob_σ = ODEProblem(dψdt!, σ, (t_i, t_f), parameters)
        sol_σ  = solve(prob_σ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
        # sol_σ  = solve(prob_σ, dt=fixed_dt,adaptive=false,abstol=tol_ode, reltol=tol_ode,save_everystep=false)
        σ .= sol_σ.u[end]
        
        transform!(σ, eigvectors, tmp_σ)           # transform the state to the eigenspace  #rotating the state to the eigenspace
        transform!(ψ, eigvectors, tmp_ψ)           # transform the state to the eigenspace  #rotating the state to the eigenspace

        
        # transform!(σ, repeated_device_action', tmp_σ)
        # transform!(ψ, repeated_device_action', tmp_ψ)
    end

    gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,eigvalues,eigvectors,(t_[end]-δt/2),n_signals+1)
    
    # fixed_dt=δt/100
    # parameters = [signals, n_sites, drives, eigvalues,false,fixed_dt] # dψ_dt_grad!
    parameters = [signals, n_sites, drives, eigvalues,false]
    prob_ψ = ODEProblem(dψdt!, ψ, (T-δt/2, T), parameters)
    sol_ψ  = solve(prob_ψ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    # sol_ψ  = solve(prob_ψ, dt=fixed_dt,adaptive=false,abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    ψ .= sol_ψ.u[end]

    prob_σ = ODEProblem(dψdt!, σ, (T-δt/2, T), parameters)
    sol_σ  = solve(prob_σ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    # sol_σ  = solve(prob_σ, dt=fixed_dt,adaptive=false,abstol=tol_ode, reltol=tol_ode,save_everystep=false)
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
                                eigvalues,
                                eigvectors,
                                t,
                                time_index)
    
    dim= length(ψ)
    dH_dΩ = zeros(ComplexF64, dim, dim)
    device_action = Vector{ComplexF64}(undef, dim)
    for k in 1:n_sites
        dH_dΩ .+= exp(im*frequency(multi_signal.channels[k], t)*t).*drives[k]
        dH_dΩ .+= dH_dΩ'

        # derivative of interaction picture Hamiltonian
        ## dt=multi_signal.channels[k].δt  
        # device_action .= exp.((im*t) .* eigvalues) 
        ## device_action .= exp.((-im*dt) .* eigvalues)                 
        # expD =Diagonal(device_action)                   
        # lmul!(expD, dH_dΩ); rmul!(dH_dΩ, expD')  
        ## dH_dΩ = eigvectors' * dH_dΩ * eigvectors
        
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
                                        eigvectors,
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
    tmp_ψ = zeros(ComplexF64, length(ψ0))


    # time evolution with direct exponentiation
    σ .= single_trotter_exponentiation_step(ψ,signals, n_sites, drives, eigvalues, dt/2,t_series[1])
    transform!(σ, eigvectors, tmp_σ)           # transform the state to the eigenspace  #rotating the state to the eigenspace
    for i in 2:n_trotter_steps
        σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues, dt,t_series[i])
        transform!(σ, eigvectors, tmp_σ)           # transform the state to the eigenspace  #rotating the state to the eigenspace
    end
    σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues, dt/2,t_series[end])
    transform!(σ, eigvectors, tmp_σ)           # transform the state to the eigenspace  #rotating the state to the eigenspace
    
    transform!(σ, eigvectors',tmp_σ)           # transform the state to the device space
    σ .*=exp.((im*T)*eigvalues)               # rotate phases for final exp(iHDT)
    transform!(σ, eigvectors, tmp_σ)           # transform the state to the eigenspace

    σ .= mul!(tmp_σ,cost_ham,σ)               # calculate C|ψ⟩    
    
    # # reverse time evolution of the sigma state
    transform!(σ,eigvectors',tmp_σ)            # transform the state to the device space
    σ .*=exp.((im*T)*eigvalues)               # rotate phases for final exp(iHDT)
    transform!(σ, eigvectors, tmp_σ)           # transform the state to the eigenspace



    #time evolution backward for sigma state
    σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives,eigvalues, dt/2,t_series[end],true)
    transform!(σ, eigvectors', tmp_σ)           # transform the state to the eigenspace  #rotating the state to the eigenspace
    for i in reverse(2:n_trotter_steps)
        σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues, dt,t_series[i],true)
        transform!(σ, eigvectors', tmp_σ)           # transform the state to the eigenspace  #rotating the state to the eigenspace

    end
    σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues, dt/2,t_series[1],true)
    transform!(σ, eigvectors', tmp_σ)           # transform the state to the eigenspace  #rotating the state to the eigenspace

    #calculating gradient by evolving both \psi and \sigma states
    gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,eigvalues,eigvectors,t_series[1],1)
    σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues, Δt/2,t_series[1])
    ψ .= single_trotter_exponentiation_step(ψ,signals, n_sites, drives, eigvalues, Δt/2,t_series[1])
    transform!(σ, eigvectors, tmp_σ)           # transform the state to the eigenspace  #rotating the state to the eigenspace
    transform!(ψ, eigvectors, tmp_ψ)           # transform the state to the eigenspace  #rotating the state to the eigenspace
    for i in 2:n_signals
        gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,eigvalues,eigvectors,t_series[i],i)
        ψ .= single_trotter_exponentiation_step(ψ,signals, n_sites, drives, eigvalues, Δt,t_series[i])
        σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues, Δt,t_series[i])
        transform!(σ, eigvectors, tmp_σ)           # transform the state to the eigenspace  #rotating the state to the eigenspace
        transform!(ψ, eigvectors, tmp_ψ)           # transform the state to the eigenspace  #rotating the state to the eigenspace
    end
    gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,eigvalues,eigvectors,t_series[end],n_signals+1)
    σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues, Δt/2,t_series[end])
    ψ .= single_trotter_exponentiation_step(ψ,signals, n_sites, drives, eigvalues, Δt/2,t_series[end])
    return ∂Ω

end


function gradientsignal_ODE_rotate(ψ0,
                            T,
                            signals,
                            n_sites,
                            n_levels,
                            drives,
                            a_q,
                            eigvalues,
                            eigvectors,
                            cost_ham,
                            n_signals,
                            ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites);
                            tol_ode=1e-8) 

    # eigvalues, eigvecs = eigen(Hstatic)
    tmp_σ = zeros(ComplexF64, length(ψ0))
    tmp_ψ = zeros(ComplexF64, length(ψ0))
    ψ     = copy(ψ0)
    σ     = copy(ψ0)
    t_    = range(0,T,length=n_signals+1)
    δt    = T/n_signals
    N     = length(ψ0)
    V     = eigvectors*Diagonal(exp.((-im*δt ) * eigvalues)) *eigvectors'
    tmpV = Vector{ComplexF64}(undef, N) 
    tmpM_ = [Matrix{ComplexF64}(undef, n_levels,n_levels) for q ∈ 1:n_sites]  
    tmpK_ = [Matrix{ComplexF64}(undef,  n_levels^q,  n_levels^q) for q ∈ 1:n_sites]

    
    #evolve the sigma state with ODE in forward direction
    # parameters = [signals, n_sites, drives,eigvalues,false]
    # prob = ODEProblem(dψdt!, σ, (0.0,T), parameters)
    # sol  = solve(prob, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    # σ .= sol.u[end]

    σ .= _step(σ, t_[1], δt/2, signals, n_sites, a_q, tmpV, tmpM_, tmpK_)
    transform!(σ, V, tmpV)
    for i ∈ (2:n_signals)
        t_i = t_[i]-δt/2
        σ .= _step(σ, t_i, δt, signals, n_sites, a_q, tmpV, tmpM_, tmpK_)
        transform!(σ, V, tmpV)                #transform!(σ, V, tmpV)=> σ=mul!(tmpV, V, σ) 
    end
    σ .= _step(σ, t_[end], δt/2, signals, n_sites, a_q, tmpV, tmpM_, tmpK_)
   

    transform!(σ, eigvectors',tmpV)           # transform the state to the device space
    σ .*=exp.((im*T)*eigvalues)               # rotate phases for final exp(iHDT)
    transform!(σ, eigvectors, tmpV)           # transform the state to the eigenspace

    σ .= mul!(tmp_σ,cost_ham,σ)               # calculate C|ψ⟩    
    
    # # reverse time evolution of the sigma state
    transform!(σ,eigvectors',tmpV)            # transform the state to the device space
    σ .*=exp.((im*T)*eigvalues)               # rotate phases for final exp(iHDT)
    transform!(σ, eigvectors, tmpV)           # transform the state to the eigenspace
    
    # parameters = [signals, n_sites, drives,eigvalues,true]
    # prob_ = ODEProblem(dψdt!, σ, (T,0.0), parameters)
    # sol_  = solve(prob_, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    # σ .= sol_.u[end]

    σ .= _step(σ, t_[end], δt/2, signals, n_sites, a_q, tmpV, tmpM_, tmpK_,true)
    transform!(σ, V', tmpV)
    for i ∈ reverse(2:n_signals)
        t_i = t_[i]-δt/2
        σ .= _step(σ, t_i, δt, signals, n_sites, a_q, tmpV, tmpM_, tmpK_,true)
        transform!(σ, V', tmpV)
    end
    σ .= _step(σ, t_[1], δt/2, signals, n_sites, a_q, tmpV, tmpM_, tmpK_,true)
   
    
    #calculating gradient by evolving both \psi and \sigma states
    gradient_eachstep!(∂Ω, 1, σ, ψ, t_[1],δt/2 , signals,
                                n_sites, a_q,tmpV,tmpM_,tmpK_)

    parameters = [signals, n_sites, drives, eigvalues,false,δt/100]
    prob_ψ = ODEProblem(dψdt_grad!, ψ, (0.0,δt/2), parameters)
    sol_ψ  = solve(prob_ψ, dt=δt/100,adaptive=false,abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    ψ .= sol_ψ.u[end]
    prob_σ = ODEProblem(dψdt_grad!, σ, (0.0, δt/2), parameters)
    sol_σ  = solve(prob_σ,dt=δt/100,adaptive=false, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    σ .= sol_σ.u[end]
    transform!(σ, V, tmpV)
    transform!(ψ, V, tmpV)



    # parameters = [signals, n_sites, drives, eigvalues,false]
    # prob_ψ = ODEProblem(dψdt, ψ, (0.0,δt/2), parameters)
    # sol_ψ  = solve(prob_ψ,abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    # ψ .= sol_ψ.u[end]
    # prob_σ = ODEProblem(dψdt, σ, (0.0, δt/2), parameters)
    # sol_σ  = solve(prob_σ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    # σ .= sol_σ.u[end]
    # transform!(σ, eigvectors, tmp_σ )           # transform the state to the eigenspace  #rotating the state to the eigenspace
    # transform!(ψ, eigvectors, tmp_ψ)           # transform the state to the eigenspace  #rotating the state to the eigenspace



    # σ .= _step(σ, t_[1], δt/2, signals, n_sites, a_q, tmpV, tmpM_, tmpK_)
    # ψ .= _step(ψ, t_[1], δt/2, signals, n_sites, a_q, tmpV, tmpM_, tmpK_)
    # transform!(σ, V, tmpV)
    # transform!(ψ, V, tmpV)


    for i ∈ (2:n_signals)
        t_i = t_[i]-δt/2
        t_f = t_[i]+δt/2
        gradient_eachstep!(∂Ω, i, σ, ψ, t_i,δt , signals,
                                n_sites, a_q,tmpV,tmpM_,tmpK_)

        parameters = [signals, n_sites, drives, eigvalues,false,δt/100]
        prob_ψ = ODEProblem(dψdt_grad!, ψ, (t_i, t_f), parameters)
        sol_ψ = solve(prob_ψ,dt=δt/100,adaptive=false, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
        ψ .= sol_ψ.u[end]
        prob_σ = ODEProblem(dψdt_grad!, σ, (t_i, t_f), parameters)
        sol_σ  = solve(prob_σ, dt=δt/100,adaptive=false,abstol=tol_ode, reltol=tol_ode,save_everystep=false)
        σ .= sol_σ.u[end]
        transform!(σ, V, tmpV)
        transform!(ψ, V, tmpV)

        # parameters = [signals, n_sites, drives, eigvalues,false]
        # prob_ψ = ODEProblem(dψdt, ψ, (t_i, t_f), parameters)
        # sol_ψ = solve(prob_ψ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
        # ψ .= sol_ψ.u[end]
        # prob_σ = ODEProblem(dψdt, σ, (t_i, t_f), parameters)
        # sol_σ  = solve(prob_σ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
        # σ .= sol_σ.u[end]
        # transform!(σ, eigvectors, tmp_σ )           # transform the state to the eigenspace  #rotating the state to the eigenspace
        # transform!(ψ, eigvectors, tmp_ψ)           # transform the state to the eigenspace  #rotating the state to the eigenspace


        # σ .= _step(σ, t_i, δt, signals, n_sites, a_q, tmpV, tmpM_, tmpK_)
        # ψ .= _step(ψ, t_i, δt, signals, n_sites, a_q, tmpV, tmpM_, tmpK_)
        # transform!(σ, V, tmpV)
        # transform!(ψ, V, tmpV)


    end
    gradient_eachstep!(∂Ω, n_signals+1, σ, ψ, t_[end]-δt/2 ,δt/2 , signals,
                                n_sites, a_q,tmpV,tmpM_,tmpK_)
                                
    parameters = [signals, n_sites, drives, eigvalues,false,δt/100]
    prob_ψ = ODEProblem(dψdt_grad!, ψ, (T-δt/2, T), parameters)
    sol_ψ  = solve(prob_ψ, dt=δt/100,adaptive=false,abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    ψ .= sol_ψ.u[end]
    prob_σ = ODEProblem(dψdt_grad!, σ, (T-δt/2, T), parameters)
    sol_σ  = solve(prob_σ, dt=δt/100,adaptive=false,abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    σ .= sol_σ.u[end]

    
    # parameters = [signals, n_sites, drives, eigvalues,false]
    # prob_ψ = ODEProblem(dψdt!, ψ, (T-δt/2, T), parameters)
    # sol_ψ  = solve(prob_ψ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    # ψ .= sol_ψ.u[end]
    # prob_σ = ODEProblem(dψdt!, σ, (T-δt/2, T), parameters)
    # sol_σ  = solve(prob_σ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    # σ .= sol_σ.u[end]


    # σ .= _step(σ, t_[end], δt/2, signals, n_sites, a_q, tmpV, tmpM_, tmpK_)
    # ψ .= _step(ψ, t_[end], δt/2, signals, n_sites, a_q, tmpV, tmpM_, tmpK_)

    return ∂Ω
end



function gradient_eachstep!(∂Ω, i, σ, ψ, t, τ, multi_signal, n_qubits, a, tmpV, tmpM_, tmpK_)
    for q ∈ 1:n_qubits
        tmpM_[q] .= one(a)
    end
    for q ∈ 1:n_qubits
        ν = frequency(multi_signal.channels[q], t)
        z = exp(im*ν*t)
        tmpM_[q] .= z .* a                  # ADD za TERM
        tmpM_[q] .= Hermitian(tmpM_[q])     # Hermitian ADDS a'
        O = kron_concat(tmpM_, tmpK_)
        mul!(tmpV, O, ψ)

        # CALCULATE GRADIENT
        σAψ = -im*τ * (σ' * tmpV)       # THE BRAKET ⟨σ|A|ψ⟩
        # σAψ = -im * (σ' * tmpV) 
        ∂Ω[i,q] = σAψ + σAψ'              # THE GRADIENT ⟨σ|A|ψ⟩ + ⟨ψ|A|σ⟩
        tmpM_[q] .= one(a)
    end
end




""" Auxiliary function to evolve a single step in time. """
function _step(ψ, t, τ, signals,n_qubits, a, tmpV, tmpM_, tmpK_, adjoint=false)
   
    for q ∈ 1:n_qubits
        Ω = amplitude(signals.channels[q], t)
        ν = frequency(signals.channels[q], t)
        z = Ω * exp(im*ν*t)
        tmpM_[q] .= z .* a  

        # construct the time evolution operator
        tmpM_[q] .= exp(( ((-1)^adjoint)* -im*τ) .* Hermitian(tmpM_[q]))
    end
    O = kron_concat(tmpM_, tmpK_)
    return mul!(tmpV, O, ψ)
end


function dψdt_grad!(dψ,ψ,parameters,t)

    signals   = parameters[1]
    n_sites   = parameters[2]
    drives    = parameters[3]
    eigvalues = parameters[4]
    adjoint   = parameters[5]
    dt = parameters[6]
    

    dim= length(ψ)
    # println(t)
    H = zeros(ComplexF64, dim, dim)
    device_action = Vector{ComplexF64}(undef, dim)
    for k in 1:n_sites
        H .+= amplitude(signals.channels[k], t)*exp(im*frequency(signals.channels[k], t)*t).*drives[k]
    end
    H .+= H'
    # constructing interaction picture Hamiltonian
    device_action .= exp.((-im*dt*(-1)^adjoint).*eigvalues)
    H_interaction = Diagonal(device_action)*H*(Diagonal(device_action))'

    # constructing the time derivative or schrodinger equation
    dψ .= -im*H_interaction*ψ
end