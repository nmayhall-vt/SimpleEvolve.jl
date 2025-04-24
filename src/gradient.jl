using SimpleEvolve

"""
gradientsignal_ODE(ψ0, T, signals, n_sites, drives, eigvalues, device_action_independent_t, cost_ham, n_signals, ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites); tol_ode=1e-8)
    Function to compute the gradient of the energy 
    with respect to the amplitude of the signal using ODE
    args:
        ψ0        : Initial state vector
        T         : Total time for evolution
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
                            basis = "eigenbasis",
                            tol_ode=1e-8,
                            n_trotter_steps=2000,
                            n_levels=2) 

    # eigvalues, eigvecs = eigen(Hstatic)
    tmp_σ = zeros(ComplexF64, length(ψ0))
    if basis != "eigenbasis"
        ψ0 = eigvectors' * ψ0
    end
    ψ     = copy(ψ0)
    σ     = copy(ψ0)
    t_    = range(0,T,length=n_signals+1)
    t_series=range(0,T,length=n_trotter_steps+1)
    δt    = T/n_signals
    dt    = T/n_trotter_steps
    tmp_σ = zeros(ComplexF64, length(ψ0))
    tmp_ψ = zeros(ComplexF64, length(ψ0))
    V     = eigvectors*Diagonal(exp.((-im*dt ) * eigvalues)) *eigvectors'
    # repeated_device_action = eigvectors*Diagonal(exp.((-im*δt ) * eigvalues)) *eigvectors'
    tmpM_ = [Matrix{ComplexF64}(undef, n_levels,n_levels) for q ∈ 1:n_sites]  
    tmpK_ = [Matrix{ComplexF64}(undef,  n_levels^q,  n_levels^q) for q ∈ 1:n_sites]
    a     = a_q(n_levels)

    #evolve the sigma state with ODE in forward direction
    parameters = [signals, n_sites, drives,eigvalues,false, eigvectors]
    prob = ODEProblem(dψdt!, σ, (0.0,T), parameters)
    sol  = solve(prob, abstol=tol_ode, reltol=tol_ode,save_everystep=false,maxiters=1e8)
    σ .= sol.u[end]
    σ .= σ /norm(σ)
    # σ .*=exp.((im*T)*eigvalues)                # rotate phases for final exp(iHDT)

    # if cost_Hamiltonian is in eigenbasis comment out next line
    transform!(σ, eigvectors,tmp_σ)            # transform the state to the qubitbasis
    σ .= mul!(tmp_σ,cost_ham,σ)                # calculate C|ψ⟩    
    
    # # reverse time evolution of the sigma state
    transform!(σ,eigvectors',tmp_σ)            # transform the state to the eigenbasis
    σ .*=exp.((-im*T)*eigvalues)               # rotate phases for final exp(iHDT)

    # parameters = [signals, n_sites, drives,eigvalues,true,eigvectors] #should it be true? 
    # # or the time already taking care of it in solve function, I could not find any solid proof/evidence of it
    # prob_ = ODEProblem(dψdt!, σ, (T,0.0), parameters)
    # sol_  = solve(prob_,alg_hints = [:stiff], abstol=tol_ode, reltol=tol_ode,save_everystep=false,maxiters=1e8)
    # σ .= sol_.u[end]
    # σ .= σ /norm(σ)

    transform!(σ, eigvectors, tmp_σ)          # transform the state to the qubitspace
    σ .= single_step(σ, t_series[end], dt/2, signals, n_sites, a, tmp_σ, tmpM_, tmpK_,true)
    transform!(σ, V', tmp_σ)
    for i ∈ reverse(2:n_trotter_steps)
        t_i = t_series[i]
        σ .= single_step(σ, t_i, dt, signals, n_sites, a, tmp_σ, tmpM_, tmpK_,true)
        transform!(σ, V', tmp_σ)
    end
    σ .= single_step(σ, t_series[1], dt/2, signals, n_sites, a, tmp_σ, tmpM_, tmpK_,true)
    σ .= σ /norm(σ)
    transform!(σ, eigvectors', tmp_σ)


    for i ∈ (1:n_signals)

        t_i = t_[i]
        t_f = t_[i]+δt
        gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,eigvalues,eigvectors,t_i,i)
        parameters = [signals, n_sites, drives, eigvalues,false, eigvectors]
        prob_ψ = ODEProblem(dψdt!, ψ, (t_i, t_f), parameters)
        sol_ψ = solve(prob_ψ,alg_hints = [:stiff],abstol=tol_ode, reltol=tol_ode,save_everystep=false,maxiters=1e8)
        ψ .= sol_ψ.u[end]

        prob_σ = ODEProblem(dψdt!, σ, (t_i, t_f), parameters)
        sol_σ  = solve(prob_σ,alg_hints = [:stiff], abstol=tol_ode, reltol=tol_ode,save_everystep=false,maxiters=1e8)
        σ .= sol_σ.u[end]
    
    end
    gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,eigvalues,eigvectors,(t_[end]),n_signals+1)
    
    σ .= σ /norm(σ)
    ψ .= ψ /norm(ψ)
    if basis != "eigenbasis" # rotating out of the eigenspace
        ψ .= mul!(tmp_ψ, eigvectors, ψ)
        σ .= mul!(tmp_ψ, eigvectors, σ)
    end
    return ∂Ω , ψ, σ
end

"""
Pass the evolved wavefunction so that we can apply the cost Hamiltonian
without doing backward evolution

"""
function gradientsignal_ODE_alternative(ψ0,
                            T,
                            signals,
                            n_sites,
                            drives,
                            eigvalues,
                            eigvectors,
                            cost_ham,
                            n_signals,
                            ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites);
                            basis = "eigenbasis",
                            tol_ode=1e-8) 

    # eigvalues, eigvecs = eigen(Hstatic)
    tmp_σ = zeros(ComplexF64, length(ψ0))
    if basis != "eigenbasis"
        ψ0 = eigvectors' * ψ0
    end
    ψ     = copy(ψ0)
    σ     = copy(ψ0)
    t_    = range(0,T,length=n_signals+1)
    δt    = T/n_signals
    tmp_σ = zeros(ComplexF64, length(ψ0))
    tmp_ψ = zeros(ComplexF64, length(ψ0))
    # repeated_device_action = eigvectors*Diagonal(exp.((-im*δt ) * eigvalues)) *eigvectors'

    transform!(σ, eigvectors,tmp_σ)            # transform the state to the qubitbasis
    σ .= mul!(tmp_σ,cost_ham,σ) 
    transform!(σ,eigvectors',tmp_σ)   

    
    #calculating gradient by evolving both ψ and σ states
    gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,eigvalues,eigvectors,t_[1],1)

    for i ∈ (1:n_signals)
        t_i = t_[i]
        t_f = t_[i]+δt
        parameters = [signals, n_sites, drives, eigvalues,false, eigvectors]
        prob_ψ = ODEProblem(dψdt!, ψ, (t_i, t_f), parameters)
        sol_ψ = solve(prob_ψ,abstol=tol_ode, reltol=tol_ode,save_everystep=false,maxiters=1e8)
        ψ .= sol_ψ.u[end]

        prob_σ = ODEProblem(dψdt!, σ, (t_i, t_f), parameters)
        sol_σ  = solve(prob_σ,alg_hints = [:stiff], abstol=tol_ode, reltol=tol_ode,save_everystep=false,maxiters=1e8)
        σ .= sol_σ.u[end]

        gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,eigvalues,eigvectors,t_f,i+1)
    end
   
    ψ .= ψ/norm(ψ)
    σ .= σ/norm(σ)
    if basis != "eigenbasis" # rotating out of the eigenspace
        ψ .= mul!(tmp_ψ, eigvectors, ψ)
        σ .= mul!(tmp_ψ, eigvectors, σ)
    end
    return ∂Ω, ψ, σ
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
    ψ0=copy(ψ)
    σ0=copy(σ)
    for k in 1:n_sites
        dH_dΩ .+= exp(im*frequency(multi_signal.channels[k], t)*t).*drives[k]
        dH_dΩ .+= dH_dΩ'

        # derivative of interaction picture Hamiltonian
        ## dt=multi_signal.channels[k].δt  
        device_action .= exp.((im*t) .* eigvalues) 
        ## device_action .= exp.((-im*dt) .* eigvalues)                 
        expD =Diagonal(device_action)                   
        lmul!(expD, dH_dΩ); rmul!(dH_dΩ, expD')  
        ## dH_dΩ = eigvectors' * dH_dΩ * eigvectors
        
        AΨ = dH_dΩ * ψ0
        # calculate gradient ⟨σ|A|ψ⟩
        σAψ = -im * (σ0' * AΨ)
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
                                        cost_ham,
                                        n_signals,
                                        ∂Ω = Matrix{Float64}(undef, n_signals+1, n_sites);
                                        basis = "eigenbasis",
                                        n_trotter_steps=1000)
           
    # eigvalues, eigvecs = eigen(Hstatic)
    if basis != "eigenbasis"
        ψ0 = eigvectors' * ψ0
    end
    ψ = copy(ψ0)
    σ = copy(ψ0)
    t_series=range(0,T,n_trotter_steps+1)
    t_= range(0,T,length=n_signals+1)
    dt= T/n_trotter_steps
    Δt =T/n_signals
    tmp_σ = zeros(ComplexF64, length(ψ0))
    tmp_ψ = zeros(ComplexF64, length(ψ0))


    # time evolution with direct exponentiation
    for i in 1:n_trotter_steps+1
        σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues, eigvectors,dt,t_series[i])
    end
    σ .= σ /norm(σ)
    σ .*=exp.((im*T)*eigvalues)                # rotate phases for final exp(iHDT)

    # if cost_Hamiltonian is in eigenbasis comment out next line
    transform!(σ, eigvectors,tmp_σ)            # transform the state to the qubitbasis
    σ .= mul!(tmp_σ,cost_ham,σ)                # calculate C|ψ⟩    
    
    # # reverse time evolution of the sigma state
    transform!(σ,eigvectors',tmp_σ)            # transform the state to the eigenbasis
    σ .*=exp.((-im*T)*eigvalues)                # rotate phases for final exp(iHDT)


    #time evolution backward for sigma state
    for i in reverse(1:n_trotter_steps+1)
        σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues,eigvectors, dt,t_series[i],true)
    end
    σ .= σ /norm(σ)
    #calculating gradient by evolving both ψ and σ states
    for i in 1:n_signals+1
        gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,eigvalues,eigvectors,t_[i],i)
        ψ .= single_trotter_exponentiation_step(ψ,signals, n_sites, drives, eigvalues,eigvectors, Δt,t_[i])
        σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues,eigvectors, Δt,t_[i])
    end
    gradient_eachtimestep!(∂Ω,ψ,σ,signals,n_sites,drives,eigvalues,eigvectors,t_series[end],n_signals+1)
  
    ψ .= ψ/norm(ψ)
    σ .= σ/norm(σ)
    if basis != "eigenbasis" # rotating out of the eigenspace
        ψ .= mul!(tmp_ψ, eigvectors, ψ)
        σ .= mul!(tmp_ψ, eigvectors, σ)
    end
    return ∂Ω, ψ, σ

end


function gradientsignal_rotate(ψ0,
                            T,
                            signals,
                            n_sites,
                            n_levels,
                            a_q,
                            eigvalues,
                            eigvectors,
                            cost_ham,
                            n_signals,
                            ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites);
                            n_trotter_steps=1000,
                            basis = "eigenbasis") 
    ## transform!(σ, V, tmpV)=> σ=mul!(tmpV, V, σ) 
    # eigvalues, eigvecs = eigen(Hstatic)
    tmp_σ = zeros(ComplexF64, length(ψ0))
    if basis == "eigenbasis" # rotating out of the eigenspace 
        mul!(tmp_σ, eigvectors, ψ0)
    end
    tmp_ψ = zeros(ComplexF64, length(ψ0))
    ψ     = copy(ψ0)
    σ     = copy(ψ0)
    t_    = range(0,T,length=n_signals+1)
    t_series=range(0,T,length=n_trotter_steps+1)
    δt    = T/n_signals
    Δt    = T/n_trotter_steps
    V     = eigvectors*Diagonal(exp.((-im*δt ) * eigvalues)) *eigvectors'
    V_backward = eigvectors*Diagonal(exp.((im*δt ) * eigvalues)) *eigvectors'
    tmpM_ = [Matrix{ComplexF64}(undef, n_levels,n_levels) for q ∈ 1:n_sites]  
    tmpK_ = [Matrix{ComplexF64}(undef,  n_levels^q,  n_levels^q) for q ∈ 1:n_sites]


    σ .= single_step(σ, t_series[1], Δt/2, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
    transform!(σ, V, tmp_σ)
    for i ∈ (2:n_trotter_steps)
        t_i = t_series[i]
        σ .= single_step(σ, t_i, Δt, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
        transform!(σ, V, tmp_σ)                #transform!(σ, V, tmpV)=> σ=mul!(tmpV, V, σ) 
    end
    σ .= single_step(σ, t_series[end], Δt/2, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
    σ .= σ /norm(σ)
    transform!(σ, eigvectors',tmp_σ)          # transform the state to the eigen space
    σ .*=exp.((im*T)*eigvalues)               # rotate phases for final exp(iHDT)
    transform!(σ, eigvectors, tmp_σ)          # transform the state to the qubitspace

    σ .= mul!(tmp_σ,cost_ham,σ)               # calculate C|ψ⟩    
    
    # # reverse time evolution of the sigma state
    transform!(σ,eigvectors',tmp_σ)           # transform the state to the device space
    σ .*=exp.((-im*T)*eigvalues)               # rotate phases for final exp(iHDT)
    transform!(σ, eigvectors, tmp_σ)          # transform the state to the qubitspace
    

    σ .= single_step(σ, t_series[end], Δt/2, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_,true)
    transform!(σ, V', tmp_σ)
    for i ∈ reverse(2:n_trotter_steps)
        t_i = t_series[i]
        σ .= single_step(σ, t_i, Δt, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_,true)
        transform!(σ, V', tmp_σ)
    end
    σ .= single_step(σ, t_series[1], Δt/2, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_,true)
    σ .= σ /norm(σ)

    #calculating gradient by evolving both ψ and σ states
    gradient_eachstep!(∂Ω, 1, σ, ψ, t_[1],δt/2 , signals,
                                n_sites, a_q,tmp_σ,tmpM_,tmpK_)


    σ .= single_step(σ, t_[1], δt/2, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
    ψ .= single_step(ψ, t_[1], δt/2, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
    transform!(σ, V, tmp_σ)
    transform!(ψ, V, tmp_σ)


    for i ∈ (2:n_signals)
        t_i = t_[i]
        gradient_eachstep!(∂Ω, i, σ, ψ, t_i,δt , signals,
                                n_sites, a_q,tmp_σ,tmpM_,tmpK_)
        σ .= single_step(σ, t_i, δt, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
        ψ .= single_step(ψ, t_i, δt, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
        transform!(σ, V, tmp_σ)
        transform!(ψ, V, tmp_σ)


    end
    gradient_eachstep!(∂Ω, n_signals+1, σ, ψ, t_[end] ,δt/2 , signals,
                                n_sites, a_q,tmp_σ,tmpM_,tmpK_)
    σ .= single_step(σ, t_[end], δt/2, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
    ψ .= single_step(ψ, t_[end], δt/2, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
    
    transform!(ψ, eigvectors',tmp_ψ)           # transform the state to the device space
    transform!(σ, eigvectors',tmp_ψ)
    ψ .*=exp.((im*T)*eigvalues)                 # rotate phases for final exp(iHDT)
    σ .*=exp.((im*T)*eigvalues)
    
    ψ ./= norm(ψ)
    σ ./= norm(σ)
    if basis != "eigenbasis" # rotating out of the eigenspace
        ψ .= mul!(tmp_ψ, eigvectors, ψ)
        σ .= mul!(tmp_ψ, eigvectors, σ)
    end
    return ∂Ω,ψ,σ
end
"""
Pass the evolved wavefunction so that we can apply the cost Hamiltonian
without doing backward evolution

"""

function gradientsignal_rotate_alternate(ψ0,
                            T,
                            signals,
                            n_sites,
                            n_levels,
                            a_q,
                            eigvalues,
                            eigvectors,
                            cost_ham,
                            n_signals,
                            ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites);
                            n_trotter_steps=1000,
                            basis = "eigenbasis") 
    ## transform!(σ, V, tmpV)=> σ=mul!(tmpV, V, σ) 
    # eigvalues, eigvecs = eigen(Hstatic)
    tmp_σ = zeros(ComplexF64, length(ψ0))
    if basis == "eigenbasis" # rotating out of the eigenspace 
        mul!(tmp_σ, eigvectors, ψ0)
    end
    tmp_ψ = zeros(ComplexF64, length(ψ0))
    tmp_V = zeros(ComplexF64, length(ψ0))
    ψ     = copy(ψ0)
    σ     = copy(ψ0)
    t_    = range(0,T,length=n_signals+1)
    δt    = T/n_signals
    Δt    = T/n_trotter_steps
    V     = eigvectors*Diagonal(exp.((-im*δt ) * eigvalues)) *eigvectors'
    tmpM_ = [Matrix{ComplexF64}(undef, n_levels,n_levels) for q ∈ 1:n_sites]  
    tmpK_ = [Matrix{ComplexF64}(undef,  n_levels^q,  n_levels^q) for q ∈ 1:n_sites]
    
    σ .= mul!(tmp_σ,cost_ham,σ)
    
    #calculating gradient by evolving both ψ and σ states
    gradient_eachstep!(∂Ω, 1, σ, ψ, t_[1],δt/2 , signals,
                                n_sites, a_q,tmp_V,tmpM_,tmpK_)


    σ .= single_step(σ, t_[1], δt/2, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
    ψ .= single_step(ψ, t_[1], δt/2, signals, n_sites, a_q, tmp_ψ, tmpM_, tmpK_)
    transform!(σ, V, tmp_σ)
    transform!(ψ, V, tmp_ψ)


    for i ∈ (2:n_signals)
        t_i = t_[i]
        gradient_eachstep!(∂Ω, i, σ, ψ, t_i,δt , signals,
                                n_sites, a_q,tmp_V,tmpM_,tmpK_)
        σ .= single_step(σ, t_i, δt, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
        ψ .= single_step(ψ, t_i, δt, signals, n_sites, a_q, tmp_ψ, tmpM_, tmpK_)
        transform!(σ, V, tmp_σ)
        transform!(ψ, V, tmp_ψ)
    end
    gradient_eachstep!(∂Ω, n_signals+1, σ, ψ, t_[end] ,δt/2 , signals,
                                n_sites, a_q,tmp_V,tmpM_,tmpK_)
    σ .= single_step(σ, t_[end], δt/2, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
    ψ .= single_step(ψ, t_[end], δt/2, signals, n_sites, a_q, tmp_ψ, tmpM_, tmpK_)
    
    transform!(ψ, eigvectors',tmp_ψ)           # transform the state to the device space
    transform!(σ, eigvectors',tmp_σ)
    ψ .*=exp.((im*T)*eigvalues)                 # rotate phases for final exp(iHDT)
    σ .*=exp.((im*T)*eigvalues)
    
    ψ ./= norm(ψ)
    σ ./= norm(σ)
    if basis != "eigenbasis" # rotating out of the eigenspace
        ψ .= mul!(tmp_ψ, eigvectors, ψ)
        σ .= mul!(tmp_σ, eigvectors, σ)
    end
    return ∂Ω,ψ,σ
end



function gradient_eachstep!(∂Ω, i, σ, ψ, t, τ, multi_signal, n_qubits, a, tmpV, tmpM_, tmpK_)
    for q ∈ 1:n_qubits
        tmpM_[q] .= one(a)
    end
    ψ0=copy(ψ)
    σ0=copy(σ)
    for q ∈ 1:n_qubits
        ν = frequency(multi_signal.channels[q], t)
        z = exp(im*ν*t)
        tmpM_[q] .= z .* a                  # ADD za TERM
        tmpM_[q] .= Hermitian(tmpM_[q])     # Hermitian ADDS a'
        O = kron_concat(tmpM_, tmpK_)
        mul!(tmpV, O, ψ0)

        # CALCULATE GRADIENT
        # σAψ = -im*τ * (σ0' * tmpV)       # THE BRAKET ⟨σ|A|ψ⟩
        σAψ = -im * (σ0' * tmpV) 
        ∂Ω[i,q] = σAψ + σAψ'              # THE GRADIENT ⟨σ|A|ψ⟩ + ⟨ψ|A|σ⟩
        tmpM_[q] .= one(a)
    end
end




""" Auxiliary function to evolve a single step in time. """
function single_step(ψ, t, τ, signals,n_qubits, a, tmpV, tmpM_, tmpK_, adjoint=false)
   
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
    dt        = parameters[6]
    

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

function gradientsignal_rotate_ode(ψ0,
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
                            basis = "eigenbasis",
                            tol_ode=1e-8) 

    # eigvalues, eigvecs = eigen(Hstatic)
    tmp_σ = zeros(ComplexF64, length(ψ0))
    if basis != "eigenbasis"
        ψ0 = eigvectors' * ψ0
    end
    tmp_ψ = zeros(ComplexF64, length(ψ0))
    ψ     = copy(ψ0)
    σ     = copy(ψ0)
    t_    = range(0,T,length=n_signals+1)
    δt    = T/n_signals
    N     = length(ψ0)
    V     = Diagonal(exp.((-im*δt ) * eigvalues)) 
    tmpM_ = [Matrix{ComplexF64}(undef, n_levels,n_levels) for q ∈ 1:n_sites]  
    tmpK_ = [Matrix{ComplexF64}(undef,  n_levels^q,  n_levels^q) for q ∈ 1:n_sites]

    
    #evolve the sigma state with ODE in forward direction
    parameters = [signals, n_sites, drives,eigvalues,false,eigvectors]
    prob = ODEProblem(dψdt!, σ, (0.0,T), parameters)
    sol  = solve(prob, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    σ .= sol.u[end]

    σ .*=exp.((im*T)*eigvalues)               # rotate phases for final exp(iHDT)
    transform!(σ, eigvectors, tmp_σ)          # transform the state to the qubitspace

    σ .= mul!(tmp_σ,cost_ham,σ)               # calculate C|ψ⟩    
    
    # # reverse time evolution of the sigma state
    transform!(σ,eigvectors',tmp_σ)           # transform the state to the eigen space
    σ .*=exp.((im*T)*eigvalues)               # rotate phases for final exp(iHDT)
  
    parameters = [signals, n_sites, drives,eigvalues,true,eigvectors]
    prob_ = ODEProblem(dψdt!, σ, (T,0.0), parameters)
    sol_  = solve(prob_, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    σ .= sol_.u[end]

    transform!(σ, eigvectors, tmp_σ)          # transform the state to the qubitspace
    transform!(ψ, eigvectors, tmp_σ)
    # # calculating gradient by evolving both ψ and σ states
    gradient_eachstep!(∂Ω, 1, σ, ψ, t_[1],δt/2 , signals,
                                n_sites, a_q,tmp_σ,tmpM_,tmpK_)
    transform!(σ,eigvectors',tmp_σ)           # transform the state to the eigen space
    transform!(ψ,eigvectors',tmp_σ)           
    
    # parameters = [signals, n_sites, drives, eigvalues,false,δt/100]
    # prob_ψ = ODEProblem(dψdt_grad!, ψ, (0.0,δt/2), parameters)
    # sol_ψ  = solve(prob_ψ, dt=δt/100,adaptive=false,abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    # ψ .= sol_ψ.u[end]
    # prob_σ = ODEProblem(dψdt_grad!, σ, (0.0, δt/2), parameters)
    # sol_σ  = solve(prob_σ,dt=δt/100,adaptive=false, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    # σ .= sol_σ.u[end]
    # transform!(σ, V, tmpV)
    # transform!(ψ, V, tmpV)



    parameters = [signals, n_sites, drives, eigvalues,false,eigvectors]
    prob_ψ = ODEProblem(dψdt!, ψ, (0.0,δt/2), parameters)
    sol_ψ  = solve(prob_ψ,abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    ψ .= sol_ψ.u[end]
    prob_σ = ODEProblem(dψdt!, σ, (0.0, δt/2), parameters)
    sol_σ  = solve(prob_σ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    σ .= sol_σ.u[end]

    transform!(σ, eigvectors, tmp_σ )           # transform the state to the qubitspace to compute the gradient
    transform!(ψ, eigvectors, tmp_σ)           


    for i ∈ (2:n_signals)
        t_i = t_[i]-δt/2
        t_f = t_[i]+δt/2
        gradient_eachstep!(∂Ω, i, σ, ψ, t_i,δt , signals,
                                n_sites, a_q,tmp_σ,tmpM_,tmpK_)
        # transform!(σ, V, tmp_σ)
        # transform!(ψ, V, tmp_σ)

        transform!(σ,eigvectors',tmp_σ)           # transform the state to the eigen space to evolve
        transform!(ψ,eigvectors',tmp_σ)  
                                
        # parameters = [signals, n_sites, drives, eigvalues,false,δt/100]
        # prob_ψ = ODEProblem(dψdt_grad!, ψ, (t_i, t_f), parameters)
        # sol_ψ = solve(prob_ψ,dt=δt/100,adaptive=false, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
        # ψ .= sol_ψ.u[end]
        # prob_σ = ODEProblem(dψdt_grad!, σ, (t_i, t_f), parameters)
        # sol_σ  = solve(prob_σ, dt=δt/100,adaptive=false,abstol=tol_ode, reltol=tol_ode,save_everystep=false)
        # σ .= sol_σ.u[end]
        # transform!(σ, V, tmpV)
        # transform!(ψ, V, tmpV)

        parameters = [signals, n_sites, drives, eigvalues,false,eigvectors]
        prob_ψ = ODEProblem(dψdt!, ψ, (t_i, t_f), parameters)
        sol_ψ = solve(prob_ψ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
        ψ .= sol_ψ.u[end]
        prob_σ = ODEProblem(dψdt!, σ, (t_i, t_f), parameters)
        sol_σ  = solve(prob_σ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
        σ .= sol_σ.u[end]
        transform!(σ, eigvectors, tmp_σ )           # transform the state to the qubitspace to compute the gradient
        transform!(ψ, eigvectors, tmp_σ)           

    end
    gradient_eachstep!(∂Ω, n_signals+1, σ, ψ, t_[end]-δt/2 ,δt/2 , signals,
                                n_sites, a_q,tmp_σ,tmpM_,tmpK_)
                                
    # parameters = [signals, n_sites, drives, eigvalues,false,δt/100]
    # prob_ψ = ODEProblem(dψdt_grad!, ψ, (T-δt/2, T), parameters)
    # sol_ψ  = solve(prob_ψ, dt=δt/100,adaptive=false,abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    # ψ .= sol_ψ.u[end]
    # prob_σ = ODEProblem(dψdt_grad!, σ, (T-δt/2, T), parameters)
    # sol_σ  = solve(prob_σ, dt=δt/100,adaptive=false,abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    # σ .= sol_σ.u[end]

    transform!(σ,eigvectors',tmp_σ)           # transform the state to the eigen space to evolve
    transform!(ψ,eigvectors',tmp_σ) 
    parameters = [signals, n_sites, drives, eigvalues,false,eigvectors]
    prob_ψ = ODEProblem(dψdt!, ψ, (T-δt/2, T), parameters)
    sol_ψ  = solve(prob_ψ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    ψ .= sol_ψ.u[end]
    prob_σ = ODEProblem(dψdt!, σ, (T-δt/2, T), parameters)
    sol_σ  = solve(prob_σ, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    σ .= sol_σ.u[end]
    if basis != "eigenbasis" # rotating out of the eigenspace
        ψ .= mul!(tmp_ψ, eigvectors, ψ)
        σ .= mul!(tmp_ψ, eigvectors, σ)
    end
    return ∂Ω
end

function gradientsignal_finite_difference(ψ0,
                            T,
                            signals,
                            n_sites,
                            drives,
                            eigvalues,
                            eigvectors,
                            cost_ham,
                            n_signals,
                            ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites);
                            basis = "eigenbasis",
                            ϵ = 1e-6) 

    # eigvalues, eigvecs = eigen(Hstatic)
    if basis != "eigenbasis"
        ψ0 = eigvectors' * ψ0
    end
    ψ     = copy(ψ0)
    σ     = copy(ψ0)
    t_    = range(0,T,length=n_signals+1)
    δt    = T/n_signals 
    # println("n_signals: ", n_signals)
    for i in 1:n_sites
        # Create two signal copies for ±ε perturbations
        signals_δt_plus = copy(signals)
        signals_δt_minus = copy(signals)
        
        old_sample = signals_δt_plus.channels[i].samples
        # Compute finite differences
        for j in 1:n_signals+1
            
            signals_δt_plus.channels[i].samples[j]  = old_sample[j] + ϵ
            signals_δt_minus.channels[i].samples[j] = old_sample[j] - ϵ
            t_f = t_[j] 
            # display(j)
            # Positive perturbation
            e_plus, _ = costfunction_ode(ψ, eigvalues, signals_δt_plus, n_sites, 
                                        drives, eigvectors, t_f, cost_ham)
            
            # Negative perturbation
            e_minus, _ = costfunction_ode(ψ, eigvalues, signals_δt_minus, n_sites,
                                        drives, eigvectors, t_f, cost_ham)
            
            # Central difference gradient
            ∂Ω[j, i] = (e_plus - e_minus) / (2ϵ)
            signals_δt_plus  = copy(signals)
            signals_δt_minus = copy(signals)
        end
    end

    return ∂Ω
end


function gradientsignal_fd(ψ0, T, signals, n_sites, drives, eigvalues,
                                          eigvectors, cost_ham, n_signals,
                                          ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites);
                                          basis="eigenbasis", ϵ=1e-6)
    # Basis transformation
    basis != "eigenbasis" && (ψ0 = eigvectors' * ψ0)
    
    ψ = copy(ψ0)
    t_ = range(0, T, length=n_signals+1)
    δt = T/n_signals

    # Precompute original samples for each channel
    original_samples = [copy(sig.samples) for sig in signals.channels]

    for i in 1:n_sites
        # Create lightweight copies for perturbation
        channel_plus = copy(signals.channels[i])
        channel_minus = copy(signals.channels[i])
        
        for j in 1:n_signals+1
            t_f = t_[j]

            # Perturb only the j-th sample in i-th channel
            channel_plus.samples[j] = original_samples[i][j] + ϵ
            channel_minus.samples[j] = original_samples[i][j] - ϵ

            # Build temporary signals with modified channel
            signals_plus = MultiChannelSignal(
                [k == i ? channel_plus : signals.channels[k] for k in 1:n_sites]
            )
            signals_minus = MultiChannelSignal(
                [k == i ? channel_minus : signals.channels[k] for k in 1:n_sites]
            )

            # Compute energies
            e_plus, _ = costfunction_ode(ψ, eigvalues, signals_plus, n_sites, 
                                       drives, eigvectors, t_f, cost_ham)
            # e_minus, _ = costfunction_ode(ψ, eigvalues, signals_minus, n_sites,
                                        # drives, eigvectors, t_f, cost_ham)
            e0, _ = costfunction_ode(ψ, eigvalues, signals, n_sites,
                                        drives, eigvectors, t_f, cost_ham)
            # Central difference gradient
            # ∂Ω[j, i] = (e_plus - e_minus) / (2ϵ)
            ∂Ω[j, i] = (e_plus - e0) / (ϵ)
            # Reset to original for next iteration
            channel_plus.samples[j] = original_samples[i][j]
            channel_minus.samples[j] = original_samples[i][j]
        end
    end

    return ∂Ω
end
function Base.copy(ds::DigitizedSignal{T}) where T
    # Deep copy the samples vector; Float64 fields are immutable
    return DigitizedSignal{T}(copy(ds.samples), ds.δt, ds.carrier_freq)
end