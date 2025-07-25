using SimpleEvolve
using Plots
using Random
using LinearAlgebra
using DifferentialEquations

"""
dψdt!(dψ, ψ, parameters, t)
    Function to compute the time derivative of the state vector ψ
    args:
        dψ       : Time derivative of the state vector to compute
        ψ        : Current state vector
        parameters: Parameters for the Hamiltonian: 
                    signals, n_sites, drives, eigvalues
        t        : Current time
    returns:
        dψdt     : shrodinger equation


"""

function dψdt!(dψ,ψ,parameters,t)

    signals   = parameters[1]
    n_sites   = parameters[2]
    drives    = parameters[3]
    eigvalues = parameters[4]
    adjoint   = parameters[5]

    dim= length(ψ)
    # println(t)
    H = zeros(ComplexF64, dim, dim)
    device_action = Vector{ComplexF64}(undef, dim)
    # tmp_ψ = Vector{ComplexF64}(undef, dim)
    for k in 1:n_sites
        amp = amplitude(signals.channels[k], t)
        freq = frequency(signals.channels[k], t)
        term = amp * exp(im*freq*t)
        H .+= term .* drives[k]
        # H .+= amplitude(signals.channels[k], t)*exp(im*frequency(signals.channels[k], t)*t).*drives[k]
    end
    H .+= H'
    
    # constructing interaction picture Hamiltonian
    device_action .= exp.((im*t*(-1)^adjoint) .*eigvalues)                  
    expD = Diagonal(device_action)                          
    lmul!(expD, H); rmul!(H, expD') 

    # constructing the time derivative or schrodinger equation
    dψ .= -im*H*ψ
end

"""
evolve_ODE(ψ0, T, signals, n_sites, drives, eigvalues; tol_ode=1e-8)
    Function to evolve the state vector using ODE solver
    args:
        ψ0       : Initial state vector
        T        : Total time for evolution
        signals   : Signals to be evolved
        n_sites   : Number of sites in the system
        drives    : External drives applied to the system
        eigvalues : Eigenvalues of the Hamiltonian
        tol_ode   : Tolerance for ODE solver (default is 1e-8)
    returns:
        ψ        : Evolved state vector


"""

function evolve_ODE(ψ0::Vector{ComplexF64},
                    T::Float64,
                    signals,
                    n_sites::Int64,
                    drives,
                    eigvalues,
                    eigvectors::Matrix{ComplexF64};
                    basis = "eigenbasis",
                    tol_ode=1e-8)

    # eigvalues, eigvecs = eigen(Hstatic)
    if basis != "eigenbasis"
        ψ0 = eigvectors' * ψ0
    end
    tmp_ψ = zeros(ComplexF64, length(ψ0))
    ψ = copy(ψ0)

    #evolve the state with ODE
    parameters = [signals, n_sites, drives, eigvalues,false]
    prob = ODEProblem(dψdt!, ψ, (0.0,T), parameters)
    sol = solve(prob,RK4(), reltol=tol_ode, abstol=tol_ode,save_everystep=false,maxiters=1e8)
    
    ψ   .= sol.u[end]
    #normalize the states
    ψ .= ψ/norm(ψ)
    if basis != "eigenbasis" # rotating out of the eigenspace
        ψ .= mul!(tmp_ψ, eigvectors, ψ)
    end
    return ψ 

end

"""
evolve_direct_exponentiation(ψ0, T, signals, n_sites, drives, eigvalues, n_trotter_steps)
    Function to evolve the state vector using direct exponentiation to formulate the time evolution operator
    args:
        ψ0       : Initial state vector
        T        : Total time for evolution
        signals   : Signals to be evolved
        n_sites   : Number of sites in the system
        drives    : External drives applied to the system
        eigvalues : Eigenvalues of the Hamiltonian
        n_trotter_steps: Number of Trotter steps for exponentiation
    returns:
        ψ        : Evolved state vector


"""

function evolve_direct_exponentiation(ψ0::Vector{ComplexF64},
                                        T::Float64,
                                        signals,
                                        n_sites::Int64,
                                        drives,
                                        eigvalues,
                                        eigvectors::Matrix{ComplexF64};
                                        basis = "eigenbasis",
                                        n_trotter_steps=1000)
           
    # eigvalues, eigvecs = eigen(Hstatic)
    if basis != "eigenbasis"
        ψ0 = eigvectors' * ψ0
    end
    tmp_ψ = zeros(ComplexF64, length(ψ0))
    ψ = copy(ψ0)
    t_series=range(0,T,n_trotter_steps+1)
    dt= T/n_trotter_steps
    #tmp_ψ = Vector{ComplexF64}(undef, length(ψ0))
    
    # time evolution with direct exponentiation
    for i in 1:n_trotter_steps+1
        ψ .= single_trotter_exponentiation_step(ψ,signals, n_sites, drives, eigvalues, dt,t_series[i])
    end
    #normalize the states
    ψ .= ψ/norm(ψ)
    if basis != "eigenbasis" # rotating out of the eigenspace
        ψ .= mul!(tmp_ψ, eigvectors, ψ)
    end
    return ψ
end

"""
single_trotter_exponentiation_step(ψ,signals, n_sites, drives, eigvalues, dt, t; adjoint=false)
    Function to perform a single Trotter step for exponentiation
    args:
        ψ        : Current state vector
        signals   : Signals to be evolved
        n_sites   : Number of sites in the system
        drives    : External drives applied to the system
        eigvalues : Eigenvalues of the Hamiltonian
        dt       : Time step for Trotter step
        t        : Current time
        adjoint  : Boolean flag for adjoint operation (default is false)
                 to evolve the state forward or backward in time
    returns:
        device_action: Evolved state vector

"""

function single_trotter_exponentiation_step(ψ::Vector{ComplexF64},
                                signals, 
                                n_sites::Int64,
                                drives,
                                eigvalues,
                                dt::Float64,
                                t::Float64,
                                adjoint=false)
    
    dim= length(ψ)
    H = zeros(ComplexF64, dim, dim)
    device_action = Vector{ComplexF64}(undef, dim)
    #tmp_ψ = Vector{ComplexF64}(undef, dim)
    for k in 1:n_sites
        amp = amplitude(signals.channels[k], t)
        freq = frequency(signals.channels[k], t)
        term = amp * exp(im*freq*t)
        H .+= term .* drives[k]
        # H .+= amplitude(signals.channels[k], t)*exp(im*frequency(signals.channels[k], t)*t).*drives[k]
    end
    H .+= H'
    # constructing interaction picture Hamiltonian
    device_action .= exp.((im*t) .*eigvalues)                  
    expD = Diagonal(device_action)                          
    lmul!(expD, H); rmul!(H, expD') 
    # prepare time evolution operator for the trotter step
    H .= exp((-im*dt*(-1)^adjoint).*H)
    
    #apply the time evolution operator
    mul!(device_action, H, ψ)
    return device_action
end

"""
infidelity(ψ, φ)
    Function to compute the infidelity between two state vectors
    args:
        ψ : First state vector
        φ : Second state vector
    returns:
        infidelity value

"""
function infidelity(ψ,φ)
    return 1 - abs2(ψ'*φ)
end

"""
trotter_evolve
    Function to evolve the state vector using trotterization of time
    Time-evolves an initial quantum state `ψ0` under a time-dependent Hamiltonian using a second-order Trotter-Suzuki approximation in the eigenbasis of a static part of the Hamiltonian.

    # Arguments
    - `ψ0::Vector{ComplexF64}`: Initial quantum state vector in the device or eigenbasis, depending on `basis`.
    - `T::Float64`: Total time of evolution.
    - `signals`: Control signals or time-dependent coefficients that define the dynamic/drive part of the Hamiltonian. Must be compatible with `_step`.
    - `n_sites::Int64`: Number of qubits or quantum sites in the system.
    - `n_levels::Int64`: Local Hilbert space dimension for each site.
    - `a_q::Matrix{Float64}`: Precomputed matrix of annihilation operators.
    - `eigvalues`: Eigenvalues of the static (time-independent) Hamiltonian.
    - `eigvectors::Matrix{ComplexF64}`: Corresponding eigenvectors of the static Hamiltonian.

    # Optional Keyword Arguments
    - `basis::String="eigenbasis"`: Indicates the basis in which `ψ0` is defined. If `"eigenbasis"`, state is evolved and transformed accordingly; if not, appropriate basis rotations are applied.
    - `n_trotter_steps::Int=1000`: Number of Trotter time steps. A larger number improves accuracy but increases computational cost.

    # Returns
    - `ψ::Vector{ComplexF64}`: The evolved quantum state at time `T`, in the same basis as the input.


"""
function trotter_evolve(ψ0::Vector{ComplexF64},
                            T::Float64,
                            signals,
                            n_sites::Int64,
                            n_levels::Int64,
                            a_q::Matrix{Float64},
                            eigvalues,
                            eigvectors::Matrix{ComplexF64};
                            basis = "eigenbasis",
                            n_trotter_steps=1000) 
    # eigvalues, eigvecs = eigen(Hstatic)
    tmp_ψ = zeros(ComplexF64, length(ψ0))
    if basis == "eigenbasis" # rotating out of the eigenspace
        ψ0 .= mul!(tmp_ψ, eigvectors, ψ0)
    end
    ψ     = copy(ψ0)
    t_    = range(0,T,length=n_trotter_steps+1)
    δt    = T/n_trotter_steps

    # repeated device action
    V     = eigvectors*Diagonal(exp.((-im*δt ) * eigvalues)) *eigvectors'
    tmpM_ = [Matrix{ComplexF64}(undef, n_levels,n_levels) for q ∈ 1:n_sites]  
    tmpK_ = [Matrix{ComplexF64}(undef,  n_levels^q,  n_levels^q) for q ∈ 1:n_sites]


    ψ .= _step(ψ, t_[1], δt/2, signals, n_sites, a_q, tmp_ψ, tmpM_, tmpK_)
    transform!(ψ, V, tmp_ψ)
    for i ∈ (2:n_trotter_steps)
        t_i = t_[i]
        ψ .= _step(ψ, t_i, δt, signals, n_sites, a_q, tmp_ψ, tmpM_, tmpK_)
        transform!(ψ, V, tmp_ψ)                # transform!(σ, V, tmpV)=> σ=mul!(tmpV, V, σ) 
    end
    ψ .= _step(ψ, t_[end], δt/2, signals, n_sites, a_q, tmp_ψ, tmpM_, tmpK_)
    # transform!(ψ, V, tmp_ψ)
    
    transform!(ψ, eigvectors',tmp_ψ)           # transform the state to the device space
    ψ.*=exp.((im*T)*eigvalues)                 # rotate phases for final exp(iHDT)
    
    ψ ./= norm(ψ)
    if basis != "eigenbasis" # rotating out of the eigenspace
        ψ .= mul!(tmp_ψ, eigvectors, ψ)
    end
    return ψ
end

""" Auxiliary function to evolve a single step in time. """
function _step(ψ::Vector{ComplexF64},
            t::Float64,
            τ::Float64,
            signals,
            n_qubits::Int64, 
            a::Matrix{Float64},
            tmpV::Vector{ComplexF64},
            tmpM_::Vector{Matrix{ComplexF64}},
            tmpK_::Vector{Matrix{ComplexF64}},
            adjoint=false)
   
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
"""
dψdt_multiple_states!(dΨ, Ψ, parameters, t)
    Function to compute schrodinger equation of multiple state vectors
    args:
        dΨ       : Time derivative of the state vectors to compute
        Ψ        : Current state vectors
        parameters: Parameters for the Hamiltonian: 
                    signals, n_sites, drives, eigvalues, adjoint
        t        : Current time
    returns:
        dψdt     : schrodinger equation for multiple states

"""

function dψdt_multiple_states!(dΨ, Ψ, parameters, t)
    signals   = parameters[1]
    n_sites   = parameters[2]
    drives    = parameters[3]
    eigvalues = parameters[4]
    adjoint   = parameters[5]

    dim = size(Ψ, 1)
    num_states = size(Ψ, 2)
    
    # Construct time-dependent Hamiltonian
    H = zeros(ComplexF64, dim, dim)
    for k in 1:n_sites
        amp = amplitude(signals.channels[k], t)
        freq = frequency(signals.channels[k], t)
        H .+= amp * exp(im*freq*t) .* drives[k]
    end
    H .+= H'  # Ensure Hermiticity
    
    # Interaction picture transformation
    device_action = exp.((im*t*(-1)^adjoint) .* eigvalues)
    expD = Diagonal(device_action)
    lmul!(expD, H)
    rmul!(H, expD')
    
    # Apply Hamiltonian to each state column simultaneously
    @views for i in 1:num_states
        mul!(dΨ[:,i], H, Ψ[:,i], -im, false)
    end
end

"""
function to evolve multiple states using ODE solver simultaneously
"""
function evolve_ODE_multiple_states(
                                Ψ0::Matrix{ComplexF64},
                                T::Float64,
                                signals,
                                n_sites::Int,
                                drives,
                                eigvalues,
                                eigvectors::Matrix{ComplexF64};
                                basis = "eigenbasis",
                                tol_ode = 1e-8)
    # If not in eigenbasis, rotate initial states into eigenbasis
    if basis != "eigenbasis"
        Ψ0 = eigvectors' * Ψ0
    end

    # Prepare ODE parameters
    parameters = [signals, n_sites, drives, eigvalues, false]

    # Set up and solve the ODE
    prob = ODEProblem(dψdt_multiple_states!, Ψ0, (0.0, T), parameters)
    sol = solve(prob, RK4(), reltol=tol_ode, abstol=tol_ode, save_everystep=false, maxiters=1e8)

    # Extract and normalize the final states
    Ψ_final = sol.u[end]
    for i in 1:size(Ψ_final, 2)
        Ψ_final[:, i] ./= norm(Ψ_final[:, i])
    end

    # Rotate back out of eigenbasis if necessary
    if basis != "eigenbasis"
        Ψ_final = eigvectors * Ψ_final
    end

    return Ψ_final
end

