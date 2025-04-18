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
    for k in 1:n_sites
        H .+= amplitude(signals.channels[k], t)*exp(im*frequency(signals.channels[k], t)*t).*drives[k]
    end
    H .+= H'
    # constructing interaction picture Hamiltonian
    device_action .= exp.((-im*t*(-1)^adjoint).*eigvalues)
    H_interaction = Diagonal(device_action)*H*(Diagonal(device_action))'

    # constructing the time derivative or schrodinger equation
    dψ .= -im*H_interaction*ψ
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

function evolve_ODE(ψ0,
                    T,
                    signals,
                    n_sites,
                    drives,
                    eigvalues;
                    tol_ode=1e-8)

    # eigvalues, eigvecs = eigen(Hstatic)
    ψ = copy(ψ0)

    #evolve the state with ODE
    parameters = [signals, n_sites, drives, eigvalues,false]
    prob = ODEProblem(dψdt!, ψ, (0.0,T), parameters)
    sol  = solve(prob, abstol=tol_ode, reltol=tol_ode,save_everystep=false)
    ψ   .= sol.u[end]
    #normalize the states
    ψ .= ψ/norm(ψ)
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

function evolve_direct_exponentiation(ψ0,
                                        T,
                                        signals,
                                        n_sites,
                                        drives,
                                        eigvalues,
                                        n_trotter_steps)
           
    # eigvalues, eigvecs = eigen(Hstatic)
    ψ = copy(ψ0)
    t_series=range(0,T,n_trotter_steps+1)
    dt= T/n_trotter_steps
    
    # time evolution with direct exponentiation
    ψ .= single_trotter_exponentiation_step(ψ,signals, n_sites, drives, eigvalues, dt/2,t_series[1])
    for i in 2:n_trotter_steps
        ψ .= single_trotter_exponentiation_step(ψ,signals, n_sites, drives, eigvalues, dt,t_series[i])
    end
    ψ .= single_trotter_exponentiation_step(ψ,signals, n_sites, drives, eigvalues, dt/2,t_series[end])
    #normalize the states
    ψ .= ψ/norm(ψ)
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

function single_trotter_exponentiation_step(ψ,signals, n_sites, drives, eigvalues, dt, t,adjoint=false)
    
    dim= length(ψ)
    H = zeros(ComplexF64, dim, dim)
    device_action = Vector{ComplexF64}(undef, dim)
    for k in 1:n_sites
        H .+= amplitude(signals.channels[k], t)*exp(im*frequency(signals.channels[k], t)*t).*drives[k]
    end
    H .+= H'
    # constructing interaction picture Hamiltonian
    device_action .= exp.((-im*t).*eigvalues)
    H_interaction = Diagonal(device_action)*H*(Diagonal(device_action))'

    # prepare time evolution operator for the trotter step
    H_interaction .= exp((-im*dt*(-1)^adjoint).*H_interaction)
    #
    #apply the time evolution operator
    mul!(device_action, H_interaction, ψ)
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