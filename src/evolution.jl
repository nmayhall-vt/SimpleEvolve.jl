using SimpleEvolve
using Plots
using Random
using LinearAlgebra
using DifferentialEquations



function dψdt!(dψ,ψ,parameters,t)

    signals   = parameters[1]
    n_sites   = parameters[2]
    drives    = parameters[3]
    eigvalues = parameters[4]

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

    # constructing the time derivative or schrodinger equation
    dψ .= -im*H_interaction*ψ
end

function evolve_ODE(ψ0,
                    T,
                    signals,
                    n_sites,
                    drives,
                    eigvalues)

    # eigvalues, eigvecs = eigen(Hstatic)
    ψ = copy(ψ0)

    #evolve the state with ODE
    parameters = [signals, n_sites, drives, eigvalues]
    prob = ODEProblem(dψdt!, ψ, (0.0,T), parameters)
    sol  = solve(prob, BS3(), abstol=1e-8, reltol=1e-8,save_everystep=false)
    ψ   .= sol.u[end]
    #normalize the states
    ψ .= ψ/norm(ψ)
    return ψ

end



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
    H_interaction_ = exp((-im*dt*(-1)^adjoint).*H_interaction)
    #
    #apply the time evolution operator
    ψ .= H_interaction_*ψ
    return ψ
end
function infidelity(ψ,φ)
    return 1 - abs2(ψ'*φ)
end