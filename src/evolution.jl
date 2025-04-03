using SimpleEvolve
using Plots
using Random
using LinearAlgebra
using DifferentialEquations


function transform!(
    x::Vector{T}, A::AbstractMatrix{<:Number}, _x::Vector{T}
) where T <: Number
    x .= mul!(_x, A, x)
end
function dψdt!(dψ,ψ,parameters,t)

    signal=parameters[1]
    n_sites = parameters[2]
    drives = parameters[3]
    eigvals = parameters[4]

    dim= length(ψ)
    H = zeros(ComplexF64, dim, dim)
    device_action = Vector{ComplexF64}(undef, dim)
    for k in 1:n_sites
        H .+= amplitude(signal, t)*exp(im*frequency(signal, t)*t).*drives[k]
    end
    H .+= H'
    # constructing interaction picture Hamiltonian
    device_action .= exp.((-im*t).*eigvals)
    H_interaction = Diagonal(device_action)*H*(Diagonal(device_action))'

    # constructing the time derivative or schrodinger equation
    dψ .= -im*H_interaction*ψ
end

function evolve_ODE(ψ0,
                    T,
                    signal,
                    n_sites,
                    drives,
                    Hstatic)
    eigvals, eigvecs = eigen(Hstatic)
    ψ = copy(ψ0)

    #evolve the state with ODE
    parameters = [signal, n_sites, drives, eigvals]
    prob = ODEProblem(dψdt!, ψ, (0.0,T), parameters)
    sol = solve(prob, BS3(), abstol=1e-10, reltol=1e-11,save_everystep=false)
    ψ = sol.u[end]
    #normalize the states
    ψ .= ψ/norm(ψ)
    return ψ

end



function evolve_direct_exponentiation(ψ0,
                                        T,
                                        signal,
                                        n_sites,
                                        drives,
                                        Hstatic,
                                        δt,
                                        n_samples)
    eigvals, eigvecs = eigen(Hstatic)
    ψ = copy(ψ0)
    t_series=range(0,T,n_samples+1)
    
    function single_trotter_exponentiation_step(ψ,signal, n_sites, drives, eigvals, δt, t)
        dim= length(ψ)
        H = zeros(ComplexF64, dim, dim)
        device_action = Vector{ComplexF64}(undef, dim)
        for k in 1:n_sites
            H .+= amplitude(signal, t)*exp(im*frequency(signal, t)*t).*drives[k]
        end
        H .+= H'
        # constructing interaction picture Hamiltonian
        device_action .= exp.((-im*t).*eigvals)
        H_interaction = Diagonal(device_action)*H*(Diagonal(device_action))'

        # prepare time evolution operator for the trotter step
        H_interaction .= exp((-im*δt).*H_interaction)
        #
        #apply the time evolution operator
            ψ .= H_interaction*ψ
        return ψ
    end
    # time evolution with direct exponentiation
    ψ .= single_trotter_exponentiation_step(ψ,signal, n_sites, drives, eigvals, δt/2,t_series[1])
    for i in 2:n_samples
        ψ .= single_trotter_exponentiation_step(ψ,signal, n_sites, drives, eigvals, δt,t_series[i])
    end
    ψ .= single_trotter_exponentiation_step(ψ,signal, n_sites, drives, eigvals, δt/2,t_series[end])
    #normalize the states
    ψ .= ψ/norm(ψ)
    return ψ
end

function infidelity(ψ,φ)
    return 1 - abs2(ψ'*φ)
end