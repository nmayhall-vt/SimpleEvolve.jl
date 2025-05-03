using SimpleEvolve
using LinearAlgebra
"""
penalty_function(Ω::Vector{Float64}, Ω₀::Float64)
    Computes the penalty term for the control amplitudes Ω.
        f(x) = exp(y - 1 / y) for y > 0, y= abs(x) - 1
        
    Args:
        Ω   : Control amplitudes
        Ω₀  : Reference amplitude
    Returns:
        penalty : Computed penalty term
"""

function penalty_function(Ω::Vector{Float64}, Ω₀::Float64)
    penalty = 0.0
    for x in Ω ./ Ω₀
        y = abs(x) - 1
        if y > 0
            penalty += exp(y - 1 / y)
        end
    end
    return penalty
end
"""
penalty_gradient(Ω::Vector{Float64}, Ω₀::Float64)
    Computes the gradient of the penalty term for the control amplitudes Ω.
        df(x)/dx = sign(x) * h * (1 + 1 / y^2) / Ω₀ for y > 0, y= abs(x) - 1
        where h = exp(y - 1 / y)
        
    Args:
        Ω   : Control amplitudes
        Ω₀  : Reference amplitude
    Returns:
        grad : Computed gradient of the penalty term

"""
function penalty_gradient(Ω::Vector{Float64}, Ω₀::Float64)
    grad = zeros(length(Ω))
    for (i, x) in enumerate(Ω ./ Ω₀)
        y = abs(x) - 1
        if y > 0
            h = exp(y - 1 / y)
            dh_dx = h * (1 + 1 / y^2) / Ω₀
            grad[i] = sign(x) * dh_dx
        end
    end
    return grad
end
"""
costfunction_ode_with_penalty(ψ0::Vector{ComplexF64}, eigvals::Vector{Float64}, signal, n_sites::Int, drives, eigvectors::Matrix{ComplexF64}, T::Float64, Cost_ham; basis = "eigenbasis", tol_ode=1e-8, λ::Float64=0.1, Ω₀::Float64=1.0+2π+0.02)
    Computes the cost function with a penalty term for the control amplitudes.
    
    Args:
        ψ0         : Initial state vector
        eigvals    : Eigenvalues of the Hamiltonian
        signal     : Control signal
        n_sites    : Number of sites
        drives     : Drives applied to the system
        eigvectors : Eigenvectors of the Hamiltonian
        T          : Total time for evolution
        Cost_ham   : Hamiltonian for cost function
        basis      : Basis for the computation (default: "eigenbasis")
        tol_ode    : Tolerance for ODE solver (default: 1e-8)
        λ          : Penalty weight (default: 0.1)
        Ω₀         : Reference amplitude (default: 1.0 + 2π + 0.02)
        
    Returns:
        cost      : Computed cost function value with penalty
        ψ_ode      : Evolved state vector

"""


function costfunction_ode_with_penalty(ψ0::Vector{ComplexF64},
                         eigvals::Vector{Float64},
                         signal, 
                         n_sites::Int, 
                         drives,
                         eigvectors::Matrix{ComplexF64}, 
                         T::Float64, 
                         Cost_ham;
                         basis = "eigenbasis",
                         tol_ode=1e-8,
                         λ::Float64=0.1,
                         Ω₀::Float64=1.0+2π+0.02) 

    # Evolve the state using ODE
    ψ_ode = evolve_ODE(ψ0, T, signal, n_sites, drives, eigvals, eigvectors;
                       basis=basis, tol_ode=tol_ode)
    fidelity_cost = real(ψ_ode' * Cost_ham * ψ_ode)

    # Extract control amplitudes Ω from signal
    n_timesteps = length(signal.channels[1].samples)
    Ω = zeros(n_timesteps, n_sites)
    for i in 1:n_sites
        Ω[:, i] = signal.channels[i].samples
    end
    Ω_flat = reshape(Ω, :)

    # Compute penalty
    penalty = penalty_function(Ω_flat, Ω₀)

    return fidelity_cost + λ * penalty, ψ_ode
end

#=
We can use this function in the script directly for gradient with penalty 
=#

function gradient_ode_opt!(Grad, samples; λ=0.1, Ω₀=1.0+2π+0.02)
    Grad = reshape(Grad, :, n_qubits)
    samples = reshape(samples, n_samples+1, n_qubits)

    # Step 1: Reconstruct MultiChannelSignal
    signals_ = [DigitizedSignal(samples[:, i], δt, carrier_freqs[i]) for i in 1:n_qubits]
    signals = MultiChannelSignal(signals_)

    # Step 2: Compute fidelity gradient
    grad_ode, ψ_ode, σ_ode = SimpleEvolve.gradientsignal_ODE(ψ_initial,
                            T,
                            signals,
                            n_qubits,
                            drives,
                            eigvalues,
                            eigvectors,
                            Cost_ham,
                            n_samples_grad,
                            δΩ;
                            basis="qubitbasis",
                            tol_ode=tol_ode)

    grad_ode_expanded = validate_and_expand(δΩ_, grad_ode,
                                            n_samples_grad,
                                            n_samples,
                                            n_qubits, 
                                            T, 
                                            carrier_freqs,
                                            :whittaker_shannon)

    # Step 3: Compute penalty gradient and add to total gradient
    for k in 1:n_qubits
        for i in 1:n_samples+1
            grad_fidelity = grad_ode_expanded[i, k]
            x = samples[i, k] / Ω₀
            y = abs(x) - 1

            grad_penalty = 0.0
            if y > 0
                h = exp(y - 1 / y)
                dh_dx = h * (1 + 1 / y^2) / Ω₀
                grad_penalty = sign(x) * dh_dx
            end

            Grad[i, k] = grad_fidelity + λ * grad_penalty
        end
    end
    return Grad
end
#=
We can use this function in the script directly for costfunction with penalty 
=#
function costfunction_o_opt(samples; λ=0.1, Ω₀=1.0+2π+0.02)
    # Reshape the flat vector into (n_samples+1) × n_qubits
    samples = reshape(samples, n_samples+1, n_qubits)

    # Build signals
    signals_ = [DigitizedSignal(samples[:, i], δt, carrier_freqs[i]) for i in 1:n_qubits]
    signals = MultiChannelSignal(signals_)

    # Compute original cost (e.g., expectation of Cost_ham)
    energy, ϕ = costfunction_ode(ψ_initial, eigvalues, signals, n_qubits,
                                 drives, eigvectors, T, Cost_ham;
                                 basis="qubitbasis", tol_ode=tol_ode)

    # Compute penalty term
    penalty = 0.0
    for k in 1:n_qubits
        for i in 1:n_samples+1
            x = samples[i, k] / Ω₀
            y = abs(x) - 1
            if y > 0
                penalty += exp(y - 1 / y)
            end
        end
    end

    return energy + λ * penalty
end
