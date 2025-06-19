using SimpleEvolve
using LinearAlgebra
using FFTW
using Zygote
using SpecialFunctions
# using ForwardDiff

"""
penalty_function(Ω::Vector{ComplexF64}, Ω₀::Float64)
    Computes the penalty term for the control amplitudes Ω.
        f(x) = exp(y - 1 / y) for y > 0, y= abs(x) - 1
        
    Args:
        Ω   : Control amplitudes
        Ω₀  : Reference amplitude
    Returns:
        penalty : Computed penalty term
"""

function penalty_function(Ω, Ω₀::Float64)
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
penalty_gradient(Ω::Vector{ComplexF64}, Ω₀::Float64)
    Computes the gradient of the penalty term for the control amplitudes Ω.
        df(x)/dx = sign(x) * h * (1 + 1 / y^2) / Ω₀ for y > 0, y= abs(x) - 1
        where h = exp(y - 1 / y)
        
    Args:
        Ω   : Control amplitudes
        Ω₀  : Reference amplitude
    Returns:
        grad : Computed gradient of the penalty term

"""
function penalty_gradient(Ω, Ω₀::Float64)
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
                         λ::Float64=1.0,
                         Ω₀::Float64=2π*0.02) 

    # Evolve the state using ODE
    ψ_ode = evolve_ODE(ψ0, T, signal, n_sites, drives, eigvals, eigvectors;
                       basis=basis, tol_ode=tol_ode)
    fidelity_cost = real(ψ_ode' * Cost_ham * ψ_ode)

    # Extract control amplitudes Ω from signal
    n_timesteps = length(signal.channels[1].samples)
    Ω = zeros(ComplexF64, n_timesteps, n_sites)
    for i in 1:n_sites
        Ω[:, i] = signal.channels[i].samples
    end
    Ω_flat = reshape(Ω, :)

    # Compute penalty
    penalty = penalty_function(Ω_flat, Ω₀)

    return fidelity_cost + λ * penalty, ψ_ode
end
"""
multiple states with complex pulse.
penalty terms are added in real and complex amplitudes both
"""


function costfunction_ode_ssvqe_with_penalty(
                                        Ψ0::Matrix{ComplexF64},
                                        eigvals::Vector{Float64},
                                        signal,
                                        n_sites::Int,
                                        drives,
                                        eigvectors::Matrix{ComplexF64},
                                        T::Float64,
                                        Cost_ham;
                                        basis = "eigenbasis",
                                        tol_ode = 1e-8,
                                        weights = nothing,
                                        weighted = false,
                                        λ::Float64 = 1.0,
                                        Ω₀::Float64 = 2π*0.02
)
    n_states = size(Ψ0, 2)
    if weights === nothing && weighted === true
        weights = reverse(collect(1.0:-0.1:1.0-0.1*(n_states-1)))
    end

    # Evolve all initial states
    Ψ_ode = evolve_ODE_multiple_states(
        Ψ0, T, signal, n_sites, drives, eigvals, eigvectors;
        basis = basis, tol_ode = tol_ode
    )

    # Compute energies for each state
    energies = [real(ψ' * Cost_ham * ψ) for ψ in eachcol(Ψ_ode)]
    if weighted == true
        cost = sum(energies .* weights[1:n_states])
    else
        cost = sum(energies)
    end

    # Extract control amplitudes Ω from signal
    n_timesteps = length(signal.channels[1].samples)
    Ω_real = zeros(ComplexF64, n_timesteps, n_sites)
    Ω_imag = zeros(ComplexF64, n_timesteps, n_sites)
    for i in 1:n_sites
        Ω_real[:, i] = real(signal.channels[i].samples)
        Ω_imag[:, i] = imag(signal.channels[i].samples)
    end
    Ω_flat_real = reshape(Ω_real, :)
    Ω_flat_imag = reshape(Ω_imag, :)
    Ω_total=hcat(Ω_flat_real,Ω_flat_imag)
    Ω_flat=reshape(Ω_total,:)

    # Compute penalty
    penalty = penalty_function(Ω_flat, Ω₀)

    return cost + λ * penalty, Ψ_ode, energies
end
"""
cost function that adds penalty terms for only real amplitudes

"""
function costfunction_ssvqe_with_penalty_real(
                                        Ψ0::Matrix{ComplexF64},
                                        eigvals::Vector{Float64},
                                        signal,
                                        n_sites::Int,
                                        drives,
                                        eigvectors::Matrix{ComplexF64},
                                        T::Float64,
                                        Cost_ham;
                                        basis = "eigenbasis",
                                        tol_ode = 1e-8,
                                        weights = nothing,
                                        weighted = false,
                                        λ::Float64 = 1.0,
                                        Ω₀::Float64 = 2π*0.02
)
    n_states = size(Ψ0, 2)
    if weights === nothing && weighted === true
        weights = reverse(collect(1.0:-0.1:1.0-0.1*(n_states-1)))
    end

    # Evolve all initial states
    Ψ_ode = evolve_ODE_multiple_states(
        Ψ0, T, signal, n_sites, drives, eigvals, eigvectors;
        basis = basis, tol_ode = tol_ode
    )

    # Compute energies for each state
    energies = [real(ψ' * Cost_ham * ψ) for ψ in eachcol(Ψ_ode)]
    if weighted == true
        cost = sum(energies .* weights[1:n_states])
    else
        cost = sum(energies)
    end

    # Extract control amplitudes Ω from signal
    n_timesteps = length(signal.channels[1].samples)
    Ω_real = zeros(ComplexF64, n_timesteps, n_sites)
    for i in 1:n_sites
        Ω_real[:, i] = real(signal.channels[i].samples)
    end
    Ω_flat_real = reshape(Ω_real, :)

    # Compute penalty
    penalty = penalty_function(Ω_flat_real, Ω₀)

    return cost + λ * penalty, Ψ_ode, energies
end

#=
We can use this function in the script directly for gradient with penalty with real pulse only
=#

function gradient_ode_opt!(Grad, samples; λ=1.0, Ω₀=1.0+(2π*0.02))
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



# Define fftfreq for normalized frequency axis
function fftfreq(N::Int)
    return [i <= N ÷ 2 ? i / N : (i - N) / N for i in 0:N-1]
end

"""
    bandwidth_penalty(Ω::Vector{Float64}, ν::Float64; cutoff::Float64=0.2)

Computes a smooth bandwidth penalty by suppressing high-frequency components
in the Fourier domain using a smooth low-pass filter based on the error function.

# Arguments
- `Ω::Vector{Float64}`: The input signal.
- `ν::Float64`: Sharpness of the low-pass filter transition (higher ν = sharper).
- `cutoff::Float64`: Normalized frequency cutoff ∈ (0, 0.5). Frequencies above this are penalized.

# Returns
- `penalty::Float64`: Sum of squared differences between the original and filtered signal.
"""
function bandwidth_penalty(Ω::AbstractVector{<:Real}; ν=10.0, cutoff=0.2)

    N = length(Ω)
    Ω_fft = fft(Ω)
    freqs = fftfreq(N)
    freqs_normalized = abs.(freqs)  # Ensure symmetry for real-valued signal

    # Smooth low-pass filter using the error function
    smooth_filter = 1.0 .- erf.(ν .* (freqs_normalized .- cutoff))

    # Apply filter and inverse FFT to reconstruct time-domain signal
    Ω_fft_filtered = Ω_fft .* smooth_filter
    Ω_filtered = real(ifft(Ω_fft_filtered))

    # Compute L2 penalty from the difference
    penalty = sum(abs2, Ω .- Ω_filtered)
    return penalty
end



"""
    bandwidth_penalty_and_gradient(Ω::Vector{Float64}, ν::Float64; cutoff=0.2)

Computes both the bandwidth regularization penalty and its gradient using reverse-mode AD (Zygote).

# Arguments
- `Ω`: Real-valued flattened control signal.
- `ν`: Sharpness of the low-pass filter's transition.
- `cutoff`: Normalized frequency cutoff (default: 0.2).

# Returns
- `penalty`: A scalar measuring deviation from the low-passed signal.
- `grad`: Gradient vector of the penalty w.r.t. `Ω`.
"""
Zygote.@nograd fftfreq
function bandwidth_penalty_and_gradient(Ω::Vector{Float64}, ν::Float64; cutoff=0.2)
    penalty, back = Zygote.pullback(x -> bandwidth_penalty(x; ν=ν, cutoff=cutoff), Ω)
    grad = back(1.0)[1]
    return penalty, grad
end

# Generic lowpass filter function (handles real/complex signals)
function lowpass_filter(x::AbstractVector, cutoff, fs; order=4)
    normalized_cutoff = cutoff / (fs/2)
    responsetype = Lowpass(normalized_cutoff)
    designmethod = Butterworth(order)
    return filt(digitalfilter(responsetype, designmethod), x)
end

"""
this function computes bandwidth of a signal
"""

function signal_bandwidth(signal, fs; energy_threshold=0.95)
    N = length(signal)
    spectrum = abs.(fft(signal)).^2
    spectrum = spectrum[1:div(N,2)]  # One-sided for real signals
    freqs = FFTW.fftfreq(N, 1/fs)[1:div(N,2)]

    # Normalize power spectrum
    norm_spectrum = spectrum / sum(spectrum)

    # Sort bins by energy (descending)
    sorted_inds = sortperm(norm_spectrum, rev=true)
    cum_energy = 0.0
    used_inds = Int[]

    for idx in sorted_inds
        push!(used_inds, idx)
        cum_energy += norm_spectrum[idx]
        if cum_energy >= energy_threshold
            break
        end
    end

    # Bandwidth is the difference between max and min frequency in used_inds
    minf = minimum(freqs[used_inds])
    maxf = maximum(freqs[used_inds])
    bandwidth = maxf - minf

    return bandwidth, minf, maxf
end
