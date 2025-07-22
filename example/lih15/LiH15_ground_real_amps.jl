using SimpleEvolve
using NPZ
using Plots
using LinearAlgebra
using Optim
using LineSearches
using Random
using JLD2

T = 32.0
println("T=", T)
Cost_ham = npzread("lih30.npy")
# display(Cost_ham)
n_qubits = round(Int, log2(size(Cost_ham, 1)))
n_levels = 2
SYSTEM = "lih30"
freqs = 2π * collect(4.8 .+ (0.02 * (1:n_qubits)))
anharmonicities = 2π * 0.3 * ones(n_qubits)
coupling_map = Dict{QubitCoupling,Float64}()
for p in 1:n_qubits
    q = (p == n_qubits) ? 1 : p + 1
    coupling_map[QubitCoupling(p, q)] = 2π * 0.02
end
device = Transmon(freqs, anharmonicities, coupling_map, n_qubits)

# with detuning 
n_samples = 1000
# carrier_freqs = freqs 
carrier_freqs = freqs .- 2π * 0.3
δt = T / n_samples


# INITIAL PARAMETERS
samples_matrix = [2π * 0.0000002 * sin(2π * (t / n_samples)) for t in 0:n_samples, i in 1:n_qubits]
pulse_windows = range(0, T, length=n_samples + 1)
samples_initial=reshape(samples_matrix, :)
signals_ = [DigitizedSignal([samples_matrix[:, i]], δt, carrier_freqs[i]) for i in 1:n_qubits]
signals = MultiChannelSignal(signals_)

# Ground state (Hartree-Fock)
initial_state_ground = "0011"
ψ_initial_g = zeros(ComplexF64, n_levels^n_qubits)
ψ_initial_g[1+parse(Int, initial_state_ground, base=n_levels)] = 1.0 + 0im
ψ_initial = copy(ψ_initial_g)
# display(ψ_initial)
H_static = static_hamiltonian(device, n_levels)
#eigenvalues and eigenvectors of the static Hamiltonian
drives = a_fullspace(n_qubits, n_levels)
eigvalues, eigvectors = eigen(Hermitian(H_static))  # Ensures real eigenvalues
println("Eignvalues of our static Hamiltonian")
display(eigvalues)
tol_ode = 1e-6
Λ, U = eigen(Cost_ham)
E_actual = Λ[1]
println("Actual energy: $E_actual")
# display(drives[1])
# display(eigvectors)
for i in 1:n_qubits
    drives[i] = eigvectors' * drives[i] * eigvectors
end



# we have to optimize the samples in the signal
n_samples_grad = n_samples
δΩ_ = Matrix{Float64}(undef, n_samples + 1, n_qubits)
∂Ω0 = Matrix{Float64}(undef, n_samples_grad + 1, n_qubits)
τ = T / n_samples_grad


function gradient_ode!(Grad::Vector{Float64}, samples::Vector{Float64};
    λ=1.0, Ω₀=2π + 0.02)

    Grad .= 0.0
    samples_matrix = reshape(samples, n_samples + 1, n_qubits)
    # Construct signal object
    signals = MultiChannelSignal([
        DigitizedSignal(samples_matrix[:, i], δt, carrier_freqs[i])
        for i in 1:n_qubits
    ])

    # Call gradient computation
    ∂Ω0, ψ_ode, σ_ode = SimpleEvolve.gradientsignal_ODE_real(
        ψ_initial, T, signals, n_qubits, drives, eigvalues, eigvectors,
        Cost_ham, n_samples_grad;
        basis="qubitbasis", tol_ode=tol_ode
    )


    # === Apply Penalty if Required ===
    grad_final = reshape(∂Ω0, :)

    if penalty
        for i in 1:2*(n_samples+1)
            x = samples[i] / Ω₀
            y = abs(x) - 1
            grad_penalty = 0.0
            if y > 0
                h = exp(y - 1 / y)
                dh_dx = h * (1 + 1 / y^2) / Ω₀
                grad_penalty = sign(x) * dh_dx
            end
            grad_final[i] += λ * grad_penalty
        end
    end
    Grad .= grad_final

    return Grad
end



function costfunction_o(samples::Vector{Float64})
    samples_matrix = reshape(samples, (n_samples + 1, n_qubits))
    # Build signals from complex samples
    signals_ = [DigitizedSignal(samples_matrix[:, i], δt, carrier_freqs[i]) for i in 1:n_qubits]
    signals = MultiChannelSignal(signals_)

    # Compute energy 
    if penalty == true
        energy, Ψ_ode = SimpleEvolve.costfunction_ode_with_penalty_real(
            ψ_initial, eigvalues, signals, n_qubits, drives, eigvectors, T, Cost_ham;
            basis="qubitbasis", tol_ode=tol_ode)
    else
        energy, Ψ_ode = SimpleEvolve.costfunction_ode(
            ψ_initial, eigvalues, signals, n_qubits, drives, eigvectors, T, Cost_ham;
            basis="qubitbasis", tol_ode=tol_ode
        )
    end
    return energy
end

#initialization 
Grad = zeros(Float64, n_samples + 1, n_qubits)
Grad_final = zeros(Float64, n_samples + 1, n_qubits)
samples_0 = zeros(length(samples_initial))
penalty = true

@time energy_hf = costfunction_o(samples_0)
println("Hartree Fock energy ", energy_hf)
@time energy1 = costfunction_o(samples_initial)
println("initial energy ", energy1)


# OPTIMIZATION ALGORITHM
linesearch = LineSearches.MoreThuente()
# optimizer = Optim.BFGS(linesearch=linesearch)
optimizer = Optim.LBFGS(linesearch=linesearch)
# OPTIMIZATION OPTIONS
options = Optim.Options(
    show_trace=true,
    show_every=1,
    f_reltol=1e-12,
    g_tol=1e-8,
    iterations=1000,
)



tol_ode = 1e-4
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_initial, optimizer, options)
samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
tol_ode = 1e-6
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
tol_ode = 1e-8
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS
tol_ode = 1e-10
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS

#post processing
n = Int(length(samples_final) / 2)
samples_real = reshape(samples_final[1:n], (n_samples + 1, n_qubits))
samples_imag = reshape(samples_final[n+1:end], (n_samples + 1, n_qubits))
Ω = complex.(samples_real, samples_imag)
Ω = reshape(Ω, n_samples + 1, n_qubits)
Ω0 = copy(samples_matrix)
pulse_windows = range(0, T, length=n_samples + 1)
Ω_plots = plot(
    [plot(
        pulse_windows, real.(Ω[:, q])
    ) for q in 1:n_qubits]...,
    title="Final Signals",
    legend=false,
    layout=(n_qubits, 1),
)
Ω_plots_final = plot(
    [plot(
        pulse_windows, imag.(Ω[:, q])
    ) for q in 1:n_qubits]...,
    title="Final Signals",
    legend=false,
    layout=(n_qubits, 1),
)
plot(Ω_plots, Ω_plots_final, layout=(1, 2))
savefig("final_signals_$(n_qubits)_$(n_levels)_$(SYSTEM)_$(n_samples)_$(T)_penalty_on_amps.pdf")

