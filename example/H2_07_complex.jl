using SimpleEvolve
using NPZ
using Plots
using LinearAlgebra
using Optim
using LineSearches
using ForwardDiff
using ForwardDiff: GradientConfig, Chunk
using Random

Cost_ham = npzread("h207.npy") 
# Cost_ham = npzread("lih30.npy")
display(Cost_ham)
n_qubits = round(Int, log2(size(Cost_ham,1)))
n_levels = 2
# SYSTEM="lih30"
SYSTEM="h207"
freqs = 2π*collect(4.8 .+ (0.02 * (1:n_qubits)))
anharmonicities = 2π*0.3 * ones(n_qubits)
coupling_map = Dict{QubitCoupling,Float64}()
for p in 1:n_qubits
    q = (p == n_qubits) ? 1 : p + 1
    coupling_map[QubitCoupling(p,q)] = 2π*0.02
end
device = Transmon(freqs, anharmonicities, coupling_map, n_qubits)
# device = choose_qubits(1:n_qubits, Transmon(
#     2π*[3.7, 4.2, 3.5, 4.0],                    # QUBIT RESONANCE FREQUENCIES
#     2π*[0.3, 0.3, 0.3, 0.3],                    # QUBIT ANHARMONICITIES
#     Dict{QubitCoupling,Float64}(                # QUBIT COUPLING CONSTANTS
#         QubitCoupling(1,2) => 2π*.018,
#         QubitCoupling(2,3) => 2π*.021,
#         QubitCoupling(3,4) => 2π*.020,
#         QubitCoupling(1,3) => 2π*.021,
#         QubitCoupling(2,4) => 2π*.020,
#         QubitCoupling(1,4) => 2π*.021,
#     )
# ))
# freqs=2π*[3.7, 4.2, 3.5, 4.0]
T=10.0
n_samples = 40
δt = T/n_samples
t_=collect(0:δt:T)
# for i in 1:n_samples+1
#     display(t_[i]) 
# end

# INITIAL PARAMETERS
samples_matrix = [sin(2π * (t / n_samples)) + im * cos(2π * (t / n_samples)) 
                 for t in 0:n_samples, _ in 1:n_qubits]
# samples_matrix=[sin(2π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits] 
pulse_windows=range(0, T, length=n_samples+1)
samples_initial=reshape(samples_matrix, :)
# carrier_freqs = freqs
carrier_freqs =freqs.-2π*0.1
# signals_ = [DigitizedSignal([sin(2π*(t/n_samples)) for t in 0:n_samples], δt, f) for f in carrier_freqs]
signals_ = [DigitizedSignal(samples_matrix[:,f], δt, carrier_freqs[f]) for f in 1:length(carrier_freqs)]
signals = MultiChannelSignal(signals_)



# initial state
initial_state = "1"^(n_qubits÷2) * "0"^(n_qubits÷2)
ψ_initial_ = zeros(ComplexF64, n_levels^n_qubits)  
ψ_initial_[1 + parse(Int, initial_state, base=n_levels)] = one(ComplexF64) 

# @time energy1,ϕ = costfunction_ode(ψ_initial_, eigvalues, signals, n_qubits, drives,eigvectors, T,Cost_ham;basis="qubitbasis",tol_ode=1e-10)   
# ψ_initial=copy(ϕ)
ψ_initial=copy(ψ_initial_)

H_static = static_hamiltonian(device, n_levels)
#eigenvalues and eigenvectors of the static Hamiltonian
drives =a_fullspace(n_qubits, n_levels)
eigvalues, eigvectors = eigen(Hermitian(H_static))  # Ensures real eigenvalues
println("Eignvalues of our static Hamiltonian")
display(eigvalues)

tol_ode=1e-10
Λ, U = eigen(Cost_ham)
E_actual = Λ[1]
println("Actual energy: $E_actual") 
# display(drives[1])
display(eigvectors)
for i in 1:n_qubits
    drives[i] = eigvectors' * drives[i] * eigvectors
end

function costfunction_o(samples::Vector{Float64})
    # Split real vector into real and imaginary components
    n = length(samples) ÷ 2
    samples_real = samples[1:n]
    samples_imag = samples[n+1:end]
    
    # Reshape into complex matrix (n_samples+1 × n_qubits)
    samples_complex = complex.(
        reshape(samples_real, (n_samples+1, n_qubits)),
        reshape(samples_imag, (n_samples+1, n_qubits))
    )
    
    # Build signals from complex samples
    signals_ = [DigitizedSignal(samples_complex[:,i], δt, carrier_freqs[i]) for i in 1:n_qubits]
    signals = MultiChannelSignal(signals_)
    
    # Compute energy (existing logic)
    energy, _ = costfunction_ode(ψ_initial, eigvalues, signals, n_qubits, drives, eigvectors, T, Cost_ham;
                               basis="qubitbasis", tol_ode=tol_ode)
    return energy
end


function costfunction_t(samples)
    samples = reshape(samples, n_samples+1, n_qubits)
    signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
    signals= MultiChannelSignal(signals_)
    energy,ϕ =costfunction_trotter(ψ_initial, eigvalues,eigvectors,signals, n_qubits,n_levels, a,Cost_ham,T;basis="qubitbasis",  n_trotter_steps=n_trotter_steps )  
    return energy
end


# we have to optimize the samples in the signal
n_samples_grad = n_samples
δΩ_ = Matrix{Float64}(undef, n_samples+1, n_qubits)
∂Ω0 = Matrix{Float64}(undef, n_samples_grad+1, n_qubits)
τ = T/n_samples_grad
a=a_q(n_levels)
tol_ode=1e-10

function gradient_rotate!(Grad, samples)
    Grad = reshape(Grad, :, n_qubits)
    samples = reshape(samples, n_samples+1, n_qubits)
    signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
    signals= MultiChannelSignal(signals_)
    grad_ode,ψ_rotate, σ_rotate =SimpleEvolve.gradientsignal_rotate(ψ_initial,
                            T,
                            signals,
                            n_qubits,
                            n_levels,
                            a,
                            eigvalues,
                            eigvectors,
                            Cost_ham,
                            n_samples_grad,
                            ∂Ω0;
                            basis="qubitbasis",
                            n_trotter_steps=n_trotter_steps)

    for k in 1:n_qubits
        for i in 1:n_samples+1
            # considering the frequency remain as constants
            Grad[i,k] = grad_ode[i,k] 
        end
    end
    return Grad
end

function gradient_ode!(Grad::Vector{Float64}, samples::Vector{Float64})
    # Split real vector into real and imaginary components
    n = length(samples) ÷ 2
    samples_real = samples[1:n]
    samples_imag = samples[n+1:end]
    
    # Reshape into complex matrix
    samples_complex = complex.(
        reshape(samples_real, (n_samples+1, n_qubits)),
        reshape(samples_imag, (n_samples+1, n_qubits))
    )
    
    # Build signals from complex samples
    signals_ = [DigitizedSignal(samples_complex[:,i], δt, carrier_freqs[i]) for i in 1:n_qubits]
    signals = MultiChannelSignal(signals_)
    
    # Compute real and imaginary gradients
    ∂Ω_real, ∂Ω_imag, ψ_ode, σ_ode = SimpleEvolve.gradientsignal_ODE(
                                                ψ_initial,
                                                T,
                                                signals,
                                                n_qubits,
                                                drives,
                                                eigvalues,
                                                eigvectors,
                                                Cost_ham,
                                                n_samples_grad; 
                                                basis="qubitbasis", 
                                                tol_ode=tol_ode
    )
    
    # Flatten and concatenate gradients for BFGS
    Grad[1:n] = vec(∂Ω_real)
    Grad[n+1:end] = vec(∂Ω_imag)
    return Grad
end





@time energy1,ϕ = costfunction_ode(ψ_initial, eigvalues, signals, n_qubits, drives,eigvectors, T,Cost_ham;basis="qubitbasis",tol_ode=1e-10)   
println("ode evolved energy is ",energy1)

# trotter direct exponentiation evolution
n_trotter_steps = n_samples
@time energy2,ψ_d = costfunction_direct_exponentiation(ψ_initial, eigvalues,eigvectors, signals, n_qubits, drives,Cost_ham, T;basis="qubitbasis", n_trotter_steps=n_trotter_steps)
println("direct evolved energy is ",energy2)


@time energy3,ψ_t = costfunction_trotter(ψ_initial, eigvalues,eigvectors,signals, n_qubits,n_levels, a,Cost_ham,T;basis="qubitbasis",  n_trotter_steps=n_trotter_steps) 
println("trotter evolved energy is ",energy3)


# OPTIMIZATION ALGORITHM
linesearch = LineSearches.MoreThuente()
optimizer = Optim.BFGS(linesearch=linesearch)
# optimizer = Optim.LBFGS(linesearch=linesearch)
# OPTIMIZATION OPTIONS
options = Optim.Options(
        show_trace = true,
        show_every = 1,
        f_reltol   = 1e-11,
        g_tol      = 1e-9,
        iterations = 200,
)
Ω0=copy(samples_matrix)
# Convert initial complex samples to real vector (real + imaginary)
samples_initial = [real(Ω0[:]); imag(Ω0[:])]
Grad = zeros(Float64, 2 * (n_samples+1) * n_qubits)
grad_initial=gradient_ode!(Grad, samples_initial)
grad_intial= reshape(grad_initial, n_samples+1, n_qubits)
grad_initial_real= grad_intial[1:n_samples+1, :]
grad_initial_imag= grad_intial[n_samples+2:end, :]
p1=plot(
    [plot(pulse_windows, grad_initial_real[:,q], label="Initial Real Gradient") for q in 1:n_qubits]...,
    title = "Initial Real Gradient",
    layout = (n_qubits,1),
    legend = false
)
p2=plot(
    [plot(pulse_windows, grad_initial_imag[:,q], label="Initial Imaginary Gradient") for q in 1:n_qubits]...,
    title = "Initial Imaginary Gradient",
    layout = (n_qubits,1),
    legend = false
)
plot((p1,p2, layout=(1,2))
savefig("initial_gradients_complex_$(n_qubits)_$(n_levels)_$(SYSTEM)_$(n_samples)_$(T).pdf")
println("Initial gradient norm: ", norm(grad_initial))


optimizer = Optim.BFGS(linesearch=LineSearches.MoreThuente())
options = Optim.Options(show_trace=true, iterations=100, f_reltol=1e-11, g_tol=1e-9)

# Optimize
result = Optim.optimize(costfunction_o, gradient_ode!, samples_initial, optimizer, options)
samples_final = Optim.minimizer(result)
result=Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(result)
result=Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(result)

# Convert back to complex
n = length(samples_final) ÷ 2
samples_final_real = samples_final[1:n]
samples_final_imag = samples_final[n+1:end]
samples_final_reshaped = complex.(
    reshape(samples_final_real, (n_samples+1, n_qubits)),
    reshape(samples_final_imag, (n_samples+1, n_qubits))
)
# Plot initial and final signals (real and imaginary)
pulse_windows = range(0, T, length=n_samples+1)

Ω_plots_real = plot(
    [plot(pulse_windows, real(Ω0[:,q]), label="Initial") for q in 1:n_qubits]...,
    title = "Real Part",
    layout = (n_qubits,1),
    legend = false
)

Ω_plots_imag = plot(
    [plot(pulse_windows, imag(Ω0[:,q]), label="Initial") for q in 1:n_qubits]...,
    title = "Imaginary Part",
    layout = (n_qubits,1),
    legend = false
)

Ω_final_real = plot(
    [plot(pulse_windows, real(samples_final_reshaped[:,q]), label="Final") for q in 1:n_qubits]...,
    title = "Real Part (Optimized)",
    layout = (n_qubits,1),
    legend = false
)

Ω_final_imag = plot(
    [plot(pulse_windows, imag(samples_final_reshaped[:,q]), label="Final") for q in 1:n_qubits]...,
    title = "Imaginary Part (Optimized)",
    layout = (n_qubits,1),
    legend = false
)
plot( Ω_final_real, Ω_final_imag, layout=(2,2))
# plot(Ω_plots_real, Ω_plots_imag, Ω_final_real, Ω_final_imag, layout=(2,2))
savefig("final_signals_complex_$(n_qubits)_$(n_levels)_$(SYSTEM)_$(n_samples)_$(T).pdf")
