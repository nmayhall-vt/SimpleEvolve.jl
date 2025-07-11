using SimpleEvolve
using NPZ
using Plots
using LinearAlgebra
using Optim
using LineSearches
using ForwardDiff
using ForwardDiff: GradientConfig, Chunk
using Random
using FFTW

Cost_ham = npzread("lih15.npy") 
display(Cost_ham)
n_qubits = round(Int, log2(size(Cost_ham,1)))
n_levels = 2
SYSTEM="lih15"

device = choose_qubits(1:n_qubits, Transmon(
    2π*[3.7, 4.2, 3.5, 4.0],                    # QUBIT RESONANCE FREQUENCIES
    2π*[0.3, 0.3, 0.3, 0.3],                    # QUBIT ANHARMONICITIES
    Dict{QubitCoupling,Float64}(                # QUBIT COUPLING CONSTANTS
        QubitCoupling(1,2) => 2π*.018,
        QubitCoupling(2,3) => 2π*.021,
        QubitCoupling(3,4) => 2π*.020,
        QubitCoupling(1,3) => 2π*.021,
        QubitCoupling(2,4) => 2π*.020,
        QubitCoupling(1,4) => 2π*.021,
    )
))

T=25.0
n_samples = 500
δt = T/n_samples
t_=collect(0:δt:T)

carrier_freqs =[23.876104167282428,
27.01769682087222,
22.61946710584651,
25.761059759436304]
# INITIAL PARAMETERS
samples_matrix=[sin(2π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits] 
pulse_windows=range(0, T, length=n_samples+1)
samples_initial=reshape(samples_matrix, :)
signals_ = [DigitizedSignal([sin(2π*(t/n_samples)) for t in 0:n_samples], δt, f) for f in carrier_freqs]
signals = MultiChannelSignal(signals_)



# initial state
initial_state = "1"^(n_qubits÷2) * "0"^(n_qubits÷2)
ψ_initial = zeros(ComplexF64, n_levels^n_qubits)  
ψ_initial[1 + parse(Int, initial_state, base=n_levels)] = one(ComplexF64) 


H_static = static_hamiltonian(device, n_levels)
#eigenvalues and eigenvectors of the static Hamiltonian
drives =a_fullspace(n_qubits, n_levels)
eigvalues, eigvectors = eigen(Hermitian(H_static))  
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


function gradient_ode!(Grad, samples)
    Grad = reshape(Grad, :, n_qubits)
    samples = reshape(samples, n_samples+1, n_qubits)
    signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
    signals= MultiChannelSignal(signals_)
    grad_ode,ψ_ode, σ_ode =SimpleEvolve.gradientsignal_ODE(ψ_initial,
                            T,
                            signals,
                            n_qubits,
                            drives,
                            eigvalues,
                            eigvectors,
                            Cost_ham,
                            n_samples,
                            ∂Ω0;
                            basis="qubitbasis",
                            tol_ode=tol_ode)
    for k in 1:n_qubits
        for i in 1:n_samples+1
            # considering the frequency remain as constants
            Grad[i,k] = grad_ode[i,k] 
        end
    end
    return Grad
end

# we have to optimize the samples in the signal
n_samples_grad = Int(n_samples/25)
δΩ_ = Matrix{Float64}(undef, n_samples+1, n_qubits)
∂Ω0 = Matrix{Float64}(undef, n_samples+1, n_qubits)

a=a_q(n_levels)
tol_ode=1e-10
Grad = zeros(Float64, n_samples+1, n_qubits)
grad_initial=gradient_ode!(Grad, samples_initial)
display(grad_initial)
plot(grad_initial[:,1], label="original signal", color=:black,linewidth=2)

# gradientsignal for less no of samples
δΩ = zeros(n_samples_grad+1,n_qubits)
Grad_ = zeros(Float64, n_samples_grad+1, n_qubits)
samples_matrix=[sin(2π*(t/n_samples_grad)) for t in 0:n_samples_grad,i in 1:n_qubits] 
samples_init=reshape(samples_matrix, :)
display(samples_init)
dt=T/n_samples_grad
signals__ = [DigitizedSignal([sin(2π*(t/n_samples_grad)) for t in 0:n_samples_grad], dt, f) for f in carrier_freqs]
signals_ = MultiChannelSignal(signals__)

function gradient_ode_!(Grad, samples)
    Grad = reshape(Grad, :, n_qubits)
    samples = reshape(samples, n_samples_grad+1, n_qubits)
    signals__= [DigitizedSignal(samples[:,i],dt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
    signals_= MultiChannelSignal(signals__)
    grad_ode,ψ_ode, σ_ode =SimpleEvolve.gradientsignal_ODE(ψ_initial,
                            T,
                            signals_,
                            n_qubits,
                            drives,
                            eigvalues,
                            eigvectors,
                            Cost_ham,
                            n_samples_grad,
                            δΩ;
                            basis="qubitbasis",
                            tol_ode=tol_ode,
                            τ=δt)
    for k in 1:n_qubits
        for i in 1:n_samples_grad+1
            # considering the frequency remain as constants
            Grad[i,k] = grad_ode[i,k] 
        end
    end
    return Grad
end
grad_ode_reduced=gradient_ode_!(Grad_, samples_init)
display(grad_ode_reduced)

validate_and_expand(δΩ_,grad_ode_reduced,
                    n_samples_grad,
                    n_samples,
                    n_qubits , 
                    T, 
                    carrier_freqs,
                    :whittaker_shannon,
                    window_radius=6)

plot!(δΩ_[:,1], label="whittaker_shannon_fft", color=:green)
# Reconstruction using whittaker_shannon with windowed FFT
δΩ_ = Matrix{Float64}(undef, n_samples+1, n_qubits)
validate_and_expand(δΩ_,grad_ode_reduced,
                    n_samples_grad,
                    n_samples,
                    n_qubits, 
                    T, 
                    carrier_freqs,
                    :whittaker_shannon_lowpass)

plot!(δΩ_[:,1], label="whittaker_shannon_lowpass", color=:cyan)
# println("Maximum absolute value of the FFT using whittaker_shannon_lowpass interpolation: ", maximum(abs.(fft(δΩ_[:,1]))[Int(4n_samples/5):end]))

δΩ_ = Matrix{Float64}(undef, n_samples+1, n_qubits)
validate_and_expand(δΩ_,grad_ode_reduced,
                    n_samples_grad,
                    n_samples,
                    n_qubits , 
                    T, 
                    carrier_freqs,
                    :polynomial)

plot!(δΩ_[:,1], label="polynomial_interpolation", color=:orange)
δΩ_ = Matrix{Float64}(undef, n_samples+1, n_qubits)
validate_and_expand(δΩ_,grad_ode_reduced,
                    n_samples_grad,
                    n_samples,
                    n_qubits , 
                    T, 
                    carrier_freqs,
                    :linear)

plot!(δΩ_[:,1], label="linear interpolation", color=:blue)
δΩ_ = Matrix{Float64}(undef, n_samples+1, n_qubits)
validate_and_expand(δΩ_,grad_ode_reduced,
                    n_samples_grad,
                    n_samples,
                    n_qubits , 
                    T, 
                    carrier_freqs,
                    :hybrid)

plot!(δΩ_[:,1], label="hybrid interpolation", color=:yellow)
# δΩ_ = Matrix{Float64}(undef, n_samples+1, n_qubits)
# validate_and_expand(δΩ_,grad_ode_reduced,
#                     n_samples_grad,
#                     n_samples,
#                     n_qubits , 
#                     T, 
#                     carrier_freqs,
#                     :trigonometric)

# # plot!(δΩ_[:,1], label="trigonometric interpolation", color=:red)
savefig("reconstructed_signal_$(n_samples)_$(n_samples_grad).pdf")