using SimpleEvolve
using NPZ
using Plots
using LinearAlgebra
using Optim
using LineSearches
using Random
using  FFTW
using  DSP
using JLD2

Cost_ham = npzread("h6_linear_op_stretched.npy") 
# display(Cost_ham)
n_qubits = round(Int, log2(size(Cost_ham,1)))
n_levels = 2
SYSTEM="H6_1.5"
freqs = 2π*collect(4.8 .+ (0.02 * (1:n_qubits)))
anharmonicities = 2π*0.3 * ones(n_qubits)
coupling_map = Dict{QubitCoupling,Float64}()
for p in 1:n_qubits
    q = (p == n_qubits) ? 1 : p + 1
    coupling_map[QubitCoupling(p,q)] = 2π*0.02
end
device = Transmon(freqs, anharmonicities, coupling_map, n_qubits)


T=500.0
n_samples = 1000
δt = T/n_samples
t_=collect(0:δt:T)


# INITIAL PARAMETERS
samples_matrix = [2π*0.000000002* sin(2π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits] 
# display(samples_matrix)
samples_matrix = [samples_matrix[:,i] .+ im*samples_matrix[:,i] for i in 1:n_qubits] 
samples_matrix = hcat(samples_matrix...)
pulse_windows=range(0, T, length=n_samples+1)
samples_initial = [real(samples_matrix[:]); imag(samples_matrix[:])]
# carrier_freqs = freqs.-2π*0.2
carrier_freqs = freqs
signals_ = [DigitizedSignal([samples_matrix[:,i]], δt, carrier_freqs[i]) for i in 1:n_qubits]
signals = MultiChannelSignal(signals_)


# # initial state # qiskit mapping
# initial_state = "111000000"
initial_state = "000000111"
ψ_initial = zeros(ComplexF64, n_levels^n_qubits)  
ψ_initial[1 + parse(Int, initial_state, base=n_levels)] = one(ComplexF64) 

H_static = static_hamiltonian(device, n_levels)
#eigenvalues and eigenvectors of the static Hamiltonian
drives =a_fullspace(n_qubits, n_levels)
eigvalues, eigvectors = eigen(Hermitian(H_static))  # Ensures real eigenvalues
# println("Eignvalues of our static Hamiltonian")
# display(eigvalues)
tol_ode=1e-6
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
δΩ_ = Matrix{Float64}(undef, 2*(n_samples+1), n_qubits)
∂Ω0 = Matrix{Float64}(undef, n_samples_grad+1, n_qubits)
τ = T/n_samples_grad
a=a_q(n_levels)


function gradient_ode!(Grad::Vector{Float64}, samples::Vector{Float64}; λ=1.0, Ω₀=1.0+2π+0.02)
    # Split real vector into real and imaginary components
    n = length(samples) ÷ 2
    samples_real = samples[1:n]
    samples_imag = samples[n+1:end]
    
    # Reshape into complex matrix
    samples_complex = complex.(
        reshape(samples_real, (n_samples+1, n_qubits)),
        reshape(samples_imag, (n_samples+1, n_qubits))
    )
    fft_before = Matrix{ComplexF64}(undef, n_samples+1, n_qubits)
    fft_after = similar(fft_before)
    max_freq_before = Vector{Float64}(undef, n_qubits)
    max_freq_after = similar(max_freq_before)

    # Apply lowpass filter to each qubit's pulse
    fs = 1/(sample_rate)      # Sampling frequency (GHz)
    cutoff = fs/2.0001  # Cutoff frequency (GHz)
    order = 4        # Filter order

    for i in 1:n_qubits
        pulse = samples_complex[:, i]

        # Compute FFT before filtering
        fft_before[:, i] = fft(pulse)
        # Apply lowpass filter
        filtered_pulse = SimpleEvolve.lowpass_filter(pulse, cutoff, fs; order=order)
        samples_complex[:, i] = filtered_pulse
        
        # Compute FFT after filtering
        fft_after[:, i] = fft(filtered_pulse)
        
        # Calculate maximum frequency magnitude
        max_freq_before[i] = maximum(abs.(fft_before[:, i]))
        max_freq_after[i] = maximum(abs.(fft_after[:, i]))
    end
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
    
    

    if n_samples==n_samples_grad
        # Flatten and concatenate gradients for BFGS
        Grad[1:n] = vec(∂Ω_real)
        Grad[n+1:end] = vec(∂Ω_imag)
    else
        Grad = validate_and_expand(δΩ_, grad_ode,
                                                n_samples_grad,
                                                n_samples,
                                                n_qubits, 
                                                T, 
                                                carrier_freqs,
                                                :whittaker_shannon)
    end
    # Step 3: Compute penalty gradient and add to total gradient
    Grad = reshape(Grad, 2*(n_samples+1), n_qubits)

    for k in 1:n_qubits
        for i in 1:n_samples+1
            grad_fidelity = Grad[i, k]
            x = samples_complex[i, k] / Ω₀
            y = abs(x) - 1

            grad_penalty = 0.0
            if y > 0
                h = exp(y - 1 / y)
                dh_dx = h * (1 + 1 / y^2) / Ω₀
                grad_penalty = sign(x) * dh_dx
            end

            Grad_final[i, k] = grad_fidelity + λ * grad_penalty
        end
    end
    return Grad_final
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
    # display(size(samples_complex))
    fft_before = Matrix{ComplexF64}(undef, n_samples+1, n_qubits)
    fft_after = similar(fft_before)
    max_freq_before = Vector{Float64}(undef, n_qubits)
    max_freq_after = similar(max_freq_before)

    # Apply lowpass filter to each qubit's pulse
    fs = 1/(sample_rate)      # Sampling frequency (GHz)
    cutoff = fs/2.0001  # Cutoff frequency (GHz)
    order = 4        # Filter order

    for i in 1:n_qubits
        pulse = samples_complex[:, i]

        # Compute FFT before filtering
        fft_before[:, i] = fft(pulse)
        # Apply lowpass filter
        filtered_pulse = SimpleEvolve.lowpass_filter(pulse, cutoff, fs; order=order)
        samples_complex[:, i] = filtered_pulse
        
        # Compute FFT after filtering
        fft_after[:, i] = fft(filtered_pulse)
        
        # Calculate maximum frequency magnitude
        max_freq_before[i] = maximum(abs.(fft_before[:, i]))
        max_freq_after[i] = maximum(abs.(fft_after[:, i]))
    end
    # Build signals from complex samples
    signals_ = [DigitizedSignal(samples_complex[:,i], δt, carrier_freqs[i]) for i in 1:n_qubits]
    signals = MultiChannelSignal(signals_)
    
    # Compute energy (existing logic)
    energy,ϕ = costfunction_ode_with_penalty(ψ_initial, eigvalues, signals, n_qubits, drives,eigvectors,  T,Cost_ham,basis="qubitbasis",tol_ode=tol_ode, λ=1.0)   
    return energy
end


sample_rate = 0.2
samples_initial = [real(samples_matrix[:]); imag(samples_matrix[:])]
Grad = zeros(Float64, 2*(n_samples+1), n_qubits)
Grad_final = zeros(Float64, 2*(n_samples+1), n_qubits)
samples_0=zeros(length(samples_initial))
@time energy_hf = costfunction_o(samples_0)
println("Hartree Fock energy ",energy_hf)
@time energy1= costfunction_o(samples_initial)   
println("initial energy ",energy1)


# OPTIMIZATION ALGORITHM
linesearch = LineSearches.MoreThuente()
# optimizer = Optim.BFGS(linesearch=linesearch)
optimizer = Optim.LBFGS(linesearch=linesearch)
# OPTIMIZATION OPTIONS
options = Optim.Options(
        show_trace = true,
        show_every = 1,
        f_reltol   = 1e-15,
        g_tol      = 1e-6,
        iterations = 1000,
)

tol_ode = 1e-5
println("Starting optimization... with tol_ode = $tol_ode") 
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_initial, optimizer, options)
samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
tol_ode = 1e-7
println("Starting optimization... with tol_ode = $tol_ode")
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
tol_ode = 1e-9
println("Starting optimization... with tol_ode = $tol_ode")
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS
@save "final_samples_H6_0.75.jld2" samples_final