using SimpleEvolve
using NPZ
using Plots
using LinearAlgebra
using Optim
using LineSearches
using ForwardDiff
using ForwardDiff: GradientConfig, Chunk

Cost_ham = npzread("h215.npy") 
display(Cost_ham)
n_qubits = round(Int, log2(size(Cost_ham,1)))
n_levels = 2
SYSTEM="h215"
freqs = 2π*collect(4.8 .+ (0.02 * (1:n_qubits)))
anharmonicities = 2π*0.3 * ones(n_qubits)
coupling_map = Dict{QubitCoupling,Float64}()
for p in 1:n_qubits
    q = (p == n_qubits) ? 1 : p + 1
    coupling_map[QubitCoupling(p,q)] = 2π*0.02
end
device = Transmon(freqs, anharmonicities, coupling_map, n_qubits)


T=10
n_samples = 600
δt = T/n_samples


# INITIAL PARAMETERS
# samples_matrix=[2π*0.02*sin(2π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits] 
samples_matrix=[sin(2π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits] 
pulse_windows=range(0, T, length=n_samples+1)

samples_initial=reshape(samples_matrix, :)
carrier_freqs = [30.207288739056587,30.48828132829821]
# carrier_freqs = [30.207288739056587,30.207288739056587]
# signals_ = [DigitizedSignal([2π*0.02* sin(2π*(t/n_samples)) for t in 0:n_samples], δt, f) for f in carrier_freqs]
signals_ = [DigitizedSignal([sin(2π*(t/n_samples)) for t in 0:n_samples], δt, f) for f in carrier_freqs]
signals = MultiChannelSignal(signals_)


# initial state
initial_state = "1"^(n_qubits÷2) * "0"^(n_qubits÷2)
# ψ_initial = zeros(ComplexF64, n_levels^n_qubits)  
ψ_initial = zeros(Complex{eltype(samples_initial)}, n_levels^n_qubits)                            
ψ_initial[1 + parse(Int, initial_state, base=n_levels)] = one(ComplexF64) 

H_static = static_hamiltonian(device, n_levels) |> Matrix{Complex{eltype(samples_initial)}}
# H_static = static_hamiltonian(device, n_levels)
#eigenvalues and eigenvectors of the static Hamiltonian
drives =a_fullspace(n_qubits, n_levels)
eigvalues, eigvecs = eigen(Hermitian(H_static))  # Ensures real eigenvalues
eigvalues = complex.(eigvalues)                  # Explicit complex conversion
println("Eignvalues of our static Hamiltonian")
display(eigvalues)

tol_ode=1e-10
Λ, U = eigen(Cost_ham)
E_actual = Λ[1]
println("Actual energy: $E_actual") 
# display(drives[1])
display(eigvecs)
for i in 1:n_qubits
    drives[i] = eigvecs' * drives[i] * eigvecs
end

function costfunction(samples)
    # considering the frequency remain as constants
    # samples is a vector of amplitudes as a function of time
    # vector is considered to optimize using BFGS
    samples = reshape(samples, n_samples+1, n_qubits)
    signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
    signals= MultiChannelSignal(signals_)
    energy,ϕ = costfunction_ode(ψ_initial, eigvalues, signals, n_qubits, drives,eigvecs,  T,Cost_ham,tol_ode=tol_ode)   
    return energy
end

# function gradient_fd!(Grad, samples)
#     samples = reshape(samples, n_samples+1, n_qubits)
#     signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
#     signals= MultiChannelSignal(signals_)
#     for i in 1:n_samples
#         E0,ϕ0 = costfunction_ode(ψ_initial, eigvalues, signals, n_qubits, drives,0.0+(i-1)* δt,Cost_ham,tol_ode=tol_ode)
#         E1,ϕ1 = costfunction_ode(ψ_initial, eigvalues, signals, n_qubits, drives,i*δt,Cost_ham,tol_ode=tol_ode)
        
#         Grad[i,:] .= (E1 - E0) /  δt
#     end
#     return Grad
# end




# we have to optimize the samples in the signal
n_samples_grad = n_samples
δΩ_ = Matrix{Float64}(undef, n_samples+1, n_qubits)
∂Ω0 = Matrix{Float64}(undef, n_samples_grad+1, n_qubits)
τ = T/n_samples_grad
device_action_independent_t = exp.((-im*τ).*eigvalues)
a=a_q(n_levels)
tol_ode=1e-10

function gradient_rotate!(Grad, samples)
    Grad = reshape(Grad, :, n_qubits)
    samples = reshape(samples, n_samples+1, n_qubits)
    signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
    signals= MultiChannelSignal(signals_)
    grad_ode =gradientsignal_ODE_rotate(ψ_initial,
                            T,
                            signals,
                            n_qubits,
                            n_levels,
                            drives,
                            a,
                            eigvalues,
                            eigvecs,
                            Cost_ham,
                            n_samples_grad,
                            ∂Ω0;
                            tol_ode=tol_ode)

    for k in 1:n_qubits
        for i in 1:n_samples+1
            # considering the frequency remain as constants
            Grad[i,k] = grad_ode[i,k] 
        end
    end
        
        # display(Grad)
    return Grad
end

function gradient_ode!(Grad, samples)
    Grad = reshape(Grad, :, n_qubits)
    samples = reshape(samples, n_samples+1, n_qubits)
    signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
    signals= MultiChannelSignal(signals_)
    grad_ode =gradientsignal_ODE(ψ_initial,
                            T,
                            signals,
                            n_qubits,
                            drives,
                            eigvalues,
                            eigvecs,
                            Cost_ham,
                            n_samples_grad,
                            ∂Ω0;
                            tol_ode=tol_ode)
    for k in 1:n_qubits
        for i in 1:n_samples+1
            # considering the frequency remain as constants
            Grad[i,k] = grad_ode[i,k] 
        end
    end
        
        # display(Grad)
    return Grad
end
# OPTIMIZATION ALGORITHM
linesearch = LineSearches.MoreThuente()
# optimizer = Optim.BFGS(linesearch=linesearch)
optimizer = Optim.LBFGS(linesearch=linesearch)
# OPTIMIZATION OPTIONS
options = Optim.Options(
        show_trace = true,
        show_every = 1,
        f_reltol   = 1e-9,
        g_tol      = 1e-9,
        iterations = 100,
)


# Grad = zeros(Float64, n_samples+1, n_qubits)
# grad_initial=gradient_fd!(Grad, samples_initial)
Grad = zeros(Float64, n_samples+1, n_qubits)
grad_initial=gradient_ode!(Grad, samples_initial)
Grad = zeros(Float64, n_samples+1, n_qubits)
grad_initial__=gradient_rotate!(Grad, samples_initial)
# # @assert ψ_initial isa AbstractVector{<:Complex{ForwardDiff.Dual}}
# test_grad = ForwardDiff.gradient(x -> first(costfunction_ode(ψ0, eigvals, x, ...)), samples_initial)
# @assert all(isfinite, test_grad)
# test_grad = ForwardDiff.gradient(
#     x -> first(costfunction_ode(
#         ψ_initial, 
#         eigvalues, 
#         signals, 
#         n_qubits,
#         drives,
#         T,
#         Cost_ham
#     )),
#     samples_initial
# )
# display(test_grad)

method = "trotter"
Ω0=copy(samples_matrix)

pulse_windows=range(0, T, length=n_samples+1)
Ω_plots = plot(                       
    [plot(pulse_windows, Ω0[:,q]) for q in 1:n_qubits]...,
    title = "Initial Signals",
    legend = false,
    layout = (n_qubits,1),
)
grad_plots=plot(                       
    [plot(
            pulse_windows, grad_initial__[:,q]
    ) for q in 1:n_qubits]...,
    title = "Initial Gradients",
    legend = false,
    layout = (n_qubits,1),
)
plot(Ω_plots, grad_plots, layout=(1,2))
savefig("initial_signals_$(n_qubits)_$(n_levels)_$(SYSTEM)_$(n_samples)_$(T)_$(method).pdf")



samples_initial=reshape(samples_matrix, :)
Grad = zeros(Float64, n_samples+1, n_qubits)
optimization = Optim.optimize(costfunction, gradient_rotate!, samples_initial, optimizer, options)
samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
optimization = Optim.optimize(costfunction, gradient_rotate!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
optimization = Optim.optimize(costfunction, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS
optimization = Optim.optimize(costfunction, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
optimization = Optim.optimize(costfunction, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)     # FINAL PARAMETERS




samples_final_reshaped = reshape(samples_final, n_samples+1, n_qubits)
Ω0=copy(samples_matrix)
Ω = copy(samples_final_reshaped)
pulse_windows=range(0, T, length=n_samples+1)
Ω_plots = plot(                       
    [plot(
            pulse_windows, Ω0[:,q]
    ) for q in 1:n_qubits]...,
    title = "Initial Signals",
    legend = false,
    layout = (n_qubits,1),
)
Ω_plots_final = plot(                       
    [plot(
            pulse_windows, Ω[:,q]
    ) for q in 1:n_qubits]...,
    title = "Final Signals",
    legend = false,
    layout = (n_qubits,1),
)
plot(Ω_plots, Ω_plots_final, layout=(1,2))

savefig("final_signals_$(n_qubits)_$(n_levels)_$(SYSTEM)_$(n_samples)_$(T).pdf")