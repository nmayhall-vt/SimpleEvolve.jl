using SimpleEvolve
using NPZ
using Plots
using LinearAlgebra
using Optim
using LineSearches
using ForwardDiff
using ForwardDiff: GradientConfig, Chunk
using Random

Cost_ham = npzread("lih15.npy") 
display(Cost_ham)
n_qubits = round(Int, log2(size(Cost_ham,1)))
n_levels = 3
SYSTEM="lih15"
Π = projector(n_qubits, 2, n_levels)    
Cost_ham = Hermitian(Π'*Cost_ham*Π)
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
n_samples = 150
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


# we have to optimize the samples in the signal
n_samples_grad = Int(n_samples/3)
δΩ_ = Matrix{Float64}(undef, n_samples+1, n_qubits)
a=a_q(n_levels)
tol_ode=1e-10
# gradientsignal for less no of samples
δΩ = zeros(n_samples_grad+1,n_qubits)
Grad = zeros(Float64, n_samples+1, n_qubits)
samples_matrix=[sin(2π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits] 
samples_initial=reshape(samples_matrix, :)
signals_ = [DigitizedSignal([sin(2π*(t/n_samples)) for t in 0:n_samples], δt, f) for f in carrier_freqs]
signals = MultiChannelSignal(signals_)
dt=T/n_samples_grad

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
                            n_samples_grad,
                            δΩ;
                            basis="qubitbasis",
                            tol_ode=tol_ode)

    grad_ode_expanded =validate_and_expand(δΩ_,grad_ode,
                                            n_samples_grad,
                                            n_samples,
                                            n_qubits, 
                                            T, 
                                            carrier_freqs,
                                            :whittaker_shannon)
                                            
    for k in 1:n_qubits
        for i in 1:n_samples+1
            # considering the frequency remain as constants
            Grad[i,k] = grad_ode_expanded[i,k] 
        end
    end
    return Grad
end
# OPTIMIZATION ALGORITHM
linesearch = LineSearches.MoreThuente()
optimizer = Optim.BFGS(linesearch=linesearch)
# optimizer = Optim.LBFGS(linesearch=linesearch)
# OPTIMIZATION OPTIONS
options = Optim.Options(
        show_trace = true,
        show_every = 1,
        f_reltol   = 1e-9,
        g_tol      = 1e-9,
        iterations = 200,
)


function costfunction_o(samples)
    # considering the frequency remain as constants
    # samples is a vector of amplitudes as a function of time
    # vector is considered to optimize using BFGS
    samples = reshape(samples, n_samples+1, n_qubits)
    signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
    signals= MultiChannelSignal(signals_)
    energy,ϕ = costfunction_ode(ψ_initial, eigvalues, signals, n_qubits, drives,eigvectors,  T,Cost_ham,basis="qubitbasis",tol_ode=tol_ode)   
    return energy
end


samples_initial=reshape(samples_matrix, :)
Grad = zeros(Float64, n_samples+1, n_qubits)
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_initial, optimizer, options)
samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization) 
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)

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

savefig("final_signals_$(n_qubits)_$(n_levels)_$(SYSTEM)_n_samples$(n_samples)_n_grad_samples$(n_samples_grad)_$(T).pdf")