using SimpleEvolve
using NPZ
using JLD2
using Plots
using LinearAlgebra
using Optim
using LineSearches
using Random

for dist in [0.75,0.8,0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.95, 2.1, 2.25, 2.4, 2.55,2.7, 2.85, 3.0] 
    Cost_ham = npzread("./qubit_op_N2_tapered_$(dist).npy")
    # display(Cost_ham)
    n_qubits = round(Int, log2(size(Cost_ham,1)))
    n_levels = 2
    SYSTEM="N2_tapered"
    freqs = 2π*collect(4.8 .+ (0.02 * (1:n_qubits)))
    anharmonicities = 2π*0.3 * ones(n_qubits)
    coupling_map = Dict{QubitCoupling,Float64}()
    for p in 1:n_qubits
        q = (p == n_qubits) ? 1 : p + 1
        coupling_map[QubitCoupling(p,q)] = 2π*0.02
    end
    device = Transmon(freqs, anharmonicities, coupling_map, n_qubits)
    n_samples=100
    T=40.0
    δt = T/n_samples
    t_=collect(0:δt:T)


    samples_matrix=[sin(2π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits]
    pulse_windows=range(0, T, length=n_samples+1)
    samples_initial=reshape(samples_matrix, :)
    carrier_freqs = freqs.-2π*0.1 
    signals_ = [DigitizedSignal(samples_matrix[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
    signals = MultiChannelSignal(signals_)

    initial_state = "00000111"
    ψ_initial = zeros(ComplexF64, n_levels^n_qubits)  
    ψ_initial[1 + parse(Int, initial_state, base=n_levels)] = one(ComplexF64) 

    # display(ψ_initial)
    H_static = static_hamiltonian(device, n_levels)
    #eigenvalues and eigenvectors of the static Hamiltonian
    drives =a_fullspace(n_qubits, n_levels)
    eigvalues, eigvectors = eigen(Hermitian(H_static))  # Ensures real eigenvalues
    println("Eignvalues of our static Hamiltonian")
    display(eigvalues)

    tol_ode=1e-4
    Λ, U = eigen(Cost_ham)
    E_actual = Λ[1]
    println("Actual energy: $E_actual") 
    # display(drives[1])
    display(eigvectors)
    for i in 1:n_qubits
        drives[i] = eigvectors' * drives[i] * eigvectors
    end

    # we have to optimize the samples in the signal
    n_samples_grad = n_samples
    δΩ_ = Matrix{Float64}(undef, n_samples+1, n_qubits)
    ∂Ω0 = Matrix{Float64}(undef, n_samples_grad+1, n_qubits)
    τ = T/n_samples_grad
    a=a_q(n_levels)




    function gradient_ode_opt_penalty!(Grad, samples; λ=1.0, Ω₀=5.0+(2π*0.02))
        Grad = reshape(Grad, :, n_qubits)
        samples = reshape(samples, n_samples+1, n_qubits)

        # Step 1: Reconstruct MultiChannelSignal
        signals_ = [DigitizedSignal(samples[:, i], δt, carrier_freqs[i]) for i in 1:n_qubits]
        signals = MultiChannelSignal(signals_)

        # Step 2: Compute fidelity gradient
        grad_ode, ψ_ode, σ_ode = gradientsignal_ODE(ψ_initial,
                                T,
                                signals,
                                n_qubits,
                                drives,
                                eigvalues,
                                eigvectors,
                                Cost_ham,
                                n_samples_grad,
                                ∂Ω0;
                                basis="qubitbasis",
                                tol_ode=tol_ode)
        # Step 3: Compute penalty gradient and add to total gradient
        # for k in 1:n_qubits
        #     for i in 1:n_samples+1
        #         grad_fidelity = grad_ode[i, k]
        #         x = samples[i, k] / Ω₀
        #         y = abs(x) - 1

        #         grad_penalty = 0.0
        #         if y > 0
        #             h = exp(y - 1 / y)
        #             dh_dx = h * (1 + 1 / y^2) / Ω₀
        #             grad_penalty = sign(x) * dh_dx
        #         end

        #         Grad[i, k] = grad_fidelity + λ * grad_penalty
        #     end
        # end
        for k in 1:n_qubits
            for i in 1:n_samples+1
                Grad[i, k] =grad_ode[i, k]
            end
        end
        return Grad
    end

    function costfunction_ode_opt(samples)
        # considering the frequency remain as constants
        # samples is a vector of amplitudes as a function of time
        # vector is considered to optimize using BFGS
        samples = reshape(samples, n_samples+1, n_qubits)
        signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
        signals= MultiChannelSignal(signals_)
        energy,ϕ = costfunction_ode(ψ_initial, eigvalues, signals, n_qubits, drives,eigvectors,  T,Cost_ham,basis="qubitbasis",tol_ode=tol_ode)   
        # energy,ϕ = costfunction_ode_with_penalty(ψ_initial, eigvalues, signals, n_qubits, drives,eigvectors,  T,Cost_ham,basis="qubitbasis",tol_ode=tol_ode, λ=1.0,Ω₀=5.0+(2π*0.02))   
        return energy
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
            iterations = 1000,
    )

    if dist==0.75
        samples_initial=reshape(samples_matrix, :)
    else
        samples_initial=samples_final
    end
    Grad = zeros(Float64, n_samples+1, n_qubits)
    optimization = Optim.optimize(costfunction_ode_opt, gradient_ode_opt_penalty!, samples_initial, optimizer, options)
    samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
    tol_ode = 1e-6
    optimization = Optim.optimize(costfunction_ode_opt, gradient_ode_opt_penalty!, samples_final, optimizer, options)
    samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
    tol_ode = 1e-8
    optimization = Optim.optimize(costfunction_ode_opt, gradient_ode_opt_penalty!, samples_final, optimizer, options)
    samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS


    samples_final_reshaped = reshape(samples_final, n_samples+1, n_qubits)
    final_energy= costfunction_ode_opt(samples_final)
    # Save the optimized samples
    @save "N2_tapered_samples_optimized_$(dist).jld2" samples_final_reshaped final_energy
end