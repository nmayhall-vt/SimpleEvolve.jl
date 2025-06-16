using SimpleEvolve
using NPZ
using Plots
using LinearAlgebra
using Optim
using LineSearches
using Random
using Printf
using JLD2

distances = 0.6:0.05:3.00
#[0.6,0.64,0.65,0.66,0.68]

for dist in [0.6,0.64,0.65,0.66,0.68]#distances
    dist_id = @sprintf("%.2f", dist)
    SYSTEM = "H2_$dist_id"
    file_name = "H2_sto3g_$dist_id.npy"
    println("Processing $file_name")
    Cost_ham = npzread(file_name) 
    
    # display(Cost_ham)
    n_qubits = round(Int, log2(size(Cost_ham,1)))
    n_levels = 2
    # Π = projector(n_qubits, 2, n_levels)    
    # Cost_ham = Hermitian(Π'*Cost_ham*Π)
    SYSTEM="h215"
    freqs = 2π*collect(4.8 .+ (0.02 * (1:n_qubits)))
    anharmonicities = 2π*0.3 * ones(n_qubits)
    coupling_map = Dict{QubitCoupling,Float64}()
    for p in 1:n_qubits
        q = (p == n_qubits) ? 1 : p + 1
        coupling_map[QubitCoupling(p,q)] = 2π*0.02
    end
    device = Transmon(freqs, anharmonicities, coupling_map, n_qubits)
    
    
    T=20.0
    n_samples=200
    δt = T/n_samples
    t_=collect(0:δt:T)
    
    # INITIAL PARAMETERS
    # samples_matrix=[sin(2π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits] 
    samples_matrix=[2π*0.02*sin(2π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits] 
    pulse_windows=range(0, T, length=n_samples+1)
    
    samples_initial=reshape(samples_matrix, :)
    carrier_freqs = freqs.-2π*0.1
    # carrier_freqs = freqs
    signals_ = [DigitizedSignal(samples_matrix[:,i], δt, carrier_freqs[i]) for i in 1:n_qubits]
    signals = MultiChannelSignal(signals_)
    
    
    # initial state
    ψ_initial = begin
        ψs = []
        for pattern in ["01", "11", "10","00"]#["01", "10"]
            state_str = repeat(pattern, n_qubits ÷ 2)
            ψ = zeros(ComplexF64, n_levels^n_qubits)
            ψ[1 + parse(Int, state_str, base=n_levels)] = one(ComplexF64)
            push!(ψs, ψ)
        end
        hcat(ψs...)
    end
    H_static = static_hamiltonian(device, n_levels)
    #eigenvalues and eigenvectors of the static Hamiltonian
    drives =a_fullspace(n_qubits, n_levels)
    eigvalues, eigvectors = eigen(Hermitian(H_static))  # Ensures real eigenvalues
    println("Eignvalues of our static Hamiltonian")
    # display(eigvalues)
    
    tol_ode=1e-8
    Λ, U = eigen(Cost_ham)
    E_actual = Λ[1]+Λ[2]+Λ[3]+Λ[4]
    println("Actual energy: $E_actual") 
    # display(drives[1])
    # display(eigvectors)
    for i in 1:n_qubits
        drives[i] = eigvectors' * drives[i] * eigvectors
    end
    
    @time energy1,ϕ,energies = SimpleEvolve.costfunction_ode_ssvqe(ψ_initial, eigvalues, signals, n_qubits, drives,eigvectors, T,Cost_ham;basis="qubitbasis",tol_ode=1e-10)   
    println("ode evolved energy is ",energy1)
    n_states= size(ψ_initial, 2)
    weights = reverse(collect(1.0:-0.4:1.0-0.4*(n_states-1)))
    println("weights",weights)
    # weights= [-0.2, 0.0, 0.4, 1.0]#1st and 4th state
    # weights=[0.05,1.0]#1st and 2nd state
    weights=[-0.2,0.00,0.02,1.0]
    weighted=true
    function costfunction_o(samples)
        samples = reshape(samples, n_samples+1, n_qubits)
        signals_ = [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:n_qubits]
        signals = MultiChannelSignal(signals_)
        energy, Ψ_ode, energies = SimpleEvolve.costfunction_ode_ssvqe(
            ψ_initial, eigvalues, signals, n_qubits, drives, eigvectors, T, Cost_ham;
            basis="qubitbasis", tol_ode=tol_ode,weights=weights, weighted=weighted
        )
        return energy
    end
    
    
    # Initial Hf LEVEL energy
    samples= zeros(Float64, n_samples+1, n_qubits)
    samples=reshape(samples, :)
    energy_initial = costfunction_o(samples)
    println("Initial hf LEVEL energy: $energy_initial")
    
    
    # we have to optimize the samples in the signal
    n_samples_grad = n_samples
    δΩ_ = Matrix{Float64}(undef, n_samples+1, n_qubits)
    ∂Ω0 = Matrix{Float64}(undef, n_samples_grad+1, n_qubits)
    τ = T/n_samples_grad
    a=a_q(n_levels)
    
    function gradient_ode!(Grad, samples)
        Grad = reshape(Grad, :, n_qubits)
        Grad .= 0.0
        samples = reshape(samples, n_samples+1, n_qubits)
        signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
        signals= MultiChannelSignal(signals_)
        grad_ode,ψ_ode, σ_ode =SimpleEvolve.gradientsignal_ODE_real_multiple_states(ψ_initial,
                                T,
                                signals,
                                n_qubits,
                                drives,
                                eigvalues,
                                eigvectors,
                                Cost_ham,
                                n_samples_grad;
                                basis="qubitbasis",
                                tol_ode=tol_ode)
        dim, n_states = size(ψ_ode)
        weights_states = weights
    
        for k in 1:n_qubits
            for i in 1:n_samples+1
                if weighted==true
                    Grad[i, k] = sum(weights_states[j] * grad_ode[i, k, j] for j in 1:n_states)
                else
                    Grad[i, k] = sum(grad_ode[i, k, j] for j in 1:n_states)
                end
            end
            
        end
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
            f_reltol   = 1e-14,
            g_tol      = 1e-9,
            iterations = 200,
    )
    
    
    
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
    optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
    samples_final = Optim.minimizer(optimization)
    optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
    samples_final = Optim.minimizer(optimization)
    samples_final_reshaped = reshape(samples_final, n_samples+1, n_qubits)
    #post proceesing
    samples=samples_final
    samples = reshape(samples, n_samples+1, n_qubits)
    signals_ = [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:n_qubits]
    signals = MultiChannelSignal(signals_)
    energy, Ψ_ode, energies = SimpleEvolve.costfunction_ode_ssvqe(
            ψ_initial, eigvalues, signals, n_qubits, drives, eigvectors, T, Cost_ham;
            basis="qubitbasis", tol_ode=tol_ode,weights=weights, weighted=weighted)
    a = zeros(Float64, n_states, n_states)
    for i in 1:n_states
        for j in i:n_states
            a[i, j] = real(Ψ_ode[:,i]' * Cost_ham * Ψ_ode[:,j])
            a[j, i] = a[i, j]  # Ensure symmetry
        end
    end
    A=eigen(a)
    energies=A.values
    
    B=[Λ[1]
        Λ[2]
        Λ[3]
        Λ[4]]
    fci_energies=B
    errors=A.values[1:n_states].-B[1:n_states]
    display(errors)
    @save "final_results_H2_$(dist).jld2" energies fci_energies errors samples_final weights
end