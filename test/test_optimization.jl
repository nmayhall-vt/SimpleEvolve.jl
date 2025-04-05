using SimpleEvolve
using LinearAlgebra
using Plots
using Random
using Optim
using LineSearches

# function optimize()

    T=30
    n_samples = 300
    δt = T/n_samples
    Random.seed!(2)
    freqs = [0.2,0.22]#2,2
    freqs = [0.1,0.24,0.6]#2,3
    # freqs = [0.6,0.15,0.3]#3,3
    # freqs = [0.4,0.8,0.1,0.1 ]
    freqs = [0.8,0.4,0.08,0.23]
    # freqs = [0.5,0.6,0.3,0.03,0.65]
    # freqs = [0.1,0.55,0.5,0.06,0.25,0.3]
    # freqs = [0.08,0.3,0.3,0.45,0.3,0.27,0.28]
    signals_ = [DigitizedSignal([sin(2π*(t/n_samples)) for t in 0:n_samples+1], δt, f) for f in freqs]
    signals = MultiChannelSignal(signals_)

    n_sites = 4
    n_levels = 3

    dim = n_levels^n_sites
    Hstatic = rand(Float64, dim, dim) 
    Hstatic += Hstatic'
    C = rand(Float64, dim, dim) 
    C += C'
    drives = Vector{Matrix{Float64}}([])
    for k in 1:n_sites
        Ak = rand(Float64, dim, dim) 
        push!(drives, Ak)
    end
    println("Eignvalues of our static Hamiltonian")
    display(eigvals(Hstatic))

    # initial state
    initial_state = "1"^(n_sites÷2) * "0"^(n_sites÷2)
    ψ_initial = zeros(ComplexF64, dim)                              
    ψ_initial[1 + parse(Int, initial_state, base=n_levels)] = one(ComplexF64) 
    #eigenvalues and eigenvectors of the static Hamiltonian
    eigvalues, eigvecs = eigen(Hstatic)
    for i in 1:n_sites
        drives[i] = eigvecs' * drives[i] * eigvecs
    end
    
    function costfunction(samples)
        # considering the frequency remain as constants
        # samples is a vector of amplitudes as a function of time
        # vector is considered to optimize using BFGS
        samples = reshape(samples, n_samples+1, n_sites)
        signals_= [DigitizedSignal(samples[:,i], δt, freqs[i]) for i in 1:length(freqs)]
        signals= MultiChannelSignal(signals_)
        energy,ϕ = costfunction_ode(ψ_initial, eigvalues, signals, n_sites, drives, T,C,tol_ode=1e-8)   
        return energy
    end
    Λ, U = eigen(C)
    E_actual = Λ[1]
    println("Actual energy: $E_actual") 

    # we have to optimize the samples in the signal
    n_samples_grad = 150
    δΩ_ = Matrix{Float64}(undef, n_samples+1, n_sites)
    ∂Ω0 = Matrix{Float64}(undef, n_samples_grad+1, n_sites)
    τ = T/n_samples_grad
    device_action_independent_t = exp.((-im*τ).*eigvalues)
    function gradient_ode!(Grad, samples)
        Grad = reshape(Grad, :, n_sites)
        samples = reshape(samples, n_samples+1, n_sites)
        signals_= [DigitizedSignal(samples[:,i], δt, freqs[i]) for i in 1:length(freqs)]
        signals= MultiChannelSignal(signals_)
        grad_ode =gradientsignal_ODE(ψ_initial,
                            T,
                            signals,
                            n_sites,
                            drives,
                            eigvalues,
                            device_action_independent_t,
                            C,
                            n_samples_grad,
                            ∂Ω0,
                            tol_ode=1e-8)
        grad_ode_expanded = grad_signal_expansion(δΩ_,
                            grad_ode,
                            n_samples_grad,
                            n_samples,
                            freqs,
                            δt,
                            n_sites,
                            T)
        # display(grad_ode_expanded)
        for k in 1:n_sites
            for i in 1:n_samples+1
                # considering the frequency remain as constants
                Grad[i,k] = grad_ode_expanded[i,k] 
            end
        end
        
        # display(Grad)
        return Grad
    end


    # OPTIMIZATION ALGORITHM
    linesearch = LineSearches.MoreThuente()
    optimizer = Optim.LBFGS(linesearch=linesearch)
    # OPTIMIZATION OPTIONS
    options = Optim.Options(
        show_trace = true,
        show_every = 1,
        f_reltol = 1e-9,
        g_tol = 1e-9,
        iterations = 100,
    )
    # INITIAL PARAMETERS
    samples_matrix = [sin(2π * (t / n_samples)) for t in 0:n_samples, i in 1:n_sites]
    samples_initial=reshape(samples_matrix, :)
    Grad = zeros(Float64, n_samples+1, n_sites)

    optimization = Optim.optimize(costfunction, gradient_ode!, samples_initial, optimizer, options)
    samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
    optimization = Optim.optimize(costfunction, gradient_ode!, samples_final, optimizer, options)
    samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
    optimization = Optim.optimize(costfunction, gradient_ode!, samples_final, optimizer, options)
    samples_final = Optim.minimizer(optimization) 
    samples_final_reshaped = reshape(samples_final, n_samples+1, n_sites)
    
    Λ, U = eigen(C)
    E_actual = Λ[1]
    println("Actual energy: $E_actual")

    Ω0=copy(samples_matrix)
    Ω = copy(samples_final_reshaped)
    plot(Ω0[:,1], label="initial signal ", color=:black, lw=2, linestyle=:dash)
    for i in 1:n_sites
        plot!(Ω[:,i], label="final signal $i", lw=2)
    end

    savefig("final_signals_$(n_sites)_$(n_levels).pdf")
# end
# optimize()