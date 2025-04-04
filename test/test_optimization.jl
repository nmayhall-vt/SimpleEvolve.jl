using SimpleEvolve
using LinearAlgebra
using Plots
using Random
using Optim
using LineSearches

function optimize()

    T=20
    n_samples = 100
    δt = T/n_samples
    Random.seed!(2)
    freqs = [0.2,0.21 ]
    signals_ = [DigitizedSignal([sin(2π*(t/n_samples)) for t in 0:n_samples+1], δt, f) for f in freqs]
    signals = MultiChannelSignal(signals_)

    n_sites = 2
    n_levels = 2

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
        energy,ϕ = costfunction_ode(ψ_initial, eigvalues, signals, n_sites, drives, T,C)   
        return energy
    end
     
    # we have to optimize the samples in the signal
    
    n_samples_grad = 100
    δΩ_ = Matrix{Float64}(undef, n_samples+1, n_sites)
    ∂Ω = Matrix{Float64}(undef, n_samples_grad+1, n_sites)
    
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
                            C,
                            n_samples_grad,
                            ∂Ω)
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
    optimization = Optim.optimize(costfunction, gradient_ode!, samples_final, optimizer, options)
    samples_final = Optim.minimizer(optimization)
    optimization = Optim.optimize(costfunction, gradient_ode!, samples_final, optimizer, options)
    samples_final = Optim.minimizer(optimization)
    optimization = Optim.optimize(costfunction, gradient_ode!, samples_final, optimizer, options)
    samples_final = Optim.minimizer(optimization)
    Λ, U = eigen(C)
    E_actual = Λ[1]
    println("Actual energy: $E_actual")
end
optimize()