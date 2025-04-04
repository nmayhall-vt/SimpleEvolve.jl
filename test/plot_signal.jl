using SimpleEvolve
using LinearAlgebra
using Plots
using Random
function plot_gradient_Signal()
    #plot digitized signal and gradient
    T=10
    n_samples = 1000
    δt = T/n_samples
    Random.seed!(2)
    frequency_multichannel = [0.21,0.32]
    signals_ = [DigitizedSignal([sin(2π*(t/n_samples)) for t in 0:n_samples+1], δt, f) for f in frequency_multichannel]
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
    # gradient calculation
    n_samples_grad = 1000
    τ = T/n_samples_grad
    device_action_independent_t = exp.((-im*τ).*eigvalues)
    ∂Ω = Matrix{Float64}(undef, n_samples_grad+1, n_sites)
    @time grad_ode =gradientsignal_ODE(ψ_initial,
                            T,
                            signals,
                            n_sites,
                            drives,
                            eigvalues,
                            device_action_independent_t,
                            C,
                            n_samples_grad,
                            ∂Ω)
                            
    println("gradient from ODE is ")
    display(grad_ode)
    display(norm(grad_ode))
    n_trotter_steps = 1000
    dΩ = Matrix{Float64}(undef, n_samples+1, n_sites)
    @time grad_direct =gradientsignal_direct_exponentiation(ψ_initial,
                                            T,
                                            signals,
                                            n_sites,
                                            drives,
                                            eigvalues,
                                            device_action_independent_t,
                                            n_trotter_steps,
                                            C,
                                            n_samples,
                                            dΩ )

    println("gradient from direct exponentiation is ")
    display(grad_direct)
    display(norm(grad_direct))
    

    pulse_windows=range(0, T, length=n_samples+1)
    # display(amps)
    # scatter([i*δt for i in 0:n_samples], amps, marker=:circle,markersize=10)

    Ω0=Matrix{Float64}(undef, n_samples+1, n_sites)
    for k in 1:n_sites
        Ω0[:,k] = [amplitude(signals.channels[k], i*δt) for i in 0:n_samples]
    end
    δΩ_ = Matrix{Float64}(undef, n_samples+1, n_sites)
    
    grad_updated_signals = grad_signal_expansion(δΩ_,
                                    grad_ode,
                                    n_samples_grad,
                                    n_samples,
                                    frequency_multichannel,
                                    δt,
                                    n_sites,
                                    T)
    println("gradient from ODE is ")
    display(grad_updated_signals)
    display(norm(grad_updated_signals))

    Ω = copy(Ω0)
    Ω_plots = plot(                        # GRADIENT SIGNAL PLOT
        [plot(
            pulse_windows, Ω[:,q]
        ) for q in 1:n_sites]...,
        title = "Initial Signals",
        legend = false,
        layout = (n_sites,1),
    )
    ∇Ω0 = copy(grad_updated_signals)
    ∇Ω_plots = plot([plot(pulse_windows, ∇Ω0[:,q]) for q in 1:n_sites]...,
                    title = "ODE GS",legend = false,layout = (n_sites,1),)

    ∇Ω1 = copy(grad_direct)
    ∇Ω_plots1 = plot([plot(pulse_windows, ∇Ω1[:,q]) for q in 1:n_sites]...,
                    title = "Trotter direct",legend = false,layout = (n_sites,1),)
    plot(Ω_plots, ∇Ω_plots,∇Ω_plots1, layout=(1,3))
    # plot(grad_ode[:,1], [amplitude(signal, i*δt) for i in 0:n_samples], marker=:circle)
    savefig("amps_grad.pdf")

end
plot_gradient_Signal()