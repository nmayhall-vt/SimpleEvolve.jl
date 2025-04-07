using SimpleEvolve
using Plots
using Random
using LinearAlgebra
using DifferentialEquations
function test1()

    #
    #   H(t) = Hstatic + \sumₖ Ω(t)ₖ e^(iνₖt)Aₖ +  \sumₖ Ω(t)ₖ* e^(-iνₖt)Aₖ*  
    #   

    Random.seed!(2)

    n_sites = 2
    n_levels = 2
    T = 15

    dim = n_levels^n_sites
    Hstatic = rand(Float64, dim, dim) 
    Hstatic += Hstatic'
    C = rand(Float64, dim, dim) 
    C += C'

    println("Eignvalues of our static Hamiltonian")
    display(eigvals(Hstatic))

    drives = Vector{Matrix{Float64}}([])
    for k in 1:n_sites
        Ak = rand(Float64, dim, dim) 
        push!(drives, Ak)
    end
 
    n_samples = 10
    δt = T/n_samples
    freqs=1.0
    frequency_multichannel = [0.2,0.3 ]
    amps = [sin(2*π*(t/n_samples)) for t in 0:n_samples+1]
    signal = DigitizedSignal(amps, δt, freqs)

    signals_ = [DigitizedSignal([sin(2π*(t/n_samples)) for t in 0:n_samples+1], δt, f) for f in frequency_multichannel]
    signals = MultiChannelSignal(signals_)
    scatter([i*δt for i in 0:n_samples], amps, marker=:circle,markersize=10) 

    # trying to access in between values for the amplitude 
    # that we need for adapted time steps in ODE
    n_new_samples =n_samples*10
    δt_new = T/n_new_samples
    plot!([i*δt_new for i in 0:n_new_samples], [amplitude(signal, i*δt_new) for i in 0:n_new_samples], marker=:circle) 
    savefig("amps.pdf")

    # initial state
    initial_state = "1"^(n_sites÷2) * "0"^(n_sites÷2)
    ψ_initial = zeros(ComplexF64, dim)                              
    ψ_initial[1 + parse(Int, initial_state, base=n_levels)] = one(ComplexF64) 
    #eigenvalues and eigenvectors of the static Hamiltonian
    eigvalues, eigvecs = eigen(Hstatic)
    
    for i in 1:n_sites
        drives[i] = eigvecs' * drives[i] * eigvecs
    end
    #ode evolution
    @time energy,ϕ = costfunction_ode(ψ_initial, eigvalues, signals, n_sites, drives, T,C,tol_ode=1e-8)   
    println("ode evolved energy is ",energy)

    # trotter direct exponentiation evolution
    n_trotter_steps = 1000 
    @time energy2,ψ_d = costfunction_direct_exponentiation(ψ_initial, eigvalues, signals, n_sites, drives, T, n_trotter_steps,C)
    println("direct evolved energy is ",energy2)

    
    display(infidelity(ϕ,ψ_d))

end

test1()