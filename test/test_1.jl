using SimpleEvolve
using Plots
using Random
using LinearAlgebra

function test1()

    #
    #   H(t) = Hstatic + \sumₖ Ω(t)ₖ e^(iνₖt)Aₖ +  \sumₖ Ω(t)ₖ* e^(-iνₖt)Aₖ*  
    #   

    Random.seed!(2)

    n_sites = 2
    n_levels = 2
    T = 10

    dim = n_levels^n_sites
    Hstatic = rand(Float64, dim, dim) 
    Hstatic += Hstatic'
    C = rand(Float64, dim, dim) 
    C += C'

    println(" Eignvalues of our static Hamiltonian")
    display(eigvals(Hstatic))

    drives = Vector{Matrix{Float64}}([])
    for k in 1:n_sites
        Ak = rand(Float64, dim, dim) 
        push!(drives, Ak)
    end

    n_samples = 10
    δt = T/n_samples
    amps = [sin(2*π*(t/n_samples)) for t in 0:n_samples+1]
    signal = DigitizedSignal(amps, δt, .2)

    # display(amps)
    scatter([i*δt for i in 0:n_samples], amps, marker=:circle,markersize=10) 

    # trying to access in between values for the amplitude 
    # that we need for adapted time steps in ODE
    n_new_samples =n_samples*10
    δt_new = T/n_new_samples
    plot!([i*δt_new for i in 0:n_new_samples], [amplitude(signal, i*δt_new) for i in 0:n_new_samples], marker=:circle) 
    savefig("amps.pdf")


    function dψdt(ψ, t)
    end

end

test1()