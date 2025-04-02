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
    T = 20

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
    

    drives_fullspace = a_fullspace(n_sites,n_levels,eigvecs)
    display(drives_fullspace[1])
    display(drives_fullspace[2])
    
    drive_q_dbasis=a_q(n_levels)
    display(drive_q_dbasis)
    n_samples = 1000
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


    initial_state = "1"^(n_sites÷2) * "0"^(n_sites÷2)
    ψ_initial = zeros(ComplexF64, dim)                              
    ψ_initial[1 + parse(Int, initial_state, base=n_levels)] = one(ComplexF64) 

    
    @time energy,ϕ = costfunction_ode(ψ_initial, Hstatic, signal, n_sites, drives_fullspace, T)   
    println("ode evolved energy is ",energy)
    
    @time energy2,ψ_d = costfunction_direct_exponentiation(ψ_initial, Hstatic, signal, n_sites, drives_fullspace, T, δt, n_samples)
    println("direct evolved energy is ",energy2)

    
    @time energy3,ψ_trot = costfunction_trotter(ψ_initial, Hstatic, signal, n_sites, n_levels, drive_q_dbasis, T, δt, n_samples)
    println("trotter evolved energy is ",energy3)
    display(infidelity(ϕ,ψ_d))
    display(infidelity(ϕ,ψ_trot))
end

test1()