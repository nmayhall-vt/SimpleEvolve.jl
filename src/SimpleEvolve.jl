module SimpleEvolve
using SciMLSensitivity
using ReverseDiff
using ForwardDiff

include("signal.jl")
include("evolution.jl")
include("gradient.jl")
include("helpers.jl")
include("costfunction.jl")
include("signal_reconstruction.jl")
include("devices.jl")


export DigitizedSignal
export amplitude
export frequency
export infidelity
export evolve_direct_exponentiation
export evolve_ODE
export a_fullspace
export a_q
export costfunction_ode
export costfunction_direct_exponentiation
export gradientsignal_ODE
export gradientsignal_direct_exponentiation
export gradientsignal_ODE
export MultiChannelSignal
export single_trotter_exponentiation_step
export grad_signal_expansion
export gradientsignal_rotate
export trotter_evolve
export costfunction_trotter
export gradientsignal_rotate_ode
export gradientsignal_ODE_new
export gradientsignal_finite_difference

export static_hamiltonian
export Transmon
export QubitCoupling
export choose_qubits
export validate_and_expand
export projector
export transform!
export kron_concat
end