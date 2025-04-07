module SimpleEvolve


include("signal.jl")
include("evolution.jl")
include("gradient.jl")
include("helpers.jl")
include("costfunction.jl")
include("signal_reconstruction.jl")


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


export grad_signal_expansion_ti
export grad_signal_expansion_pi
export grad_signal_expansion_linear
export grad_signal_expansion_hybrid
export grad_signal_expansion_ws
export grad_signal_expansion_hybrid
export validate_and_expand

end