module SimpleEvolve


include("signal.jl")
include("evolution.jl")
include("gradient.jl")
include("helpers.jl")
include("costfunction.jl")


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


end