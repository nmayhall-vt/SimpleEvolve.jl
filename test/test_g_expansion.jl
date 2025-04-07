using SimpleEvolve
using Plots
using FFTW
using Random
Random.seed!(2)
T = 30
n_samples = 200
δt = T/n_samples
n_sites = 6
n_samples_grad = 20
freqs = [0.2,0.3,0.32,0.24,0.5,0.6]

# Original gradient signal reduced
grad_ode_reduced = [sin(2π*(t/n_samples_grad)) for t in 0:n_samples_grad] .* rand(n_samples_grad+1,n_sites)
# plot(grad_ode_reduced[:,1])


# Original gradient signal original
n_samples_grad_ =n_samples 
grad_ode = [sin(2π*(t/n_samples_grad_)) for t in 0:n_samples_grad_] .* rand(n_samples_grad_+1,n_sites)
plot(grad_ode[:,1], label="original signal", color=:black)

# display(grad_ode[:,1])
# Reconstruction using whittaker_shannon
δΩ_ = zeros(n_samples+1,n_sites)
validate_and_expand(δΩ_,grad_ode_reduced,
                    n_samples_grad,
                    n_samples,
                    n_sites, 
                    T, 
                    freqs,
                    :whittaker_shannon)
# grad_signal_expansion_ws(δΩ_, grad_ode_reduced, n_samples_grad, n_samples, rand(n_sites), δt, n_sites, T)
plot!(δΩ_[:,1], label="whittaker_shannon_fft", color=:green)
println("Maximum absolute value of the FFT using whittaker_shannon interpolation: ", maximum(abs.(fft(δΩ_[:,1]))[Int(3n_samples/4):end]))

# Reconstruction using trigonometric interpolation
δΩ = zeros(n_samples+1,n_sites)
validate_and_expand(δΩ,grad_ode_reduced,
                    n_samples_grad,
                    n_samples,
                    n_sites, 
                    T, 
                    freqs,
                    :whittaker_shannon)
# grad_signal_expansion_ti(δΩ, grad_ode_reduced, n_samples_grad, n_samples, rand(n_sites), δt, n_sites, T)
plot!(δΩ[:,1], label="trigonometric interpolation", color=:red)
println("Maximum absolute value of the FFT using trigonometric_interpolation: ", maximum(abs.(fft(δΩ[:,1]))[Int(3n_samples/4):end]))


# Reconstruction using polynomial interpolation
δΩ__ = zeros(n_samples+1,n_sites)
validate_and_expand(δΩ__,grad_ode_reduced,
                    n_samples_grad,
                    n_samples,
                    n_sites, 
                    T, 
                    freqs,
                    :polynomial)
# grad_signal_expansion_pi(δΩ__, grad_ode_reduced, n_samples_grad, n_samples, rand(n_sites), δt, n_sites, T)
plot!(δΩ__[:,1], label="polynomial interpolation", color=:blue)
println("Maximum absolute value of the FFT using polynomial_interpolation: ", maximum(abs.(fft(δΩ__[:,1]))[Int(3n_samples/4):end]))

# Reconstruction using linear interpolation
δΩ_l = zeros(n_samples+1,n_sites)
validate_and_expand(δΩ_l,grad_ode_reduced,
                    n_samples_grad,
                    n_samples,
                    n_sites, 
                    T, 
                    freqs,
                    :linear)
# grad_signal_expansion_linear(δΩ_l, grad_ode_reduced, n_samples_grad, n_samples, rand(n_sites), δt, n_sites, T)
plot!(δΩ_l[:,1], label="linear interpolation", color=:orange)
println("Maximum absolute value of the FFT using linear_interpolation: ", maximum(abs.(fft(δΩ_l[:,1]))[Int(3n_samples/4):end]))

# Reconstruction using hybrid interpolation linear+polynomial
δΩ_h = zeros(n_samples+1,n_sites)
validate_and_expand(δΩ_h,grad_ode_reduced,
                    n_samples_grad,
                    n_samples,
                    n_sites, 
                    T, 
                    freqs,
                    :hybrid)
# grad_signal_expansion_hybrid(δΩ_h, grad_ode_reduced, n_samples_grad, n_samples, rand(n_sites), δt, n_sites, T)
plot!(δΩ_h[:,1], label="hybrid interpolation", color=:purple)
println("Maximum absolute value of the FFT using hybrid_interpolation: ", maximum(abs.(fft(δΩ_h[:,1]))[Int(3n_samples/4):end]))

savefig("reconstructed_signal_$(n_samples)_$(n_samples_grad).pdf")

# Verify dimensions and anti-aliasing
@assert size(δΩ_) == (n_samples+1,n_sites) "Output dimension mismatch"
println("Maximum absolute value of the FFT: ", maximum(abs.(fft(δΩ_[:,1]))[Int(3n_samples/4):end]))
@assert maximum(abs.(fft(δΩ_[:,1])[Int(3n_samples/4):end])) < 1e-6 "Aliasing detected"
