using Statistics
abstract type AbstractSignalGenerator end

"""
    DigitizedSignal 
    A struct to represent a digitized signal with samples, time step, and carrier frequency.

"""


struct DigitizedSignal{T}<: AbstractSignalGenerator
    samples::Vector{T}
    δt::Float64
    carrier_freq::Float64
end
# function DigitizedSignal(samples::Vector{T}, δt::Float64, carrier_freq::Float64) where T
#     return DigitizedSignal{T}(samples, δt, carrier_freq)
# end
"""
    MultiChannelSignal 
    A struct to represent a multi-channel signal with multiple digitized signals.

"""
struct MultiChannelSignal
    channels::Vector{<:AbstractSignalGenerator}
end

function amplitude(signal::DigitizedSignal, t)
    
    i = Int64(floor(t / signal.δt)) + 1

    if t > length(signal.samples)*signal.δt
        throw(DimensionMismatch)
    end
    t_i = (i - 1) * signal.δt
    t == t_i && return signal.samples[i]
    
    # Calculate interpolation parameters
    α = (t - t_i) / signal.δt  # 0 ≤ α ≤ 1
    y0 = signal.samples[i]
    y1 = signal.samples[i+1]
    
    # Linear interpolation formula 
    return y0 + α * (y1 - y0)
    
end


function frequency(signal::DigitizedSignal, t)
    return signal.carrier_freq
end

function grad_signal_expansion(δΩ_,
                                grad_ode,
                                n_samples_grad,
                                n_samples,
                                frequency_multichannel,
                                δt,
                                n_sites,
                                T)
    for k in 1:n_sites
        grad_ode_k= grad_ode[:,k]
        grad_signal_k = DigitizedSignal(grad_ode_k, T/n_samples_grad, frequency_multichannel[k])
        δΩ_[:,k] = [amplitude(grad_signal_k, i*δt) for i in 0:n_samples]
    end
    return δΩ_
end


# function Base.copy(mcs::MultiChannelSignal{T}) where T
#     copied_channels = [
#         DigitizedSignal(
#             copy(ch.samples),  # Explicitly copy the samples vector
#             ch.δt,             # δt is Float64 (immutable, no need to copy)
#             ch.carrier_freq    # carrier_freq is Float64 (immutable)
#         ) 
#         for ch in mcs.channels
#     ]
#     return MultiChannelSignal(copied_channels)  
# end


"""
The idea is that a partial derivative ∂signal(t)/∂sample[j] 
as being non-zero only at times where t ≈ j * δt."""

function signalgradient_amplitude(t, j, signal::DigitizedSignal)
    i = j  # interpret j as a sample index now
    sample_time = (i - 1) * signal.δt
    
    if abs(t - sample_time) < signal.δt/2
        return 1.0  # The gradient at that point w.r.t sample j
    else
        return 0.0  # No contribution from sample j
    end
end


struct WindowedSquareWave{T<:Number}<: AbstractSignalGenerator
    frequency::Float64
    duty_cycle::Float64
    window_amplitudes::Vector{T}  
    window_durations::Vector{Float64}
    cumulative_times::Vector{Float64}
end

function WindowedSquareWave(frequency, duty_cycle, window_amplitudes, window_durations)
    cumulative_times = cumsum([0.0; window_durations])
    T = promote_type(eltype(window_amplitudes), Float64)
    WindowedSquareWave{T}(frequency, duty_cycle, window_amplitudes, window_durations, cumulative_times)
end

function amplitude(sw::WindowedSquareWave{T}, t) where T<:Number
    window_idx = searchsortedlast(sw.cumulative_times, t)
    amplitude = window_idx > length(sw.window_amplitudes) ? zero(T) : sw.window_amplitudes[window_idx]
    
    period = 1/sw.frequency
    phase = mod(t, period)
    return phase < sw.duty_cycle * period ? amplitude : zero(T)
end
function frequency(sw::WindowedSquareWave, t)
    return sw.frequency
end

"""
    WindowedGaussianPulse(amplitudes, centers, widths, frequencies)
    Create a windowed Gaussian pulse with specified amplitudes, centers, widths, and frequencies.
"""


struct WindowedGaussianPulse{T<:Number} <: AbstractSignalGenerator
    amplitudes::Vector{T}     # Complex or real amplitudes
    centers::Vector{Float64}  # Time centers (μ)
    widths::Vector{Float64}   # Standard deviations (σ)
    phases::Vector{Float64}   # Phase offsets (radians)
    frequencies::Vector{Float64} # Frequencies (Hz or rad/s)
end

function WindowedGaussianPulse(amplitudes, centers, widths, phases,frequencies)
    @assert length(centers) == length(widths) == length(phases)
    T = promote_type(eltype(amplitudes), Float64)
    WindowedGaussianPulse{T}(amplitudes, centers, widths, phases,frequencies)
end

function amplitude(pulse::WindowedGaussianPulse, t)
    s = zero(eltype(pulse.amplitudes))
    for i in eachindex(pulse.amplitudes)
        s += pulse.amplitudes[i] * 
             exp(-0.5 * ((t - pulse.centers[i]) / pulse.widths[i])^2) *
             exp(im * pulse.phases[i])  # Phase rotation without oscillation
    end
    return s
end

function frequency(pulse::WindowedGaussianPulse, t)
    return pulse.frequencies[1]
end

"""
    TanhEnvelope{T<:AbstractFloat}
Signal with hyperbolic tangent envelope for smooth transitions.

# Fields
- `amplitude::T`: Peak amplitude
- `sigma::T`: Rise/fall time constant
- `center::T`: Central time position
"""
struct TanhEnvelope{T<:Number, S<:AbstractFloat}<: AbstractSignalGenerator
    amplitude::T 
    sigma::S
    center::S
end

function value_at(sig::TanhEnvelope, t::Real) 
    return sig.amplitude * tanh((t - sig.center)/sig.sigma)
end

struct SinusoidalSignal{T<:Number} <: AbstractSignalGenerator
    amplitude::T
    frequency::Float64
    phase::Float64
end

function value_at(sig::SinusoidalSignal, t::Real,n_samples::Int64)
    return sig.amplitude * sin(2 * π *sig.frequency * t/n_samples + sig.phase)
end

function to_digitized(sig::SinusoidalSignal{T}, Δt::T, n_samples::Int) where T
    times = (0:n_samples-1) .* Δt
    samples = [value_at(sig, t,n_samples) for t in times]
    return DigitizedSignal(samples, Δt, sig.frequency)
end

function analyze_sinusoidalsignal(samples, δt)
    N = length(samples)
    fft_result = fft(samples)
    freq_bins = fftshift(fftfreq(N, δt))
    fft_mags = abs.(fftshift(fft_result))
    peak_idx = argmax(fft_mags)
    freq = abs(freq_bins[peak_idx])
    amplitude = (maximum(samples) - minimum(samples)) / 2
    return amplitude, freq
end


"""
    SignalSum{T<:AbstractFloat}
Algebraic sum of multiple signal components.
"""
struct SignalSUM <: AbstractSignalGenerator
    components::Vector{<:AbstractSignalGenerator}
end


function value_at(sig::SignalSUM, t::T) where T
    return sum(value_at(c, t) for c in sig.components)
end

"""
    gaussian_filter(signal::DigitizedSignal, bandwidth::Real)
    Apply a Gaussian filter to a digitized signal in the frequency domain.
    
    # Arguments
    - `signal`: The digitized signal to be filtered.
    - `bandwidth`: The bandwidth of the Gaussian filter.
    
    # Returns
    A new `DigitizedSignal` with the filtered samples.
"""

function gaussian_filter(signal::DigitizedSignal{T}, bandwidth::Real) where T
    # Fourier transform
    spectrum = fft(signal.samples)
    
    # Frequency axis setup
    N = length(spectrum)
    δf = 1/(N*signal.δt)  # Frequency resolution
    carrier_bin = round(Int, signal.carrier_freq/δf) + 1
    
    # Construct Gaussian window
    freq_axis = fftfreq(N, 1/signal.δt)
    gaussian = exp.(-(freq_axis .- signal.carrier_freq).^2 / (2*bandwidth^2))
    
    # Apply filter and inverse transform
    filtered_spectrum = spectrum .* fftshift(gaussian)
    filtered_signal = real(ifft(filtered_spectrum))
    
    # Preserve metadata
    return DigitizedSignal(filtered_signal, signal.δt, signal.carrier_freq)
end
