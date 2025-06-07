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
