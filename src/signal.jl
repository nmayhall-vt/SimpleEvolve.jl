"""
    DigitizedSignal 
    A struct to represent a digitized signal with samples, time step, and carrier frequency.

"""


struct DigitizedSignal{T}
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
struct MultiChannelSignal{T}
    channels::Vector{DigitizedSignal{T}}
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


function Base.copy(mcs::MultiChannelSignal{T}) where T
    # Create copies of each DigitizedSignal channel with copied samples
    copied_channels = [
        DigitizedSignal(
            copy(ch.samples),  # Copies the samples vector to prevent mutation side effects
            ch.δt,             # Float64 is immutable; no copy needed
            ch.carrier_freq    # Float64 is immutable; no copy needed
        ) 
        for ch in mcs.channels
    ]
    return MultiChannelSignal{T}(copied_channels)
end