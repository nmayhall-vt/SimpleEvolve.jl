
struct DigitizedSignal{T}
    samples::Vector{T}
    δt::Float64
    carrier_freq::Float64
end
# function DigitizedSignal(samples::Vector{T}, δt::Float64, carrier_freq::Float64) where T
#     return DigitizedSignal{T}(samples, δt, carrier_freq)
# end

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