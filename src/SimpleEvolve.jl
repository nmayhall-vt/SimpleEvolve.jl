module SimpleEvolve


struct DigitizedSignal{T}
    samples::Vector{T}
    δt::Float64
    carrier_freq::Float64
end

function amplitude(signal::DigitizedSignal, t)
    @show i = Int64(floor(t / signal.δt)) + 1
    
    if t > length(signal.samples)*signal.δt
        throw(DimensionMismatch)
    end
    return (signal.samples[i] + signal.samples[i+1]) / 2
end

export DigitizedSignal
export amplitude

end