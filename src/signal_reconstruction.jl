using DSP, SpecialFunctions
using Polynomials
using SimpleEvolve
using FFTW

# Whittaker-Shannon interpolation with anti-aliasing
"""
amplitude_ws(signal::DigitizedSignal, t; window_radius=8)
    Calculate the amplitude of a signal at time t using Whittaker-Shannon interpolation
    from the signal samples of reduced size
    Args:
        signal: Input digitized signal
        t: Time at which to evaluate the signal
        window_radius: Number of samples on each side of the center sample (default is 8)
    Returns:
        Amplitude of the signal at time t


"""
function amplitude_ws(signal::DigitizedSignal, t; window_radius=8)
    """
    Whittaker-Shannon interpolation with Hann-windowed sinc kernel
    Implements:
        f(t) = Σ_{n=-R}^{R} [x[n] ⋅ sinc((t - nΔt)/Δt) ⋅ 0.5(1 - cos(2π⋅(τ/(2RΔt) + 0.5)))]
    where R = window_radius, Δt = sampling interval
    """
    # Validate time bounds
    max_time = (length(signal.samples)-1)*signal.δt
    t > max_time && throw(DomainError(t, "Time exceeds signal duration"))
    
    # Calculate center index and window bounds
    center_idx = floor(Int, t/signal.δt) + 1  # n_center = ⌊t/Δt⌋ + 1
    start_idx = max(1, center_idx - window_radius)
    end_idx = min(length(signal.samples), center_idx + window_radius)
    
    sum_val = 0.0
    for n in start_idx:end_idx
        # Time difference from nth sample: τ = t - (n-1)Δt
        τ = t - (n-1)*signal.δt
        
        # Sinc kernel: sinc(τ/Δt) = sin(πτ/Δt)/(πτ/Δt)
        sinc_val = sinc(τ/signal.δt)
        
        # Hann window function:
        # w(τ) = 0.5(1 - cos(2π⋅(τ/(2RΔt) + 0.5))) for -RΔt ≤ τ ≤ RΔt
        x = (τ/(window_radius*signal.δt) + 1)/2  # Normalize τ to [0,1]
        window_val = 0.5 * (1 - cos(2π * x))
        
        # Accumulate weighted sample contribution
        sum_val += signal.samples[n] * sinc_val * window_val
    end
    return sum_val
end

"""
downsample(signal::Vector{T}, factor::Int)
    Downsample a signal by a given factor using anti-aliasing
    Args:
        signal: Input signal to downsample
        factor: Downsampling factor (must be an integer)
    Returns:
        Downsampled signal

"""

# Downsampling with anti-aliasing
function downsample(signal::Vector{T}, factor::Int) where T<:Real
    return decimate(signal, factor, fir=true)
end
"""
reconstruct gradient_ws(signal::DigitizedSignal, output_samples::Int)
    Anti-aliased N× upsampling from (n/N) to n samples
    Args:
        signal: Input digitized signal
        output_samples: Desired output length
    Returns:
        Reconstructed signal with anti-aliasing

"""

function reconstruct_gradient_ws(signal::DigitizedSignal, output_samples::Int)

    # Get original signal parameters
    original_samples = length(signal.samples)
    δt_original = signal.δt
    new_δt = δt_original * (original_samples/output_samples)
    
    # FFT-based interpolation
    upsampled = whittaker_shannon_fft(signal.samples, output_samples)
    
    # Anti-aliasing filter parameters
    original_nyquist = 1/(2δt_original)
    new_fs = 1/new_δt
    cutoff = original_nyquist  # Critical cutoff frequency
    
    # Apply FIR lowpass filter
    filtered = lowpass(upsampled, cutoff, new_fs)
    
    return filtered
end
"""
whittaker_shannon_fft(x::Vector{T}, output_length::Int)
    Direct FFT interpolation without oversampling
    Args:
        x: Input signal
        output_length: Desired output length
    Returns:
        Interpolated signal in the frequency domain


"""


function whittaker_shannon_fft(x::Vector{T}, output_length::Int) where T<:Real
    """
    Direct FFT interpolation without oversampling
    """
    X = rfft(x)
    freq_bins = length(X)
    
    # Calculate new frequency bin count
    new_bins = output_length ÷ 2 + 1
    
    # Frequency domain padding
    X_padded = zeros(eltype(X), new_bins)
    copyto!(X_padded, 1, X, 1, min(freq_bins, new_bins))
    
    # Inverse transform with proper scaling
    x_interp = irfft(X_padded, output_length) * (output_length/length(x))
    return real(x_interp)
end

"""
lowpasss(signal::Vector{T}, cutoff::Real, fs::Real)
    Lowpass filter a signal using a FIR filter
    Args:
        signal: Input signal
        cutoff: Cutoff frequency
        fs: Sampling frequency
    Returns:
        Filtered signal


"""

function lowpass(signal::Vector{T}, cutoff::Real, fs::Real) where T<:Real
    nyquist = fs/2
    normalized_cutoff = cutoff/nyquist
    fir = digitalfilter(Lowpass(normalized_cutoff), FIRWindow(hanning(51)))
    return filtfilt(fir, signal)  # Zero-phase filtering
end




function trigonometric_interpolation(signal::DigitizedSignal, output_samples::Int)
    """
    Trigonometric interpolation using Fourier series expansion
    Args:
        signal: Input digitized signal
        output_samples: Desired output length
    Returns:
        Reconstructed signal with anti-aliasing
    """
    x = signal.samples
    N = length(x)
    M = output_samples
    
    # Compute Fourier coefficients
    X = fft(x)
    
    # Anti-aliasing: Remove frequencies above new Nyquist
    new_nyquist_bin = floor(Int, M/(2N))
    X_filtered = [k ≤ new_nyquist_bin+1 ? X[k] : 0.0 for k in 1:length(X)]
    
    # Frequency domain padding/truncation
    X_padded = zeros(ComplexF64, M)
    copyto!(X_padded, 1, X_filtered, 1, min(length(X_filtered), M))
    
    # Inverse transform with proper scaling
    x_interp = ifft(X_padded) * (M/N)
    return real(x_interp)
end

"""
amplitude_ti(signal::DigitizedSignal, t)
    Calculate the amplitude of a signal at time t using trigonometric interpolation
    Args:
        signal: Input digitized signal
        t: Time at which to evaluate the signal
    Returns:
        Amplitude of the signal at time t


"""
function amplitude_ti(signal::DigitizedSignal, t)

    N = length(signal.samples)
    X = fft(signal.samples)
    
    sum_real = 0.0
    for k in 0:N-1
        ω = 2π*k/(N*signal.δt)
        sum_real += real(X[k+1]) * cos(ω*t) - imag(X[k+1]) * sin(ω*t)
    end
    return sum_real/N
end

"""
reconstruct_gradient_ti(signal::DigitizedSignal, output_samples::Int)
    
    Reconstruct a signal using trigonometric interpolation.
    Args:
        signal: Input digitized signal
        output_samples: Desired output length
    Returns:
        Reconstructed signal with anti-aliasing

"""
function reconstruct_gradient_ti(signal::DigitizedSignal, output_samples)
    # Trigonometric interpolation with built-in anti-aliasing
    interp_signal = trigonometric_interpolation(signal, output_samples)
    
    # Post-filtering for residual anti-aliasing
    filtered = lowpass(interp_signal, 1/(2*signal.δt), output_samples/(signal.δt*length(signal.samples)))
    return filtered
end


function polynomial_interpolation(signal::DigitizedSignal, output_samples::Int)
    """
    Global polynomial interpolation (Caution: Numerically unstable)
    """
    x_points = (0:length(signal.samples)-1) .* signal.δt
    p = fit(x_points, signal.samples)  # Global polynomial fit
    
    new_times = range(0, signal.δt*(length(signal.samples)-1), length=output_samples)
    return p.(new_times)
end

function cubic_spline_interpolation(signal::DigitizedSignal, output_samples::Int)
    """
    Piecewise cubic spline interpolation (More stable)
    """
    x_points = (0:length(signal.samples)-1) .* signal.δt
    itp = CubicSplineInterpolation(x_points, signal.samples)
    
    new_times = range(0, signal.δt*(length(signal.samples)-1), length=output_samples)
    return itp.(new_times)
end


function amplitude_pi(signal::DigitizedSignal,
                    t; 
                    order=4,
                    window_radius=2)
    """
    Local polynomial interpolation with anti-aliasing
    order: Polynomial degree (typically 2-4)
    window_radius: Number of samples each side of center (total window = 2r+1)
    """
    # Calculate sample index and validate bounds
    max_time = (length(signal.samples)-1)*signal.δt
    t = clamp(t, 0.0, max_time)  # Handle edge times
    
    center_idx = floor(Int, t/signal.δt) + 1
    start_idx = max(1, center_idx - window_radius)
    end_idx = min(length(signal.samples), center_idx + window_radius)
    
    # Extract local window
    window_samples = signal.samples[start_idx:end_idx]
    window_times = [(i-1)*signal.δt for i in start_idx:end_idx]
    
    # # Design Vandermonde matrix
    A = hcat([window_times.^n for n in 0:order]...)
    
    # Solve least-squares system
    # A = qr(hcat([window_times.^n for n in 0:order]...))
    coeffs = A \ window_samples
    
    # Evaluate polynomial at target time
    return evalpoly(t, coeffs)
end

function reconstruct_gradient_pi(signal::DigitizedSignal,
                                output_samples;order=4,
                                window_radius=2)
    # Create output time grid
    new_times = range(0, (length(signal.samples)-1)*signal.δt, length=output_samples)
    
    # Interpolate using local polynomial
    interp = [amplitude_pi(signal, t;order=order,window_radius=window_radius) for t in new_times]
    
    # Anti-aliasing filter
    original_nyquist = 1/(2*signal.δt)
    new_fs = 1 / step(new_times) 
    filtered = lowpass(interp, original_nyquist, new_fs)
    
    return filtered
end



function amplitude_linear(signal::DigitizedSignal, t)
    
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

function reconstruct_gradient_linear(signal::DigitizedSignal, output_samples::Int)
    """
    Reconstruct a signal using linear interpolation.
    """
    # Create a new time grid for interpolation
    new_times = range(0, (length(signal.samples) - 1) * signal.δt, length=output_samples)
    
    # Interpolate at each new time point
    return [amplitude_linear(signal, t) for t in new_times]
end



function reconstruct_gradient_linear_polyfit(signal::DigitizedSignal,
                                             output_samples::Int;
                                             poly_order=4)
    """
    Hybrid reconstruction: Linear → Polynomial → Anti-aliasing
    """
    # Stage 1: Linear interpolation
    linear_interp = reconstruct_gradient_linear(signal, output_samples)
    
    # Stage 2: Polynomial refinement
    time_grid = range(0, (length(signal.samples)-1)*signal.δt, length=output_samples)
    p = fit(time_grid, linear_interp, poly_order)
    poly_refined = p.(time_grid)
    
    # Stage 3: Anti-aliasing filter
    original_nyquist = 1/(2*signal.δt)
    new_fs = output_samples/((length(signal.samples)-1)*signal.δt)
    filtered = lowpass(poly_refined, original_nyquist, new_fs)
    
    return filtered
end

function reconstruct_gradient_hybrid(signal::DigitizedSignal,
                                    output_samples::Int;
                                    secondary_method=:cubic_spline,
                                    weight=0.5,
                                    poly_order=4,
                                    window_radius=8,
                                    filter_cutoff_ratio=0.8)
    """
    Hybrid reconstruction: Linear + Specified Interpolation + Anti-aliasing
    Args:
        secondary_method: 
            :trigonometric - FFT-based interpolation
            :whittaker_shannon - Windowed sinc interpolation
            :polynomial - Local polynomial interpolation
        weight: Weighting factor for linear interpolation
        poly_order: Degree for polynomial methods
        window_radius: Kernel size for windowed sinc
        filter_cutoff_ratio: Cutoff frequency ratio (0.8 = 80% of Nyquist)
    """
    # Stage 1: Linear interpolation
    linear_interp = reconstruct_gradient_linear(signal, output_samples)
    
    # Stage 2: Secondary interpolation refinement
    time_grid = range(0, (length(signal.samples)-1)*signal.δt, length=output_samples)
    
    refined = if secondary_method == :trigonometric
        reconstruct_gradient_ti(signal, output_samples)
    elseif secondary_method == :whittaker_shannon
        reconstruct_gradient_ws(signal, output_samples)
    elseif secondary_method == :polynomial
        reconstruct_gradient_pi(signal, output_samples; order=poly_order,
                                window_radius=window_radius)
    else
        throw(ArgumentError("""
        Invalid secondary method: $secondary_method. Valid options are:
        - :trigonometric
        - :whittaker_shannon
        - :polynomial
        """))
    end
    
    # Stage 3: Weighted combination
    combined = weight*linear_interp + (1.0-weight)*refined
    
    # Stage 4: Anti-aliasing filter
    original_nyquist = 1/(2*signal.δt)
    new_fs = output_samples/((length(signal.samples)-1)*signal.δt)
    filtered = lowpass(combined, filter_cutoff_ratio*original_nyquist, new_fs)
    
    return filtered
end
"""
validate_and_expand(δΩ_,
                            grad_ode,
                            n_grad_signals,
                            n_signals,
                            n_sites,
                            T,
                            frequency_multichannel,
                            method::Symbol;
                            secondary_method=:polynomial,
                            weights=0.5,
                            poly_order=4,
                            window_radius=8,
                            filter_cutoff_ratio=0.8)
    Validate and expand the gradient signal using specified interpolation methods.
    Args:
        δΩ_: Output matrix to store expanded signals
        grad_ode: Input gradient signal matrix
        n_grad_signals: Number of original gradient signals
        n_signals: Number of desired output signals
        n_sites: Number of sites
        T: Total time duration
        frequency_multichannel: Carrier frequencies for each channel
        method: Interpolation method to use
    Returns:
        Expanded gradient signal matrix

"""

function validate_and_expand(δΩ_,
                            grad_ode,
                            n_grad_signals,
                            n_signals,
                            n_sites,
                            T,
                            frequency_multichannel,
                            method::Symbol;
                            secondary_method=:polynomial,
                            weights=0.5,
                            poly_order=4,
                            window_radius=8,
                            filter_cutoff_ratio=0.8)
    # Validate input dimensions
    @assert size(grad_ode, 1) == n_grad_signals + 1 "Input matrix must have n_grad_signals+1 rows"
    
    δt_original = T / n_grad_signals
    δt_new = T / n_signals
    
    for k in 1:n_sites
        signal = DigitizedSignal(grad_ode[:,k], δt_original, frequency_multichannel[k])
        δΩ_[:,k] = reconstruct(signal, n_signals+1, method;
                                secondary_method=secondary_method,
                                weight=weights,
                                poly_order=poly_order,
                                window_radius=window_radius,
                                filter_cutoff_ratio=filter_cutoff_ratio)
    end
    return δΩ_
end
"""
reconstruct(signal::DigitizedSignal,
            output_samples::Int,
            method::Symbol;
            secondary_method=:polynomial,
            weight=0.5,
            poly_order=4,
            window_radius=8,
            filter_cutoff_ratio=0.8)
    Reconstruct a signal using specified interpolation methods.
    Args:
        signal: Input digitized signal
        output_samples: Desired output length
        method: Interpolation method to use
    Returns:
        Reconstructed signal with anti-aliasing

"""
function reconstruct(signal::DigitizedSignal,
                    output_samples::Int,
                    method::Symbol;
                    secondary_method=:polynomial,
                    weight=0.5,
                    poly_order=4,
                    window_radius=8,
                    filter_cutoff_ratio=0.8)
    new_times = range(0, (length(signal.samples)-1)*signal.δt, length=output_samples)
    
    if method == :whittaker_shannon
        # return reconstruct_gradient_ws(signal, output_samples)
        return [amplitude_ws(signal, t) for t in new_times]
    elseif method == :linear
        return [amplitude_linear(signal, t) for t in new_times]
    elseif method == :polynomial
        return reconstruct_gradient_pi(signal, output_samples;order=poly_order,
                                       window_radius=window_radius)
    elseif method == :trigonometric
        return reconstruct_gradient_ti(signal, output_samples)
    elseif method == :hybrid
        return reconstruct_gradient_hybrid(signal, output_samples;secondary_method=secondary_method,weight=weight,
                                            poly_order=poly_order,window_radius=window_radius,filter_cutoff_ratio=filter_cutoff_ratio)
    elseif method == :linear_polyfit
        return reconstruct_gradient_linear_polyfit(signal, output_samples)
    else
        throw(ArgumentError("""
        Unknown interpolation method: $method. Valid options are:
        - :whittaker_shannon (default)
        - :trigonometric
        - :linear
        - :polynomial
        - :hybrid
        - :linear_polyfit
        """))
    end
end

