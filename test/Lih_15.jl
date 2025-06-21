using SimpleEvolve
using NPZ
using Plots
using LinearAlgebra
using Optim
using LineSearches
using FFTW
using JLD2
using Random
using Printf
using Test


@testset "n_level 2 Optimization " begin
    Cost_ham = npzread("lih15.npy")
    # display(Cost_ham)
    n_qubits = round(Int, log2(size(Cost_ham, 1)))
    println("Number of qubits: ", n_qubits)
    n_levels = 2
    SYSTEM = "lih15"
    Π = projector(n_qubits, 2, n_levels)
    Cost_ham = Hermitian(Π' * Cost_ham * Π)
    device = choose_qubits(1:n_qubits, Transmon(
        2π * [4.82, 4.84, 4.86, 4.88],                    # QUBIT RESONANCE FREQUENCIES
        2π * [0.3, 0.3, 0.3, 0.3],                        # QUBIT ANHARMONICITIES
        Dict{QubitCoupling,Float64}(                      # QUBIT COUPLING CONSTANTS
            QubitCoupling(1, 2) => 2π * 0.02,
            QubitCoupling(2, 3) => 2π * 0.02,
            QubitCoupling(3, 4) => 2π * 0.02,
            QubitCoupling(1, 3) => 2π * 0.02,
            QubitCoupling(2, 4) => 2π * 0.02,
            QubitCoupling(1, 4) => 2π * 0.02,
        )
    ))

    T = 100.0
    δt = 0.5
    n_samples = Int(T / δt)
    t_ = collect(0:δt:T)
    freqs = 2π * [4.82, 4.84, 4.86, 4.88]
    carrier_freqs = freqs#.-2π * 0.1  # with detuning
    # INITIAL PARAMETERS
    samples_matrix = [2π * 0.00000002 * sin(2π * (t / n_samples)) for t in 0:n_samples, i in 1:n_qubits]
    samples_matrix = [samples_matrix[:, i] .+ im * samples_matrix[:, i] for i in 1:n_qubits]
    samples_matrix = hcat(samples_matrix...)
    pulse_windows = range(0, T, length=n_samples + 1)
    samples_initial = [real(samples_matrix[:]); imag(samples_matrix[:])]
    signals_ = [DigitizedSignal([samples_matrix[:, i]], δt, carrier_freqs[i]) for i in 1:n_qubits]
    signals = MultiChannelSignal(signals_)



    # initial state
    initial_state = "1"^(n_qubits ÷ 2) * "0"^(n_qubits ÷ 2)
    ψ_initial = zeros(ComplexF64, n_levels^n_qubits)
    ψ_initial[1+parse(Int, initial_state, base=n_levels)] = one(ComplexF64)


    H_static = static_hamiltonian(device, n_levels)
    #eigenvalues and eigenvectors of the static Hamiltonian
    drives = a_fullspace(n_qubits, n_levels)
    eigvalues, eigvectors = eigen(Hermitian(H_static))
    println("Eignvalues of our static Hamiltonian")
    # display(eigvalues)

    tol_ode = 1e-6
    Λ, U = eigen(Cost_ham)
    E_actual = Λ[1]
    println("Actual energy: $E_actual")
    # display(drives[1])
    # display(eigvectors)
    for i in 1:n_qubits
        drives[i] = eigvectors' * drives[i] * eigvectors
    end


    # we have to optimize the samples in the signal
    n_samples_grad = n_samples
    δΩ_ = Matrix{Float64}(undef, n_samples + 1, n_qubits)
    a = a_q(n_levels)
    tol_ode = 1e-6
    # gradientsignal for less no of samples
    δΩ = zeros(n_samples_grad + 1, n_qubits)
    Grad = zeros(Float64, n_samples + 1, n_qubits)
    samples_initial = reshape(samples_matrix, :)
    dt = T / n_samples_grad

    function gradient_ode!(Grad, samples;
        λ_amp=1.0, Ω₀=2π * 0.02)

        Grad .= 0.0
        n = length(samples) ÷ 2

        # Split into real/imag parts
        samples_real = samples[1:n]
        samples_imag = samples[n+1:end]

        # Reshape into complex signal matrix
        samples_complex = complex.(
            reshape(samples_real, (n_samples + 1, n_qubits)),
            reshape(samples_imag, (n_samples + 1, n_qubits))
        )
        # Pre-allocate arrays for FFT analysis
        fft_before = Matrix{ComplexF64}(undef, n_samples + 1, n_qubits)
        fft_after = similar(fft_before)
        max_freq_before = Vector{Float64}(undef, n_qubits)
        max_freq_after = similar(max_freq_before)

        # Apply lowpass filter to each qubit's pulse
        fs = 1 / (δt)      # Sampling frequency (GHz)
        cutoff = fs / 2.0001  # Cutoff frequency (GHz) 
        order = 4        # Filter order

        for i in 1:n_qubits
            pulse = samples_complex[:, i]

            # Compute FFT before filtering
            fft_before[:, i] = fft(pulse)

            # Apply lowpass filter
            filtered_pulse = SimpleEvolve.lowpass_filter(pulse, cutoff, fs; order=order)
            samples_complex[:, i] = filtered_pulse

            # Compute FFT after filtering
            fft_after[:, i] = fft(filtered_pulse)

            # Calculate maximum frequency magnitude
            max_freq_before[i] = maximum(abs.(fft_before[:, i]))
            max_freq_after[i] = maximum(abs.(fft_after[:, i]))
        end

        # Build signals from filtered complex samples
        signals_ = [DigitizedSignal(samples_complex[:, i], δt, carrier_freqs[i]) for i in 1:n_qubits]
        signals = MultiChannelSignal(signals_)

        # Call gradient computation
        ∂Ω_real, ∂Ω_imag, ψ_ode, σ_ode = SimpleEvolve.gradientsignal_ODE(
            ψ_initial, T, signals, n_qubits, drives, eigvalues, eigvectors,
            Cost_ham, n_samples_grad;
            basis="qubitbasis", tol_ode=tol_ode
        )
        # === Apply Amplitude Penalty if Required ===
        grad_final = hcat(vec(∂Ω_real), vec(∂Ω_imag))
        grad_final = reshape(grad_final, :)

        for i in 1:2*(n_samples+1)
            x = samples[i] / Ω₀
            y = abs(x) - 1
            grad_penalty = 0.0
            if y > 0
                h = exp(y - 1 / y)
                dh_dx = h * (1 + 1 / y^2) / Ω₀
                grad_penalty = sign(x) * dh_dx
            end
            grad_final[i] += λ_amp * grad_penalty
        end

        Grad .= grad_final

        return Grad
    end



    function costfunction_o(samples)
        # Split real vector into real and imaginary components
        n = length(samples) ÷ 2
        samples_real = samples[1:n]
        samples_imag = samples[n+1:end]

        # Reshape into complex matrix (n_samples+1 × n_qubits)
        samples_complex = complex.(
            reshape(samples_real, (n_samples + 1, n_qubits)),
            reshape(samples_imag, (n_samples + 1, n_qubits))
        )

        # Pre-allocate arrays for FFT analysis
        fft_before = Matrix{ComplexF64}(undef, n_samples + 1, n_qubits)
        fft_after = similar(fft_before)
        max_freq_before = Vector{Float64}(undef, n_qubits)
        max_freq_after = similar(max_freq_before)

        # Apply lowpass filter to each qubit's pulse
        fs = 1 / (δt)      # Sampling frequency (GHz)
        cutoff = fs / 2.0001  # Cutoff frequency (GHz) 
        order = 4        # Filter order

        for i in 1:n_qubits
            pulse = samples_complex[:, i]

            # Compute FFT before filtering
            fft_before[:, i] = fft(pulse)

            # Apply lowpass filter
            filtered_pulse = SimpleEvolve.lowpass_filter(pulse, cutoff, fs; order=order)
            samples_complex[:, i] = filtered_pulse

            # Compute FFT after filtering
            fft_after[:, i] = fft(filtered_pulse)

            # Calculate maximum frequency magnitude
            max_freq_before[i] = maximum(abs.(fft_before[:, i]))
            max_freq_after[i] = maximum(abs.(fft_after[:, i]))
        end

        # Build signals from filtered complex samples
        signals_ = [DigitizedSignal(samples_complex[:, i], δt, carrier_freqs[i]) for i in 1:n_qubits]
        signals = MultiChannelSignal(signals_)

        energy, Ψ_ode = SimpleEvolve.costfunction_ode_with_penalty(
            ψ_initial, eigvalues, signals, n_qubits, drives, eigvectors, T, Cost_ham;
            basis="qubitbasis", tol_ode=tol_ode
        )

        return energy
    end

    # OPTIMIZATION ALGORITHM
    linesearch = LineSearches.MoreThuente()
    # optimizer = Optim.BFGS(linesearch=linesearch)
    optimizer = Optim.LBFGS(linesearch=linesearch)
    # OPTIMIZATION OPTIONS
    options = Optim.Options(
        show_trace=true,
        show_every=1,
        f_reltol=1e-12,
        g_tol=1e-9,
        iterations=20,
    )

    tol_ode = 1e-4
    samples_initial = [real(samples_matrix[:]); imag(samples_matrix[:])]
    Grad = zeros(Float64, 2 * (n_samples + 1), n_qubits)
    optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_initial, optimizer, options)
    samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS
    tol_ode = 1e-6
    optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
    samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS
    tol_ode = 1e-8
    optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
    samples_final = Optim.minimizer(optimization)
    tol_ode = 1e-10
    optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
    samples_final = Optim.minimizer(optimization)
    E_final = costfunction_o(samples_final)
    E0 = costfunction_o(samples_initial)
    @testset "Final Energy Accuracy" begin
        @test isfinite(E_final)
        @test E_final < E0
        @test E_final - Λ[1] < 1e-8  
    end
    #post processing
    n = Int(length(samples_final) / 2)
    samples_real = reshape(samples_final[1:n], (n_samples + 1, n_qubits))
    samples_imag = reshape(samples_final[n+1:end], (n_samples + 1, n_qubits))
    Ω = complex.(samples_real, samples_imag)
    Ω = reshape(Ω, n_samples + 1, n_qubits)
    Ω0 = copy(samples_matrix)
    pulse_windows = range(0, T, length=n_samples + 1)
    Ω_plots = plot(
        [plot(
            pulse_windows, real.(Ω[:, q])
        ) for q in 1:n_qubits]...,
        title="Final real Signals",
        legend=false,
        layout=(n_qubits, 1),
    )
    Ω_plots_final = plot(
        [plot(
            pulse_windows, imag.(Ω[:, q])
        ) for q in 1:n_qubits]...,
        title="Final imag Signals",
        legend=false,
        layout=(n_qubits, 1),
    )
    plot(Ω_plots, Ω_plots_final, layout=(1, 2))
    savefig("final_signals_$(n_qubits)_$(n_levels)_$(SYSTEM)_$(n_samples)_$(T).pdf")
end