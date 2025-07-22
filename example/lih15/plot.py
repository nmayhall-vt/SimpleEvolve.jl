import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter

# Initialize data storage
data = {
    2.0: [[] for _ in range(6)],
    1.0: [[] for _ in range(6)],
    0.5: [[] for _ in range(6)],
    0.2: [[] for _ in range(6)]
}

T_values = []  # Store all T values
current_T = None
current_dt = None

filename = "lih15_dt_variation.jl.out"

with open(filename, 'r') as f:
    lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Capture T value
        if line.startswith("T = "):
            try:
                current_T = float(line.split()[-1])
                if current_T not in T_values:
                    T_values.append(current_T)
            except ValueError:
                pass
            i += 1
        
        # Capture δt value
        elif "δt = " in line:
            try:
                current_dt = float(line.split()[-1])
                # Only process if dt is in our target values
                if current_dt not in data:
                    current_dt = None
            except ValueError:
                current_dt = None
            i += 1
        
        # Process vectors
        elif current_dt is not None and current_T is not None:
            if "16-element Vector{Float64}" in line:
                # Skip 16-element vector (current line + next 16 lines)
                i += 17  # Skip vector declaration and 16 data lines
            elif "6-element Vector{Float64}" in line:
                # Process 6-element vector
                try:
                    for j in range(6):
                        if i+1+j >= len(lines):
                            break
                        value = float(lines[i+1+j].strip())
                        data[current_dt][j].append(value)
                    i += 7  # Skip vector declaration and 6 data lines
                except (ValueError, IndexError):
                    i += 1
            else:
                i += 1
        else:
            i += 1

# Print extracted data
for dt, vectors in data.items():
    print(f"\nδt = {dt}")
    for i, vec in enumerate(vectors):
        print(f"Vector {i+1}: {vec}")

# Plotting for δt = 2.0
plt.figure(figsize=(10, 6))
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
labels = ['Ground state', 'Excited state 1', 'Excited state 2', 
          'Excited state 3', 'Excited state 4', 'Excited state 5']

for i in range(6):
    plt.plot(T_values, np.abs(data[2.0][i]), 
             marker='o', linestyle='-', 
             color=colors[i], label=labels[i])

plt.xlabel('Pulse Duration (ns)', fontsize=14)
plt.ylabel('Energy Error (Ha)', fontsize=14)
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

def scientific_10x(x, _):
    return r'$10^{%d}$' % int(np.log10(x)) if x > 0 else '0'

plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_10x))
plt.xticks(T_values, rotation=45)
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.title('Energy Error vs Pulse Duration (δt = 2.0)', fontsize=16)
plt.tight_layout()
# plt.show()
plt.savefig('lih15_dt_variation_2.0.pdf', dpi=300)