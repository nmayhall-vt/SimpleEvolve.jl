
from fileinput import filename
import numpy as np
filename  ="lih30_0.1.jl.out"
# Initialize 6 empty lists
lists= [[] for _ in range(6)]
  
with open(filename, 'r') as f:
    lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "16-element Vector{Float64}" in line :
            try:
                i += 16  # Skip the 16 lines just read
            except Exception as e:
                print(f"Error parsing 16-element vector at line {i}: {e}")
        elif "6-element Vector{Float64}" in line :
            try:
                for j in range(6):
                    value = float(lines[i + 1 + j].strip())
                    lists[j].append(value)
                i += 6  # Skip the 6 lines just read
            except Exception as e:
                print(f"Error parsing 6-element vector at line {i}: {e}")
        
        i += 1
        


for i, lst in enumerate(lists):
    print(f"List {i+1}: {lst}")

#plotting MET vs error in energy
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
# T = [80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0]
# T = np.arange(110.0, 162.0, 2.0)
# # T = np.arange(112.0, 148.0, 2.0)
# T = np.arange(100.0, 182.0, 2.0)
# T = np.arange(80.0, 152.0, 2.0)
T = np.arange(110.0, 146.0, 2.0)  # Pulse durations in ns
plt.figure(figsize=(7, 6))
plt.plot(T, np.abs(lists[0]), 'o-', label='Ground state', color='blue')
plt.plot(T, np.abs(lists[1]), 's-', label="Excited state 1", color='red')
plt.plot(T, np.abs(lists[2]), 'd-', label='Excited state 2', color='green')
plt.plot(T, np.abs(lists[3]), 'v-', label='Excited state 3', color='purple')
plt.plot(T, np.abs(lists[4]), '^-', label='Excited state 4', color='orange')
plt.plot(T, np.abs(lists[5]), '*-', label='Excited state 5', color='brown')

plt.xlabel('Pulse Duration (ns)', fontsize=14)
plt.ylabel('Energy Error (Ha)', fontsize=14)

plt.yscale('log')  # Set logarithmic scale
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(-16, 16))


def scientific_10x(x, _):
    return r'$10^{%d}$' % np.log10(x) if x != 0 else '0'

plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_10x))
plt.xticks(T, rotation=45)
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.title('Energy Error vs Pulse Duration for $\delta t$ = 0.1', fontsize=16)
plt.savefig('lih30_0.1.png', dpi=300, bbox_inches='tight')
plt.show()