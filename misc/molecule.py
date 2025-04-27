"""
This code is adapted from Oinum's ctrlQ package thta follows the Qiskit Nature library. 
"""

from sys import displayhook
from matplotlib.pylab import eig
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.operators import PolynomialTensor
import numpy as np
from qiskit.quantum_info import Pauli

# Define Pauli matrices (I, X, Y, Z) as constants
I = np.array([[1, 0], [0, 1]])  # Identity
X = np.array([[0, 1], [1, 0]])  # Pauli-X
Y = np.array([[0, -1j], [1j, 0]])  # Pauli-Y
Z = np.array([[1, 0], [0, -1]])  # Pauli-Z

def pauli_string_to_matrix(pauli_str: str, coeff: float) -> np.ndarray:
    """Convert a Pauli string (e.g., 'IZ') to its matrix representation."""
    pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    matrices = [pauli_map[p] for p in pauli_str[::-1]]  # Reverse for Qiskit's qubit order
    total = np.kron(matrices[0], matrices[1])
    for mat in matrices[2:]:
        total = np.kron(total, mat)
    return coeff * total



def h2(dist=0.75):
    driver = PySCFDriver(
        atom=f"H 0.0 0.0 0.0; H 0.0 0.0 {dist}",
        basis="sto3g",
        unit=DistanceUnit.ANGSTROM
    )
    problem = driver.run()
    
    problem = driver.run()
    
    # Get the ElectronicEnergy Hamiltonian (NOT FermionicOp!)
    hamiltonian = problem.hamiltonian
    
    # Add nuclear repulsion energy to electronic integrals
    nuclear_repulsion = problem.nuclear_repulsion_energy
    hamiltonian.electronic_integrals.alpha += PolynomialTensor({"": nuclear_repulsion})
    hamiltonian.nuclear_repulsion_energy = None  # Clear to avoid duplication
    
    mapper = ParityMapper(num_particles=problem.num_particles)
    qubit_op = mapper.map(hamiltonian.second_q_op())
    
    return qubit_op.to_matrix()

def hehp(dist=1.0):
    driver = PySCFDriver(
        atom=f"He 0.0 0.0 0.0; H 0.0 0.0 {dist}",
        basis="sto3g",
        charge=1,
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )
    problem = driver.run()
    
    problem = driver.run()
    
    # Get the ElectronicEnergy Hamiltonian (NOT FermionicOp!)
    hamiltonian = problem.hamiltonian
    
    # Add nuclear repulsion energy to electronic integrals
    nuclear_repulsion = problem.nuclear_repulsion_energy
    hamiltonian.electronic_integrals.alpha += PolynomialTensor({"": nuclear_repulsion})
    hamiltonian.nuclear_repulsion_energy = None  # Clear to avoid duplication
    
    mapper = ParityMapper(num_particles=problem.num_particles)
    qubit_op = mapper.map(hamiltonian.second_q_op())
    
    return qubit_op.to_matrix()

def lih(dist=1.5):
    driver = PySCFDriver(
        atom=f"Li 0.0 0.0 0.0; H 0.0 0.0 {dist}",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )
    problem = driver.run()
    
    # Get the ElectronicEnergy Hamiltonian (NOT FermionicOp!)
    hamiltonian = problem.hamiltonian
    
    # Add nuclear repulsion energy to electronic integrals
    nuclear_repulsion = problem.nuclear_repulsion_energy
    hamiltonian.electronic_integrals.alpha += PolynomialTensor({"": nuclear_repulsion})
    hamiltonian.nuclear_repulsion_energy = None  # Clear to avoid duplication
    
    mapper = ParityMapper(num_particles=problem.num_particles)
    qubit_op = mapper.map(hamiltonian.second_q_op())
    
    return qubit_op.to_matrix()


# lih_ham=lih(1.5)
# hehp_ham=hehp(1.0)
# h2_ham=h2(0.75)
# print(hehp_ham)
# print(h2_ham)
# print(lih_ham)
# print(eig(hehp_ham))
# print(eig(h2_ham))
# print(eig(lih_ham))

# print(eig(h2o_ham))
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.units import DistanceUnit
import numpy as np

def h2o(dist_oh=0.96, angle_hoh=104.5):
    """
    Generates the qubit Hamiltonian matrix for H₂O with frozen core orbitals (oxygen 1s).
    """
    # Calculate H positions
    theta = np.radians(angle_hoh / 2)
    x = dist_oh * np.cos(theta)
    y = dist_oh * np.sin(theta)
    
    # Define H₂O geometry
    atom = f"""O  0.0  0.0  0.0;H  {x:.3f}  {y:.3f}  0.0;H  {-x:.3f}  {y:.3f}  0.0"""
    
    # Set up driver and problem
    driver = PySCFDriver(
        atom=atom,
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )
    problem = driver.run()
    
    # Freeze core orbitals (oxygen 1s) using ActiveSpaceTransformer
    transformer = ActiveSpaceTransformer(
    num_electrons=6,          # Total electrons: 10-4 = 6e⁻
    num_spatial_orbitals=5,   # Total orbitals: 7 - 2 (frozen) = 5
    active_orbitals=[2,3,4,5,6])  # Keep orbitals 1-6 (exclude 0))
    problem_reduced = transformer.transform(problem)
    
    # Get Hamiltonian (already includes nuclear repulsion energy)
    hamiltonian = problem_reduced.hamiltonian
    
    # Map to qubit operator
    mapper = ParityMapper(num_particles=problem_reduced.num_particles)
    qubit_op = mapper.map(hamiltonian.second_q_op())
    
    return qubit_op.to_matrix()

# Example usage
h2o_ham = h2o()
print("H2O Hamiltonian matrix:")
print(h2o_ham)
print(np.shape(h2o_ham))

ground_state_energy = eig(h2o_ham)[0][0]
print(ground_state_energy)
np.save('h2o_ham.npy', h2o_ham)