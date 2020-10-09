import qiskit.circuit.library.standard_gates
from qiskit import Aer
from qiskit import execute

def get_state_vector(circuit):
    """
    Input:
        - circuit (qiskit.QuantumCircuit)
    Returns:
        - state vector
    """
    backend = Aer.get_backend("statevector_simulator")
    psi = execute(circuit, backend).result().get_statevector()

    return psi

def get_counts(circuit, measure_all=True):
    """
    Input:
        - circuit (qiskit.QuantumCircuit)
        - measure_all=True (boolean): If true, adds measurement operations and corresponding
        classical registers for all qubits
    Returns:
        - dictionary of counts
    """
    if measure_all == True:
        circuit.measure_all()

    backend = Aer.get_backend("qasm_simulator")
    counts = execute(circuit, backend).result().get_counts(circuit)

    return counts

class Gate:
    """
    A class for easy access to Qiskit gates.
    """

    def __init__(self, gate_string):
        self.gate_string = gate_string.lower()
        self.gate = self.gate_parser()


    def gate_parser(self):
        if self.gate_string == "x":
            return Gate.X
        elif self.gate_string == "y":
            return Gate.Y
        elif self.gate_string == "z":
            return Gate.Z
        elif self.gate_string == "h":
            return Gate.H
        elif self.gate_string == "s":
            return Gate.S
        elif self.gate_string == "sdg":
            return Gate.Sdg
        elif self.gate_string == "t":
            return Gate.T
        elif self.gate_string == "tdg":
            return Gate.Tdg
        elif self.gate_string == "rx":
            return Gate.RX
        elif self.gate_string == "ry":
            return Gate.RY
        elif self.gate_string == "rz":
            return Gate.RZ
        elif self.gate_string == "u1":
            return Gate.U1
        elif self.gate_string == "u2":
            return Gate.U2
        elif self.gate_string == "u3":
            return Gate.U3
        elif self.gate_string == "cx":
            return Gate.CX
        elif self.gate_string == "cy":
            return Gate.CY
        elif self.gate_string == "cz":
            return Gate.CZ
        elif self.gate_string == "swap":
            return Gate.SWAP
        else:
            return "Invalid gate string"


    @staticmethod
    def print_gates():
        print("X\n",
              "Y\n",
              "Z\n",
              "H\n",
              "S\n",
              "Sdg\n",
              "T\n",
              "Tdg\n",
              "RX\n",
              "RY\n",
              "RZ\n",
              "U1\n",
              "U2\n",
              "U3\n",
              "CX\n",
              "CY\n",
              "CZ\n",
              "SWAP")

    # --------------------> one qubit gates <-----------------
    # Pauli
    X = qiskit.circuit.library.standard_gates.XGate()
    Y = qiskit.circuit.library.standard_gates.YGate()
    Z = qiskit.circuit.library.standard_gates.ZGate()

    # Hadamard
    H = qiskit.circuit.library.standard_gates.HGate()

    # Phase
    S = qiskit.circuit.library.standard_gates.SGate() # ~=Z**0.5
    Sdg = qiskit.circuit.library.standard_gates.SdgGate()

    T = qiskit.circuit.library.standard_gates.TGate() # ~=Z**0.25
    Tdg = qiskit.circuit.library.standard_gates.TdgGate()

    # Puali Rotations
    RX = lambda theta : qiskit.circuit.library.standard_gates.RXGate(theta)
    RY = lambda theta : qiskit.circuit.library.standard_gates.RYGate(theta)
    RZ = lambda theta : qiskit.circuit.library.standard_gates.RZGate(theta)

    # "U" gates
    U1 = lambda theta : qiskit.circuit.library.standard_gates.U1Gate(theta)
    U2 = lambda phi, lam : qiskit.circuit.library.standard_gates.U2Gate(phi, lam)
    U3 = lambda theta, phi, lam : qiskit.circuit.library.standard_gates.U2Gate(theta, phi, lam)

    # --------------------> two qubit gates <-----------------
    CX = qiskit.circuit.library.standard_gates.CXGate()
    CY = qiskit.circuit.library.standard_gates.CYGate()
    CZ = qiskit.circuit.library.standard_gates.CZGate()

    RXX = lambda theta : qiskit.circuit.library.standard_gates.RXXGate(theta)
    RZX = lambda theta : qiskit.circuit.library.standard_gates.RZXGate(theta)
    RZZ = lambda theta : qiskit.circuit.library.standard_gates.RZZGate(theta)
    RYY = lambda theta : qiskit.circuit.library.standard_gates.RYYGate(theta)

    swap = qiskit.circuit.library.standard_gates.SwapGate()
