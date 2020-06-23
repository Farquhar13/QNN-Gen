from abc import ABC, abstractmethod
from qiskit import QuantumCircuit, QuantumRegister
import qiskit
import numpy as np

from Utility import Gate


# ------------> Encode Base Class <------------
class Encode(ABC):
    """
    Base class for encoding. Derived classes must overwrite the abstract methods.
    """

    def __init__(self):
        pass

    @abstractmethod
    def circuit(self, x):
        pass

    @abstractmethod
    def n_qubits(self, x):
        pass

    def all_circuits(self, D):
        circuit_list = []

        for x in D:
            circuit_list.append(self.circuit(x))

        return circuit_list


# ------------> Angle Encoding <------------
class AngleEncoding(Encode):
    """
    Angle Encoding class. Assumes data is feature-normazized.
    """

    def __init__(self, gate=Gate.RY, scaling=np.pi/2):
        """
        Attributes:
            gate=Gate.RY [Qiskit Gate]: The rotation gate
            scaling=np.pi/2 [float]: Number by which to scale normalized input data.
            The defualt scaling is pi/2 which does not induce a relative phase
            difference.
        """

        self.gate = gate
        self.scaling = scaling

    def n_qubits(self, x):
        """
        Input:
            - x [np.ndarray]: The input data to encode

        Returns:
            - Number of qubits needed to encode x
        """

        return len(x)

    def circuit(self, x):
        """
        Input:
            - x [np.ndarray]: The input data to encode

        Returns:
            - [qiskit.QuantumCircuit]: The circuit that encodes x

        Assumes data is feature-normalized. Assumes every element in x should be in [0, 1].
        """

        n_qubits = self.n_qubits(x)

        Sx = QuantumCircuit(n_qubits)

        for i in range(n_qubits):
            Sx.append(self.gate(2 * self.scaling * x[i]), [i])

        return Sx

    def state_vector(self, x):
        """
        Input:
            - x [np.ndarray]: The input data to encode

        Returns:
            - np.array: The state vector representation of x after angle encoding
        """

        from functools import reduce

        gate_fn = lambda x: self.gate(2 * self.scaling * x).to_matrix()[:, 0]
        qubit_states = list(map(gate_fn, x[::-1]))

        """
        # Leaving this here because it might be more readable. Does the same
        # as the above code.
        qubit_states = []
        for x_i in x:
            qubit_state =  self.gate(2 * self.scaling * x_i).to_matrix()[:, 0]
            qubit_states.append(qubit_state)
        """
        return reduce(lambda a, b: np.kron(a, b), qubit_states)


# ------------> Dense Angle Encoding <------------
class DenseAngleEncoding(Encode):
    """
    Dense Angle Encoding Class.

    Assumptions:
    - Assumes data is feature-normalized
    - Assumes the number of featers in x is divisible by two. If this is not the
    case for your data, consider appending zeros.
    """

    def __init__(self, rotation_gate=Gate.RY, scaling=np.pi/2):
        """
        Attributes:
            gate=Gate.RY [Qiskit Gate]: The rotation gate
            scaling=np.pi/2 [float]: Number by which to scale normalized input data.
            The defualt scaling is pi/2 which does not induce a relative phase
            difference.
        """

        self.rotation_gate = rotation_gate
        self.scaling = scaling

    def n_qubits(self, x):
        """
        Assumes the number of features in x is divisible by two

        Input:
            - x [np.ndarray]: The input data to encode

        Returns:
            - Number of qubits needed to encode x
        """

        assert_string = "DenseAngleEncoding assumes the number of features in x is divisible by two"
        assert (len(x) / 2) % 1 == 0, assert_string

        return len(x) // 2

    def circuit(self, x):
        """
        Input:
            - x [np.ndarray]: The input data to encode

        Returns:
            - [qiskit.QuantumCircuit]: The circuit that encodes x

        Assumes data is feature-normalized.
        Assumes every element in x should be in [0, 1].
        """

        n_qubits = self.n_qubits(x)

        Sx = QuantumCircuit(n_qubits)

        for i in range(n_qubits):
            rotation_idx = 2*i
            phase_idx = 2*i + 1

            phase_factor = 2*np.pi * x[phase_idx]

            Sx.append(self.rotation_gate(2 * self.scaling * x[rotation_idx]), [i])
            Sx.u1(phase_factor, [i])

        return Sx

    def state_vector(self, x):
        """
        Input:
            - x [np.ndarray]: The input data to encode

        Returns:
            - np.array: The state vector representation of x after dense angle encoding
        """

        from functools import reduce

        qubit_states = []
        for i in range(len(x)//2):
            x_i = x[2*i] # qubit that defines the rotation
            x_j = x[2*i + 1] # qubit that defines the phase
            rotation = self.rotation_gate(2 * self.scaling * x_i).to_matrix()[:, 0]
            phase_and_rotation = [rotation[0], np.exp(2 * np.pi * 1j *x_j) * rotation[1]]
            qubit_states.append(phase_and_rotation)

        qubit_states = qubit_states[::-1] # match qiskit qubit ordering

        return reduce(lambda a, b: np.kron(a, b), qubit_states)


# ------------> Binary Phase Encoding <------------
class BinaryPhaseEncoding(Encode):
    """
    Binary Phase Encoding Class.

    Assumptions:
        - Assumes binary data with each feature in {-1, 1}
        - Assumes the number of featers in x is a power of two
        case for your data, consider appending zeros.
    """

    def __init__(self, method="SF", ancilla=True):
        """
        Attributes:
            - method="SF": For sign-flip algorithm.
            - ancilla=True [Boolean]: Add ancillary qubit to circuit

        Hypergraph state generation as a method in a future release
        """

        self.method = method
        self.ancilla = ancilla

    def n_qubits(self, x):
        """
        Input:
            - x [np.ndarray]: The input data to encode

        Returns:
            - Number of qubits needed to encode x. Includes ancilla qubit if
            self.ancilla is true

        Assumes the number of features in x is a power of 2
        """

        assert_string = "BinaryPhaseEncoding assumes the number of features in x is a power of 2"
        assert np.log2(len(x)) % 1 == 0, assert_string

        n_qubits = int(np.log2(len(x)))
        if self.ancilla == True:
            n_qubits += 1

        return n_qubits


    def circuit(self, x, ancilla=None):
        """
        Input:
            - x [np.ndarray]: The input data to encode
            - ancilla=True [Boolean]: Adds an ancillary qubit to the circuit, if true

        Returns:
            - [qiskit.QuantumCircuit]: The circuit that encodes x

        Assumptions:
            - Assumes binary data with each feature in {-1, 1}
            - Assumes the number of featers in x is a power of two
            case for your data, consider appending zeros.
        """

        if ancilla is None:
            ancilla = self.ancilla

        if self.method == "SF":
            return self.SF(x, ancilla)
        elif self.method == "HSGS":
            raise NotImplementedError
        else:
            raise ValueError


    def SF(self, x, ancilla):
        """
        Sign-Flip algorithm for data encoding

        Input:
            - x [np.ndarray]: The input data to encode
            - ancilla=True [Boolean]: Adds an ancillary qubit to the circuit, if true

        Returns:
            - [qiskit.QuantumCircuit]: The circuit that encodes x using the SF algorithm


        Assumptions:
            - Assumes binary data with each feature in {-1, 1}
            - Assumes the number of featers in x is a power of two
            case for your data, consider appending zeros.
        """

        # Validate input
        """
        assert_string = "Elements of x vector must be either -1 or 1"
        assert all((x == -1) | (x == 1)), assert_string
        """

        x = np.array(x)
        d = len(x) # dimensionality of data vector
        n_qubits_ancilla = self.n_qubits(x) # qubits plus ancilla qubit if self.ancilla=True

        if self.ancilla == True:
            n_qubits = n_qubits_ancilla - 1 # non ancilla qubits
        else:
            n_qubits = n_qubits_ancilla

        Sx = QuantumCircuit(n_qubits_ancilla)
        qubit_idx_list = list(range(n_qubits))

        # Generate computational basis vectors in Dirac notation
        basis_labels = [("{:0%db}"%n_qubits).format(k) for k in range(d)]

        # Create multi-controlled Z gate, or single Z gate if N = 1qubit.
        Z = qiskit.circuit.library.standard_gates.ZGate()
        if n_qubits == 1:
            z_op = Z
        else:
            z_op = Z.control(n_qubits-1)

        # Full layer of H
        Sx.h(qubit_idx_list)

        # Find all components with a -1 factor in i (and thus our target state vector)
        indices = np.where(x == -1)[0]
        if indices.size > 0:
            for idx in indices:
                # Need to switch qubits in the 0 state so CZ will take effect
                for i, b in enumerate(basis_labels[idx]):
                    if b == '0':
                        Sx.x((n_qubits-1)-i) # (N-1)-i is to match the qubit ordering Qiskit uses (reversed)

                Sx.append(z_op, qubit_idx_list)

                # And switch the flipped qubits back
                for i, b in enumerate(basis_labels[idx]):
                    if b == '0':
                        Sx.x((n_qubits-1)-i)

        return Sx


class BasisEncoding(Encode):
    """
    Basis Encoding Class.

    Encodes binary vectors (can be thought of as bit strings) into the state
    with the corresponding label in Dirac notation.
    """

    def __init__(self):
        pass

    def n_qubits(self, x):
        """
        Input:
            - x [np.ndarray]: The input data to encode

        Returns:
            - Number of qubits needed to encode x.
        """

        return len(x)

    def circuit(self, x):
        """
        Input:
            - x [np.ndarray]: The input data to encode

        Returns:
            - [qiskit.QuantumCircuit]: The circuit that encodes x

        Assumptions:
            - Assumes binary data with each feature in {0, 1}
        """

        assert (x.count(0) + x.count(1)) == len(x), "All features must be {0, 1}"


        x = np.array(x)
        x_reversed = x[::-1] # match Qiskit qubit ordering

        n_qubits = self.n_qubits(x) # bit pedantic, but it's consistent with other classes
        Sx = QuantumCircuit(n_qubits)

        one_indices = np.where(x_reversed == 1)[0]
        for i in one_indices:
            Sx.x(i)

        return Sx
