"""
Will need to import Measurement for default_measurements
"""

from abc import ABC, abstractmethod
from qiskit import QuantumCircuit, QuantumRegister
import qiskit
import numpy as np

from Utility import Gate


class Model(ABC):
    """
    Class to define a QML model, whether it be a particular ansatz or a perceptron model.

    Abstract class. Derived classes overwrite circuit() and default_measuremnt(). Derived classes should
    also have a self.params attribute.
    """

    def __init__(self):
        """
        Attributes:
            - n_qubits [int]: Number of encoded qubits in the encoding circuit to be joined with
            the model circuit
            - params [np.ndarray]: The model's learnable parameters
        """

        self.n_qubits = None
        self.parameters = None

    @abstractmethod
    def circuit():
        """
        Abstract method. Overwrite to return the circuit.

        Returns:
            - [qiskit.QuantumCircuit]: The circuit defined by the model
        """
        pass


    def default_measurement():
        """
        Often, the model implies what measurements are sensible.

        Returns:
            - [Measurement]: The default measurement for the model
        """

        raise NotImplementedError

    @staticmethod
    def print_models():
        """
        Prints the available derived classes.
        """

        print("TensorTreeNetwork" + "\n",
              "BinaryPerceptron")


class TensorTreeNetwork(Model):
    """
    Class to implement Tensor Tree Networks.

    Assumptions:
        - Number of features is a power of two.
    """

    def __init__(self, n_qubits=None, angles=None, rotation_gate=Gate.RY, entangling_gate=Gate.CX):
        """
        Inputs:
            - n_qubits=None (int): Number of encoded qubits
            - anlges=None (list): Parameters for rotation_gate. Sets self.parameters
            - rotation_gate=Gate.RY (qiskit.circuit.library.standard_gates): 1-qubit rotation gate
            - entangling_gate=Gate.CX (qiskit.circuit.library.standard_gates): 2-qubit entangling gate
        """

        self.n_layers = None
        self.measurement_qubit = None
        self.parameters = angles
        self.n_qubits = n_qubits # may update attributes above
        self.rotation_gate = rotation_gate
        self.entangling_gate = entangling_gate

    @property
    def n_qubits(self):
        return self._n_qubits

    @n_qubits.setter
    def n_qubits(self, n_qubits):
        """
        Sets self.n_qubits. Assumes n_qubits is a power of two.

        Also may update self.n_layers, self.measurement_qubit, and self.angles
        """

        assert np.log2(n_qubits) % 1 == 0, "Assumes n_qubits is a power of two."
        self._n_qubits = n_qubits

        self.n_layers = int(np.log2(n_qubits))
        self.measurement_qubit = n_qubits // 2

        if self.parameters is None:
            self.parameters = self.get_rand_angles(n_qubits)
        else:
            if int(2**(self.n_layers + 1)) != len(self.parameters):
                import warnings
                warnings.warn("""Number of quibts no longer match the number of parameters.
                              To update parameters use set_params(new_angles) or
                              init_rand_angles(n_qubits).""")


    def get_rand_angles(self, n_qubits):
        """
        Generates random numbers from [0, pi)
        """

        n_parameters = 2**(self.n_layers + 1) - 2

        rand_angles = np.random.rand(n_parameters) * np.pi

        return rand_angles


    def circuit(self):
        """
        Returns:
            - qiskit.QuantumCircuit
        """
        n_qubits = self.n_qubits
        angles = self.parameters
        qc = QuantumCircuit(n_qubits)

        def layer(angle_idx):
            qubit_pairs = [(active_qubits[2*i], active_qubits[2*i+1]) for i in range(int(len(active_qubits)/2))]

            next_active = []

            for i, q in enumerate(qubit_pairs):
                qc.append(self.rotation_gate(angles[angle_idx]), [q[0]])
                qc.append(self.rotation_gate(angles[angle_idx + 1]), [q[1]])

                if q[0] < self.measurement_qubit:
                    qc.append(self.entangling_gate, [q[0], q[1]])
                    next_active.append(q[1])
                else:
                    qc.append(self.entangling_gate, [q[1], q[0]])
                    next_active.append(q[0])

                angle_idx += 2

            return next_active, angle_idx

        active_qubits = np.arange(n_qubits) # qubit indices
        n_active = len(active_qubits)

        angle_idx = 0
        while n_active >= 2:
            active_qubits, angle_idx = layer(angle_idx)
            n_active = len(active_qubits)

        return qc


    def default_measurement(self, observable=None, measurement_qubit=None):
        """
        Default measurement is the Z expectation of the measurement_qubit

        Inputs:
            - observable=None [Observable]: If None, defaults to Observable.Z()
            - measurement_qubit [int]: If None, defaults to self.measurement_quibt = n_qubits//2

        Returns:
            - Expectation() measurement object
        """
        if observable is None:
            observable = Observable.Z()

        if measurement_qubit is None:
            measurement_qubit = self.measurement_qubit

        expectation = Expectation(measurement_qubit, observable)

        return expectation


class BinaryPerceptron(Model):
    """
    Class to implement Binary Perceptron models.

    Assumptions:
        - Number of features is a power of two.
        - Features are binary elements in {-1, 1}
    """

    def __init__(self, n_qubits=None, weights=None):
        """
        Attribuites:
            - n_qubits [int]
            - weights [np.ndarray]: A binary vector with elements in {-1, 1}
        """

        # Ensure weights is a np array if passed
        if weights is not None:
            self.parameters = np.array(weights)
        else:
            self.parameters = weights
        self.measurement_qubit = None
        self.n_qubits = n_qubits

    @property
    def n_qubits(self):
        return self._n_qubits

    @n_qubits.setter
    def n_qubits(self, n_qubits):
        """
        Input:
            n_qubits [int]: number of encoded qubits

        Sets self.n_qubits. Assumes n_qubits is a power of two.
        Also may update self.n_layers, self.measurement_qubit, and self.angles
        """

        assert np.log2(n_qubits) % 1 == 0, "Assumes n_qubits is a power of two."
        self._n_qubits = n_qubits

        if self.measurement_qubit is None:
            if n_qubits == 1:
                self.measurement_qubit = 0
            else:
                self.measurement_qubit = n_qubits

        if self.parameters is None:
            self.parameters = self.init_random_weights(self.n_qubits)
        else:
            if int(2**n_qubits) != len(self.parameters):
                import warnings

                warnings.warn("""Number of quibts no longer match the number of parameters.
                              To update parameters use set_params(new_weights) or
                              init_random_weights(n_qubits).""")

    def init_random_weights(self, n_qubits):
        return np.random.choice([-1, 1], 2**n_qubits)

    def circuit(self):
        """
        Returns the circuit for the model.
        """

        w = self.parameters

        # Not including ancilla
        N = self.n_qubits
        qubit_idx_list = list(range(N))

        # Generate computational basis vectors in Dirac notation
        basis_labels = [("{:0%db}"%N).format(k) for k in range(len(w))]

        # No need for ancilla if N = 1
        if N ==1:
            Sx = QuantumCircuit(N)
            qubit_ancilla_idx_list = qubit_idx_list

        else:
            Sx = QuantumCircuit(N + 1) # + 1 for ancilla

            qubit_ancilla_idx_list = qubit_idx_list + [N] # [N] is the index of the ancilla

        # Create multi-controlled Z gate, or single Z gate if N = 1qubit. Likewise for X
        Z = qiskit.circuit.library.standard_gates.ZGate()
        X = qiskit.circuit.library.standard_gates.XGate()
        if N == 1:
            z_op = Z
            x_op = X

        else:
            z_op = Z.control(N-1)
            x_op = X.control(N)

        # Find all components with a -1 factor in i (and thus our target state vector)
        indices = np.where(w == -1)[0]
        if indices.size > 0:
            for idx in indices:
                # Need to switch qubits in the 0 state so CZ will take effect
                for i, b in enumerate(basis_labels[idx]):
                    if b == '0':
                        Sx.x((N-1)-i) # (N-1)-i is to match the qubit ordering Qiskit uses (reversed)

                Sx.append(z_op, qubit_idx_list)

                # And switch the flipped qubits back
                for i, b in enumerate(basis_labels[idx]):
                    if b == '0':
                        Sx.x((N-1)-i)

        Sx.h(qubit_idx_list)
        Sx.x(qubit_idx_list)
        Sx.append(x_op, qubit_ancilla_idx_list)

        return Sx


    def default_measurement(self):
        """
        A measurement on the ancilla qubit. If the probability of the ancilla being in the |1> is
        greater the than the threshold, output gives label 1. Otherwise, label 0.

        Default threshold is 0.5.

        Returns
            - ProbabilityThreshold Measurement object.
        """

        pt = ProbabilityThreshold(qubits=self.measurement_qubit,
                                 p_zero=False,
                                 threshold=0.5,
                                 labels=[1, 0])

        return pt
