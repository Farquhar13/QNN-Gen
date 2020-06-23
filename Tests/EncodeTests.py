import unittest

# Find a better way to import
import sys
sys.path.append('../QNN-Gen/')

import Encode
import numpy as np
import qiskit
from Utility import get_state_vector
from Utility import Gate
from Utility import get_counts
from functools import reduce

class TestEncode(unittest.TestCase):

    def test_angle_encoding1(self):
        """ n_qubits """
        ae = Encode.AngleEncoding()
        x = np.arange(10)
        n_qubits = ae.n_qubits(x)

        self.assertEqual(10, n_qubits, 'Error in n_qubits')

    def test_angle_encoding2(self):
        """ Default arguments """
        ae = Encode.AngleEncoding()
        x = np.random.random(4)

        psi = ae.state_vector(x)
        circuit = ae.circuit(x)
        circuit_psi = get_state_vector(circuit)

        is_equal = np.allclose(psi, circuit_psi)

        self.assertTrue(is_equal)

    def test_angle_encoding3(self):
        """ Testing scaling factor"""

        random_scaling = np.random.uniform(0, np.pi)
        ae = Encode.AngleEncoding(scaling=random_scaling)
        x = np.random.random(4)

        psi = ae.state_vector(x)
        circuit = ae.circuit(x)
        circuit_psi = get_state_vector(circuit)

        is_equal = np.allclose(psi, circuit_psi)

        self.assertTrue(is_equal)

    def test_angle_encoding4(self):
        """ Testing different gates """

        #gate_set = [Gate.RX, Gate.RZ, Gate.U1]
        # RZ has no .to_matrix() ???

        gate_set = [Gate.RX, Gate.U1]

        ae = Encode.AngleEncoding()
        x = np.random.random(4)

        is_equal_list = []
        for gate in gate_set:
            ae.gate = gate
            psi = ae.state_vector(x)
            circuit = ae.circuit(x)
            circuit_psi = get_state_vector(circuit)

            is_equal_list.append(np.allclose(psi, circuit_psi))

        self.assertTrue(is_equal_list)


    def test_dense_angle_encoding1(self):
        """ n_qubits """
        dae = Encode.DenseAngleEncoding()
        x = np.arange(10)
        n_qubits = dae.n_qubits(x)

        self.assertEqual(5, n_qubits, 'Error in n_qubits')


    def test_dense_angle_encoding2(self):
        """ Default arguments """
        dae = Encode.DenseAngleEncoding()
        x = np.random.random(4)

        psi = dae.state_vector(x)
        circuit = dae.circuit(x)
        circuit_psi = get_state_vector(circuit)

        is_equal = np.allclose(psi, circuit_psi)

        self.assertTrue(is_equal)

    def test_dense_angle_encoding3(self):
        """ Testing scaling factor"""

        random_scaling = np.random.uniform(0, np.pi/2)
        dae = Encode.DenseAngleEncoding(scaling=random_scaling)
        x = np.random.random(4)

        psi = dae.state_vector(x)
        circuit = dae.circuit(x)
        circuit_psi = get_state_vector(circuit)

        is_equal = np.allclose(psi, circuit_psi)

        self.assertTrue(is_equal)

    def test_dense_angle_encoding4(self):
        """ Testing different gates """

        #gate_set = [Gate.RX, Gate, RZ, Gate.U1]
        # RZ has no .to_matrix() ???

        gate_set = [Gate.RX, Gate.U1]

        dae = Encode.DenseAngleEncoding()
        x = np.random.random(4)

        is_equal_list = []
        for gate in gate_set:
            dae.gate = gate
            psi = dae.state_vector(x)
            circuit = dae.circuit(x)
            circuit_psi = get_state_vector(circuit)

            is_equal_list.append(np.allclose(psi, circuit_psi))

            all_true = reduce(lambda a, b: (a and b), is_equal_list)

        self.assertTrue(all_true)


    def test_binary_phase_encoding1(self):
        """ n_qubits ancilla=False"""
        elements = [-1, 1]
        x = np.random.choice(elements, 8)
        bp = Encode.BinaryPhaseEncoding(ancilla=False)
        n_qubits = bp.n_qubits(x)

        self.assertEqual(3, n_qubits, 'Error in n_qubits')

    def test_binary_phase_encoding2(self):
        """ n_qubits ancilla=True"""
        elements = [-1, 1]
        x = np.random.choice(elements, 8)
        bp = Encode.BinaryPhaseEncoding(ancilla=True)
        n_qubits = bp.n_qubits(x)

        self.assertEqual(4, n_qubits, 'Error in n_qubits')

    def test_binary_phase_encoding3(self):
        """ Default Arguemnets ancilla=False"""
        bp = Encode.BinaryPhaseEncoding(ancilla=False)
        elements = [-1, 1]
        equals_list = []
        for _ in range(5):
            x = np.random.choice(elements, 8)
            circuit = bp.circuit(x)
            circuit_psi = get_state_vector(circuit)
            psi = 1/np.sqrt(8) * x
            equals_list.append(np.allclose(psi, circuit_psi))

        all_true = reduce(lambda a, b: (a and b), equals_list)
        self.assertTrue(all_true)

    def test_binary_phase_encoding4(self):
        """ Default Arguemnets ancilla=True (default)"""
        bp = Encode.BinaryPhaseEncoding()
        elements = [-1, 1]
        equals_list = []
        zero_state = np.array([1, 0])
        for _ in range(1):
            x = np.random.choice(elements, 8)
            circuit = bp.circuit(x)
            circuit_psi = get_state_vector(circuit)
            psi = np.kron(zero_state, (1/np.sqrt(8) * x))
            equals_list.append(np.allclose(psi, circuit_psi))

        all_true = reduce(lambda a, b: (a and b), equals_list)
        self.assertTrue(all_true)


    def test_basis_encoding1(self):
        """ n_qubits """

        x = [1, 0, 0, 0, 1, 1, 1]
        basis = Encode.BasisEncoding()
        n_qubits = basis.n_qubits(x)

        self.assertEqual(len(x), n_qubits, 'Error in n_qubits')

    def test_basis_encoding2(self):
        """ Default """

        x = [1, 0, 0, 0, 1, 1, 1]
        basis = Encode.BasisEncoding()
        circuit = basis.circuit(x)
        counts = get_counts(circuit)

        key = None
        value = None
        for k,v in counts.items():
            key = k

        expected_dirac_label = '1000111'

        self.assertEqual(len(counts.items()), 1)
        self.assertEqual(key, expected_dirac_label)

    def test_basis_encoding3(self):
        """ Input validation """
        pass


    """
    def test_list(self):
        l = [False, True]

        all_true = reduce(lambda a, b: (a and b), l)
        self.assertTrue(all_true)
    """


if __name__ == "__main__":
    unittest.main()
