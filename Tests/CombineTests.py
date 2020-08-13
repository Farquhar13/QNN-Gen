import unittest

# Find a better way to import
import sys
sys.path.append('../QNN-Gen/')

import Encode
import Model
import Measurement
from Combine import combine
import numpy as np
import qiskit

class TestEncode(unittest.TestCase):

    def test_combine1(self):
        x = np.array([1, 0, 0, 0, 0, 0, 0, 1])
        encoder = Encode.BasisEncoding()
        model = Model.TreeTensorNetwork()
        full_circuit = combine(x, encoder, model)

        print(full_circuit)
        self.assertTrue(isinstance(full_circuit, qiskit.QuantumCircuit))

    def test_combine2(self):
        x = np.array([0.75, 0, 0.25, 0])
        encoder = Encode.AngleEncoding()
        model = Model.EntangledQubit()
        full_circuit = combine(x, encoder, model)

        print(full_circuit)
        self.assertTrue(isinstance(full_circuit, qiskit.QuantumCircuit))

    def test_combine3(self):
        x = np.array([-1, 1, 1, -1])
        encoder = Encode.BinaryPhaseEncoding()
        model = Model.BinaryPerceptron()
        full_circuit = combine(x, encoder, model)

        print(full_circuit)
        self.assertTrue(isinstance(full_circuit, qiskit.QuantumCircuit))

if __name__ == "__main__":
    unittest.main()
