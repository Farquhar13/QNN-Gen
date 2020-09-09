import unittest

# Find a better way to import
import sys
sys.path.append('../QNN-Gen/')

from qnn_gen import encode
from qnn_gen.model import TreeTensorNetwork, EntangledQubit, BinaryPerceptron
from qnn_gen import measurement
from qnn_gen.observable import Observable
from qnn_gen.combine_run import combine
from qnn_gen.combine_run import run
import numpy as np
import qiskit

class Testencode(unittest.TestCase):

    def test_combine1(self):
        x = np.array([1, 0, 0, 0, 0, 0, 0, 1])
        encoder = encode.BasisEncoding()
        model = TreeTensorNetwork()
        full_circuit = combine(x, encoder, model)

        print(full_circuit)
        self.assertTrue(isinstance(full_circuit, qiskit.QuantumCircuit))

    def test_combine2(self):
        x = np.array([0.75, 0, 0.25, 0])
        encoder = encode.AngleEncoding()
        model = EntangledQubit()
        full_circuit = combine(x, encoder, model)

        print(full_circuit)
        self.assertTrue(isinstance(full_circuit, qiskit.QuantumCircuit))

    def test_combine3(self):
        """ BinaryPhase, BinaryPerceptron """
        x = np.array([-1, 1, 1, -1])
        encoder = encode.BinaryPhaseEncoding()
        model = BinaryPerceptron()
        full_circuit = combine(x, encoder, model)

        print(full_circuit)
        self.assertTrue(isinstance(full_circuit, qiskit.QuantumCircuit))

    def test_combine4(self):
        """ With BinaryEncoding and a different model """
        x = np.array([-1, 1, 1, -1])
        encoder = encode.BinaryPhaseEncoding(ancilla=False)
        model = TreeTensorNetwork()

        full_circuit = combine(x, encoder, model)

        print(full_circuit)
        self.assertTrue(isinstance(full_circuit, qiskit.QuantumCircuit))

    def test_combine5(self):
        """ With BinaryEncoding and a different model """
        x = np.array([-1, 1, 1, -1])
        encoder = encode.BinaryPhaseEncoding()
        model = TreeTensorNetwork()

        full_circuit = combine(x, encoder, model)

        print(full_circuit)
        self.assertTrue(isinstance(full_circuit, qiskit.QuantumCircuit))

    def test_combine6(self):
        """ circuits with differnent number of qubits """
        x = np.array([0.5, 0, 0, -0.5])
        encoder = encode.DenseAngleEncoding()
        model = BinaryPerceptron()

        full_circuit = combine(x, encoder, model)

        print(full_circuit)
        self.assertTrue(isinstance(full_circuit, qiskit.QuantumCircuit))

    def test_combine7(self):
        """ With measurement argument --- Probability """
        x = np.array([1, 0, 0, 1])
        encoder = encode.BasisEncoding()
        model = TreeTensorNetwork()
        X_obs = Observable.X()
        measure = measurement.Probability(0, observable=X_obs)
        full_circuit = combine(x, encoder, model, measure)

        print(full_circuit)
        self.assertTrue(isinstance(full_circuit, qiskit.QuantumCircuit))

    def test_combine8(self):
        """ With measurement argument --- Expecation """
        x = np.array([1, 0, 0, 1])
        encoder = encode.BasisEncoding()
        model = TreeTensorNetwork()
        Y_obs = Observable.Y()
        measure = measurement.Expectation(0, observable=Y_obs)
        full_circuit = combine(x, encoder, model, measure)

        print(full_circuit)
        self.assertTrue(isinstance(full_circuit, qiskit.QuantumCircuit))

    def test_combine9(self):
        """ With measurement argument --- ProbabilityThreshold """
        x = np.array([1, 0, 0, 1])
        encoder = encode.BasisEncoding()
        model = TreeTensorNetwork()
        measure = measurement.ProbabilityThreshold(3)
        full_circuit = combine(x, encoder, model, measure)

        print(full_circuit)
        self.assertTrue(isinstance(full_circuit, qiskit.QuantumCircuit))

    def test_run(self):
        X = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [1, 0, 0, 0]])

        encoder = encode.BasisEncoding()
        model = TreeTensorNetwork()
        measure = measurement.Probability(0)
        predictions = run(X, encoder, model, measure)
        right_length = (len(predictions) == len(X))
        right_type = isinstance(predictions, np.ndarray)

        self.assertTrue(right_length and right_type)

if __name__ == "__main__":
    unittest.main()
