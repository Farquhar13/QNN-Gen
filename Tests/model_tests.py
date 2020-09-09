import unittest
import sys
sys.path.append('../QNN-Gen/')

from qnn_gen import model
import qiskit

class TestModel(unittest.TestCase):

    def test_TTN_circuit(self):
        ttn = model.TreeTensorNetwork(n_qubits=8)
        ttn_circuit = ttn.circuit()
        self.assertTrue(isinstance(ttn_circuit, qiskit.QuantumCircuit))

    def test_BP_circuit(self):
        bp = model.BinaryPerceptron(n_qubits=8)
        circuit = bp.circuit()
        self.assertTrue(isinstance(circuit, qiskit.QuantumCircuit))

    def test_ET(self):
        et = model.EntangledQubit(n_qubits=8)
        circuit = et.circuit()
        self.assertTrue(isinstance(circuit, qiskit.QuantumCircuit))

if __name__ == "__main__":
    unittest.main()
