import unittest
import sys
sys.path.append('../QNN-Gen/')

from qnn_gen import model
import qiskit

class TestModel(unittest.TestCase):

    def test_TTN_circuit(self):
        ttn = model.TreeTensorNetwork(n_qubits=2)
        ttn_circuit = ttn.circuit()
        self.assertTrue(isinstance(ttn_circuit, qiskit.QuantumCircuit))

    def test_TTN_circuit1(self):
        ttn = model.TreeTensorNetwork(n_qubits=4)
        ttn_circuit = ttn.circuit()
        self.assertTrue(isinstance(ttn_circuit, qiskit.QuantumCircuit))

    def test_TTN_circuit2(self):
        ttn = model.TreeTensorNetwork(n_qubits=8)
        ttn_circuit = ttn.circuit()
        self.assertTrue(isinstance(ttn_circuit, qiskit.QuantumCircuit))

    def test_TTN_circuit3(self):
        ttn = model.TreeTensorNetwork(n_qubits=16)
        ttn_circuit = ttn.circuit()
        self.assertTrue(isinstance(ttn_circuit, qiskit.QuantumCircuit))

    def test_BP_circuit(self):
        bp = model.BinaryPerceptron(n_qubits=2)
        circuit = bp.circuit()
        self.assertTrue(isinstance(circuit, qiskit.QuantumCircuit))

    def test_BP_circuit1(self):
        bp = model.BinaryPerceptron(n_qubits=4)
        circuit = bp.circuit()
        self.assertTrue(isinstance(circuit, qiskit.QuantumCircuit))

    def test_BP_circuit2(self):
        bp = model.BinaryPerceptron(n_qubits=8)
        circuit = bp.circuit()
        self.assertTrue(isinstance(circuit, qiskit.QuantumCircuit))

    def test_ET(self):
        et = model.EntangledQubit(n_qubits=2, n_layers=2)
        circuit = et.circuit()
        self.assertTrue(isinstance(circuit, qiskit.QuantumCircuit))

    def test_ET1(self):
        et = model.EntangledQubit(n_qubits=4, n_layers=2)
        circuit = et.circuit()
        self.assertTrue(isinstance(circuit, qiskit.QuantumCircuit))

    def test_ET2(self):
        et = model.EntangledQubit(n_qubits=8, n_layers=2)
        circuit = et.circuit()
        self.assertTrue(isinstance(circuit, qiskit.QuantumCircuit))

if __name__ == "__main__":
    unittest.main()
