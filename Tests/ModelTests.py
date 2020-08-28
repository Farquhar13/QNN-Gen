import unittest
import sys
sys.path.append('../QNN-Gen/')

import Model
import qiskit

class TestModel(unittest.TestCase):

    def test_TTN_circuit(self):
        ttn = Model.TreeTensorNetwork(n_qubits=8)
        ttn_circuit = ttn.circuit()
        self.assertTrue(isinstance(ttn_circuit, qiskit.QuantumCircuit))

    def test_BP_circuit(self):
        bp = Model.BinaryPerceptron(n_qubits=8)
        circuit = bp.circuit()
        self.assertTrue(isinstance(circuit, qiskit.QuantumCircuit))

    def test_ET(self):
        et = Model.EntangledQubit(n_qubits=8)
        circuit = et.circuit()
        #print(circuit)
        self.assertTrue(isinstance(circuit, qiskit.QuantumCircuit))


if __name__ == "__main__":
    unittest.main()
