import unittest
import sys
sys.path.append('../QNN-Gen/')

import Model
import qiskit
class TestModel(unittest.TestCase):

    def test_TTN_circuit(self):
        ttn = Model.TensorTreeNetwork(n_qubits=4)
        ttn_circuit = ttn.circuit()
        self.assertTrue(isinstance(ttn_circuit, qiskit.QuantumCircuit))

if __name__ == "__main__":
    unittest.main()
