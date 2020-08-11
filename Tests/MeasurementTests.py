import unittest

import sys
sys.path.append('../QNN-Gen/')
import numpy as np
import qiskit
import Measurement
from Measurement import Measurement
import Utility

class TestMeasurement(unittest.TestCase):

    # ----------------> test base class <----------------

    def test_statevector_to_probability(self):
        psi = np.array([0, 1j, 1j, 0]) * 1/np.sqrt(2)
        probs = Measurement.statevector_to_probability(psi)
        expected = [0, 0.5, 0.5, 0]

        self.assertTrue(np.allclose(probs, expected))

    def test_statevector_to_probability_dict(self):
        psi = np.array([1j,0,0,1j]) * 1/np.sqrt(2)
        probs_dict = Measurement.statevector_to_probability_dict(psi)
        bit_strings = Measurement.get_bit_strings(2)
        probs = [probs_dict[bit_string] for bit_string in bit_strings]
        expected = [0.5, 0, 0, 0.5]

        self.assertTrue(np.allclose(probs, expected))

    def test_counts_to_probability(self):
        qc = qiskit.QuantumCircuit(3)
        qc.x(0)
        qc.y(1)
        qc.z(2)
        counts = Utility.get_counts(qc)

        meas_counts = Measurement.counts_to_probability(counts)

        correct_len = (len(counts) == len(meas_counts))
        correct_type = isinstance(meas_counts, dict)

        self.assertTrue( (correct_len and correct_type) )

    def test_counts_for_qubits(self):
        qc = qiskit.QuantumCircuit(3)
        qc.x(0)
        qc.y(1)
        qc.z(2)
        counts = Utility.get_counts(qc)

        qubits = [1, 2]
        qubit_counts = Measurement.counts_for_qubits(counts, qubits)

        correct_len = (len(qubit_counts) == 2**(len(qubits)))
        correct_type = isinstance(qubit_counts, dict)

        self.assertTrue( (correct_len and correct_type) )

    def test_add_measurements1(self):
        """ Test on circuit initially without classical register """
        qc = qiskit.QuantumCircuit(1)
        qc.x(0)

        Measurement.add_measurements(qc, [0])
        counts = Utility.get_counts(qc, measure_all=False)
        self.assertTrue(counts['1'] == 1024)

    def test_add_measurements2(self):
        """ Test on circuit with an existing classical register. Should not
        overwrite measurements on existing classial registers. """
        qc = qiskit.QuantumCircuit(2, 1)
        qc.x(1)
        qc.measure(0, 0)

        Measurement.add_measurements(qc, [1])
        counts = Utility.get_counts(qc, measure_all=False)
        self.assertTrue(counts['1 0'] == 1024) # space separates different classical registers

    def test_add_measurements3(self):
        """ Test with clbit argument """
        qc = qiskit.QuantumCircuit(2)
        qc.x(0)

        Measurement.add_measurements(qc, [0, 1], [1, 0])
        counts = Utility.get_counts(qc, measure_all=False)
        self.assertTrue(counts['10'] == 1024)

    def test_add_measurements4(self):
        """ Test with add_classical_register=False """
        qc = qiskit.QuantumCircuit(2, 1)
        qc.x(0)

        Measurement.add_measurements(qc, [0], [0], add_classical_register=False)
        counts = Utility.get_counts(qc, measure_all=False)
        print(qc)
        print(counts)
        self.assertTrue(counts['1'] == 1024)

if __name__ == "__main__":
    unittest.main()
