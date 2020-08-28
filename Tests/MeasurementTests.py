import unittest

import sys
sys.path.append('../QNN-Gen/')
import numpy as np
import qiskit
from Measurement import *
import Utility
from math import isclose
from Observable import Observable

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
        self.assertTrue(counts['1'] == 1024)

    def test_rotate_basis1(self):
        """ Z """
        qc = qiskit.QuantumCircuit(1)
        Z = Observable.Z()
        Measurement.rotate_basis(qc, [0], Z)
        sv = Utility.get_state_vector(qc)
        self.assertTrue(np.allclose([1, 0], sv))

    def test_rotate_basis2(self):
        """ X """
        qc = qiskit.QuantumCircuit(1)
        qc.h(0) # Rotate onto eigenvector of x w/ e.value 1
        X = Observable.X()
        Measurement.rotate_basis(qc, [0], X)
        sv = Utility.get_state_vector(qc)

        qc1 = qiskit.QuantumCircuit(1)
        qc1.x(0)
        qc1.h(0) # Rotate onto eigenvector of x with e.value -1
        Measurement.rotate_basis(qc1, [0], X)
        sv1 = Utility.get_state_vector(qc1)
        result = np.allclose([1, 0], sv) and np.allclose([0, 1], sv1)
        self.assertTrue(result)

    def test_rotate_basis3(self):
        """ Y """
        qc = qiskit.QuantumCircuit(1)
        qc.rx(-np.pi/2, 0) # Rotate onto y axis
        Y = Observable.Y()
        Measurement.rotate_basis(qc, [0], Y)
        sv = Utility.get_state_vector(qc)

        qc1 = qiskit.QuantumCircuit(1)
        qc1.rx(np.pi/2, 0) # Rotate onto negative y axis
        Measurement.rotate_basis(qc1, [0], Y)
        sv1 = Utility.get_state_vector(qc1)

        result = np.allclose([1, 0], sv) and np.allclose([0, 1], sv1)
        self.assertTrue(result)

    def test_rotate_basis4(self):
        """
        Make sure eigenvlaues of H Observable get rotated to computational
        basis by rotate basis.
        """
        qc = qiskit.QuantumCircuit(1)
        H = Observable.H()
        qc.initialize(H.eigenvectors[0], [0])
        Measurement.rotate_basis(qc, [0], H)
        sv = Utility.get_state_vector(qc)

        qc1 = qiskit.QuantumCircuit(1)
        qc1.initialize(H.eigenvectors[1], [0])
        Measurement.rotate_basis(qc1, [0], H)
        sv1 = Utility.get_state_vector(qc1)
        result = np.allclose([1, 0], sv) and (np.allclose([0, 1], sv1))
        self.assertTrue(result)

    def test_rotate_basis5(self):
        """ Testing arbitary observable functionality (X) """
        X = np.array([[0, 1],
                      [1, 0]])
        X_obs = Observable(X)

        qc = qiskit.QuantumCircuit(1)
        qc.h(0)
        Measurement.rotate_basis(qc, [0], X_obs)
        sv = Utility.get_state_vector(qc)

        qc1 = qiskit.QuantumCircuit(1)
        qc1.x(0)
        qc1.h(0) # Rotate onto eigenvector of x with e.value -1
        Measurement.rotate_basis(qc1, [0], X_obs)
        sv1 = Utility.get_state_vector(qc1)

        expected = np.allclose(Measurement.born_rule(sv[0]), 1)
        expected1 = np.allclose(Measurement.born_rule(sv1[1]), 1)
        self.assertTrue(expected and expected1)

    def test_rotate_basis6(self):
        """ Testing arbitary observable functionality (Y) """
        Y = np.array([[0, -1j],
                      [1j, 0]])
        Y_obs = Observable(Y)
        qc = qiskit.QuantumCircuit(1)
        qc.rx(-np.pi/2, [0])
        Measurement.rotate_basis(qc, [0], Y_obs)
        sv = Utility.get_state_vector(qc)

        qc1 = qiskit.QuantumCircuit(1)
        qc1.rx(np.pi/2, 0) # Rotate onto negative y axis
        Measurement.rotate_basis(qc1, [0], Y_obs)
        sv1 = Utility.get_state_vector(qc1)

        expected = np.allclose(Measurement.born_rule(sv[0]), 1)
        expected1 = np.allclose(Measurement.born_rule(sv1[1]), 1)
        self.assertTrue(expected and expected1)

    def test_probability1(self):
        """ Default arguments. Note: test may fail probabilistically when there is no error."""
        qc = qiskit.QuantumCircuit(1)
        qc.h(0)

        Measurement.add_measurements(qc, [0])
        counts = Utility.get_counts(qc, measure_all=False)

        Prob = Probability(0)
        result = Prob.output(counts)
        self.assertTrue(isclose(0.5, result[0], abs_tol=5e-2))

    def test_probability2(self):
        """ p_zero=False """
        qc = qiskit.QuantumCircuit(1)
        qc.x(0)

        Measurement.add_measurements(qc, [0])
        counts = Utility.get_counts(qc, measure_all=False)

        Prob = Probability([0], p_zero=False)
        result = Prob.output(counts)
        self.assertTrue(1 == result[0])

    def test_probability3(self):
        """ observable_basis. Note: test may fail probabilistically when there is no error. """
        qc = qiskit.QuantumCircuit(1)
        X_obs = Observable.X()
        prob = Probability([0], observable=X_obs)
        prob.rotate_basis(qc)
        counts = Utility.get_counts(qc)
        result = prob.output(counts)
        self.assertTrue(isclose(0.5, result[0], abs_tol=5e-2))

    def test_probability_threshold1(self):
        qc = qiskit.QuantumCircuit(1)
        counts = Utility.get_counts(qc)
        pt = ProbabilityThreshold(0)
        result = pt.output(counts)
        self.assertTrue(result[0] == 0)

    def test_probability_threshold2(self):
        """ p_zero=False w/ labels """
        qc = qiskit.QuantumCircuit(1)
        counts = Utility.get_counts(qc)
        pt = ProbabilityThreshold(0, p_zero=False, labels=['a', 'b'])
        result = pt.output(counts)
        self.assertTrue(result[0] == 'b')

    def test_probability_threshold3(self):
        """ threshold """
        qc = qiskit.QuantumCircuit(1)
        qc.h(0)
        counts = Utility.get_counts(qc)
        pt = ProbabilityThreshold(0, threshold=0.75)
        result = pt.output(counts)
        self.assertTrue(result[0] == 1)

    def test_probability_threshold4(self):
        """ rotate_basis """
        qc = qiskit.QuantumCircuit(1)
        qc.x(0)
        qc.h(0)
        X_obs = Observable.X()
        pt = ProbabilityThreshold(0, observable=X_obs)
        pt.rotate_basis(qc)
        counts = Utility.get_counts(qc)
        result = pt.output(counts)
        self.assertTrue(result[0] == 1)

    def test_expectation1(self):
        """ Z """
        qc = qiskit.QuantumCircuit(1)
        exp = Expectation(0)
        counts = Utility.get_counts(qc)
        result = exp.output(counts)
        expected = [1]
        is_correct = (result == expected)

        qc1 = qiskit.QuantumCircuit(1)
        qc1.x(0)
        counts1 = Utility.get_counts(qc1)
        result1 = exp.output(counts1)
        expected1 = [-1]
        is_correct1 = (result1 == expected1)
        self.assertTrue(is_correct and is_correct1)

    def test_expectation2(self):
        """ Y """
        qc = qiskit.QuantumCircuit(1)
        qc.rx(-np.pi/2, 0)
        Y_obs = Observable.Y()
        exp = Expectation(0, observable=Y_obs)
        exp.rotate_basis(qc)
        counts = Utility.get_counts(qc)
        result = exp.output(counts)
        expected = [1]
        is_correct = (result == expected)

        qc1 = qiskit.QuantumCircuit(1)
        qc1.rx(np.pi/2, 0)
        exp.rotate_basis(qc1)
        counts1 = Utility.get_counts(qc1)
        result1 = exp.output(counts1)
        expected1 = [-1]
        is_correct1 = (result1 == expected1)
        self.assertTrue(is_correct and is_correct1)

    def test_expectation2(self):
        """ qubit """
        qc = qiskit.QuantumCircuit(2)
        qc.h(0)
        exp = Expectation(1)
        exp.rotate_basis(qc)
        counts = Utility.get_counts(qc)
        result = exp.output(counts)
        expected = [1]
        is_correct = (result == expected)

if __name__ == "__main__":
    unittest.main()
