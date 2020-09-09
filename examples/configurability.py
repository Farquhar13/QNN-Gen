"""
Example to demonstrate how modeling decisions can be implemented by passing
arguments to overwride default values for class attributes.

This code implements
    - A Binary Phase encoding with an ancillary qubit
    - A Binary Perceptron model with specified starting weights
    - A Probability Threshold measurement that measures qubit 2, and
    returns the lablel "1" if the qubit is measured in |1> state with
    probability greater that 0.3, and "-1" otherwise. The measurement is
    performed with respect to the obsevable Pauli X.
"""

import qnn_gen as qg
import numpy as np

x = np.array([-1, 1, 1, -1])
weights = np.array([1, -1, 1, -1])

encoder = qg.BinaryPhaseEncoding(ancilla=True)
model = qg.BinaryPerceptron(weights=weights)
measurement = qg.ProbabilityThreshold(qubits=2,
                                      p_zero=False,
                                      threshold=0.3,
                                      labels=[1, -1],
                                      observable=qg.Observable.X())

full_circuit = qg.combine(x, encoder, model, measurement)
result = qg.run(x, encoder, model, measurement)

print(full_circuit)
print(result)
