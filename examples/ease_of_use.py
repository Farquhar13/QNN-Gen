"""
An example to illustrate ease-of-use.

Generate a Quantum Neural Network in just a few lines of code.

This code implements:
    - A Dense Angle encoding
    - An Entangled Qubit model initialized with random angles (parameters)
    - Since no measurement is passed to qg.combine the default_measurement()
    of EntangledQubit is called. The default measurement corresponds to measuring
    the Expectation of the Z observable.
"""

import qnn_gen as qg
import numpy as np

x = np.random.rand(8)

encoder = qg.DenseAngleEncoding()
model = qg.EntangledQubit()
full_circuit = qg.combine(x, encoder, model)

print(full_circuit)
result = qg.run(x, encoder, model)
print(result)
