import unittest
import sys
sys.path.append('../QNN-Gen/')

import Model
import qiskit
from Utility import Gate
import numpy as np

ttn = Model.TreeTensorNetwork(n_qubits=4, rotation_gate=Gate.RZ, entangling_gate=Gate.CY)
ttn_circuit = ttn.circuit()
print(ttn_circuit)

x = Gate.X
print(x.to_matrix())

rzx = Gate.RZX(np.pi)
print(rzx.to_matrix())

rxx = Gate.RXX(np.pi/2)
print(rxx.to_matrix())
