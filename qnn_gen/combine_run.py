from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
from qiskit import Aer
from qiskit import execute

def combine(x, encoder, model, measurement=None):
    """
    Inputs:
        - x (np.ndarray): Data example to encode
        - encoder (object derived from Encode class)
        - model (object derived from Model class)
        - measurement=None (object derived from Measurement class): If none,
        model.default_measurement will be used.

    Returns:
        - (qiksit.QuantumCircuit) The full circuit which encodes x using
        the encode object and combines the encoding, model, and measurement
        into a single circuit.
    """

    # Get circuits and measurement object
    encode_circuit = encoder.circuit(x)
    if model.n_qubits == None:
        #print("Setting model.n_qubits")
        model.n_qubits = encoder.n_qubits(x)
    model_circuit = model.circuit()
    if measurement == None:
        measurement = model.default_measurement()

    # Make sure circuits have the same number of qubits
    n_qubits_difference = int(abs(encode_circuit.num_qubits - model_circuit.num_qubits))
    if n_qubits_difference != 0:
        qr = QuantumRegister(n_qubits_difference)
        if encode_circuit.num_qubits < model_circuit.num_qubits:
            encode_circuit.add_register(qr)
        else:
            model_circuit.add_register(qr)

    # Combine circuits
    full_circuit = QuantumCircuit.compose(encode_circuit, model_circuit)

    # Rotate basis
    if measurement.rotate == True:
        measurement.rotate_basis(full_circuit)

    # Add classical registers and measurement operations
    measurement.add_measurements(full_circuit, measurement.qubits)

    return full_circuit

def run_data_point(x, encoder, model, measurement, backend, n_shots):
    full_circuit = combine(x, encoder, model, measurement)
    counts = execute(full_circuit, backend).result().get_counts(full_circuit)
    output = measurement.output(counts)
    prediction = measurement.output(counts)

    return prediction

def run(X, encoder, model,
        measurement=None,
        backend=Aer.get_backend("qasm_simulator"),
        n_shots=1024):
    """
    Determines size of dataset and calls run_data_point to get individual predictions.
    
    Inputs:
        - X (np.ndarray): Data set of examples
        - encoder (object derived from Encode class)
        - model (object derived from Model class)
        - measurement=None (object derived from Measurement class): If none,
        model.default_measurement will be used.
        - backend=Aer.get_backend("qasm_simulator") (qiskit backend):
        - n_shots=1024 (int): Number of times to run the circuit

    Returns:
        - predicitions (np.ndarray)
    """

    if measurement == None:
        measurement = model.default_measurement()

    if len(X.shape) == 1:
        # Single data example
        dataset_size = 1
        predictions = np.array(run_data_point(X, encoder, model, measurement, backend, n_shots))

    else:
        # Dataset
        dataset_size = X.shape[0]
        predictions = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            predictions[i] = run_data_point(x, encoder, model, measurement, backend, n_shots)

    return predictions
