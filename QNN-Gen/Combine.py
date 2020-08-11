from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

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

    encode_circuit = encoder.circuit(x)
    if model.n_qubits == None:
        model.n_qubits = encoder.n_qubits
    model_circuit = model.circuit()
    if measurement == None:
        measurement = model.default_measuremnt()

    # Combine circuits
    full_circuit = qiskit.combine(encode_circuit, model_circuit)

    # Rotate basis
    if measurement.rotate_basis == True:
        measument.rotate_basis(full_circuit)

    # Add classical registers and measurement operations
    measurement.add_measurements(full_circuit, measurement.qubits)

    return full_circuit
