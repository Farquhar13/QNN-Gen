from abc import ABC, abstractmethod
from qiskit import QuantumCircuit, QuantumRegister
import qiskit
import numpy as np

from Utility import Gate


class Model(ABC):
    """
    Class to define a QML model, whether it be a particular ansatz or a perceptron model.

    Abstract class. Derived classes overwrite circuit() and default_measuremnt(). Derived classes should
    also have a self.params attribute.
    """

    def __init__(self, n_qubits=None, params=None):
        """
        Attributes:
            - n_qubits [int]: Number of encoded qubits in the encoding circuit to be joined with
            the model circuit
            - params [np.ndarray]: The model's learnable parameters
        """

        self.n_qubits = self.n_qubits
        self.params = params

    def get_n_qubits(self):
        return self.n_qubits

    def set_n_qubits(self, n_qubits):
        self.n_qubits = n_qubits

    @abstractmethod
    def get_params(self):
        """
        Getter for the model's learnable parameters. This class is abstract to allow the
        attribute that holds the parameters to be renamed to be more fitting for the
        specific model. It also provides a common function to access parameters for that will
        work for any derived class.

        If the derived class uses self.params to store these parameters,
        this function can be overwritten simply as:

        return self.params
        """
        pass


    @abstractmethod
    def set_params(self, new_params):
        """
        Setter for the model's learnable parameters.
        """
        pass


    @abstractmethod
    def circuit(n_encoded_qubits, model_params, hyperparams=None):
        """
        Abstract method. Overwrite to return the circuit.

        Inputs:
            - n_encoded_qubits [int]
            - model_params
            - hyperparams

        Returns:
            - [qiskit.QuantumCircuit]: The circuit defined by the model
        """
        pass


    @abstractmethod
    def default_measurement():
        """
        Abstract method.

        Returns:
            - [Measurement]: The default measurement for the model
        """
        pass

    @staticmethod
    def print_models():
        """
        Prints the available derived classes.
        """
        
        print("TensorTreeNetwork" + "\n",
              "BinaryPerceptron")
