import numpy as np
import qiskit
from abc import ABC, abstractmethod
from .observable import Observable

class Measurement(ABC):
    """
    Abstract class with one abstract method, output(). Derived classes must overwrite this method. This class serves
    as the output layer. The output function takes the results of running a circuit transforms the results into the
    final output.

    To print the default Derived classes you can call Measurement.print_derived_classes().
    """

    def __init__(self, qubits, rotate=False):
        """
        Attributes:
            - self.qubits (list or np.array): The qubits to be measured
            - self.rotate=False (boolean): True if the measurement is performed
            with respect to a basis other than the computational basis
        """
        self.qubits = qubits
        self.rotate = rotate

    @abstractmethod
    def output(counts):
        """
        Overwrite this method in dervided classes.

        Input:
            - counts (dict): The result of running the circuit

        Returns:
            - (np.ndarray): Model output after measurement tranformations
        """
        pass

    @staticmethod
    def born_rule(amplitude):
        """
        Input:
            - amplitude (float): complex amplitude
        Returns:
            - |amplitude|^2 (float): The probability of being measured in the state
            corresponding to the amplitude
        """

        return abs(amplitude)**2

    @staticmethod
    def statevector_to_probability(psi):
        """
        Input:
            - psi (np.ndarray):
        Returns:
            - (np.ndarray) of probabilities
        """

        probs = np.array([Measurement.born_rule(amp) for amp in psi])

        return probs

    @staticmethod
    def statevector_to_probability_dict(psi):
        """
        Input:
            - psi (np.ndarray):
        Returns:
            - Dictionary of measurment outcomes as keys and their corresponding probabilities as values
        """

        n_qubits = int(np.log2(len(psi)))
        bit_strings =  Measurement.get_bit_strings(n_qubits)

        prob_dict = {key: Measurement.born_rule(value) for key, value in zip(bit_strings, psi)}

        return prob_dict


    @staticmethod
    def counts_to_probability(counts):
        """
        Inputs:
            - counts (dict): Of the form {"measurement result": count}
        Returns:
            - (dict): Of the form {"measurement result": probability}
        """

        n_shots = sum(counts.values())

        prob_dict = {key: value/n_shots for key, value in counts.items()}

        return prob_dict


    @staticmethod
    def counts_for_qubit_subset(counts, qubits, bit_strings, qubit_dict):
        """
        counts_for_qubits() calls this function if the number of measured qubits is greater
        than the length of the qubits argument. I.e. more qubits were measured than we are
        interested in getting the counts for.

        Inputs:
            - counts (dict): Dictionary with counts as values and corresponding measurement
            results as keys.
            - qubits Union(list or int): qubit index or list of qubit indices

        Returns:
            - A dictionary with the keys being all possible bit strings corresponding to the sorted
            qubit indices and values being the counts for the measument outcomes for those qubit results.

        Example:
            - qubits=[0, 4] would return a dictionary of {'00': a, '01': b, '10': c, '11': d}
            Where the bit string is the state of qubit 4 (left) and qubit 0 (right), and the values
            are the counts.
        """
        sorted_qubits = list(reversed(sorted(qubits)))
        for key, value in counts.items():
            # Look at only the revelant qubits
            reverse_position = len(key) - 1
            qubit_key = "".join([key[reverse_position - sq] for sq in sorted_qubits])

            qubit_dict[qubit_key] += value

        return qubit_dict

    @staticmethod
    def counts_for_qubits(counts, qubits):
        """
        Inputs:
            - counts (dict): Dictionary with counts as values and corresponding measurement
            results as keys.
            - qubits Union(list or int): qubit index or list of qubit indices.

        Returns:
            - A dictionary with the keys being all possible bit strings corresponding to the sorted
            qubit indices and values being the counts for the measument outcomes for those qubit results.

        Example:
            - qubits=[0, 4] would return a dictionary of {'00': a, '01': b, '10': c, '11': d}
            Where the bit string is the state of qubit 4 (left) and qubit 0 (right), and the values
            are the counts.
        """
        if isinstance(qubits, int):
            qubits = [qubits]

        bit_strings = Measurement.get_bit_strings(len(qubits))

        # Create qubit dictionary
        qubit_dict = {}
        for s in bit_strings:
            qubit_dict[s] = 0

        n_measured_qubits = len(list(counts.keys())[0])
        if n_measured_qubits > len(qubits):
            return Measurement.counts_for_qubit_subset(counts, qubits,
                                                       bit_strings, qubit_dict)
        for key, value in counts.items():
            qubit_dict[key] += value

        return qubit_dict


    @staticmethod
    def rotate_basis(circuit, qubit, observable):
        """
        Note: modifies the circuit in-place

        Input:
        - circuit (qiskit.QuantumCircuit)
        - qubit (int): qubit index
        - observable (Observable)

        Description:
            Appends the necessary gates to the circuit so that eigenbasis of the observable is rotated
        to the computational basis.
        """

        if isinstance(qubit, int):
            qubit = [qubit]

        if observable.name == "X":
            circuit.h(qubit)

        elif observable.name == "Y":
            # Alternative is the commented version
            #circuit.z(qubit)
            #circuit.s(qubit)
            #circuit.h(qubit)
            circuit.sdg(qubit)
            circuit.h(qubit)

        elif observable.name == "Z":
            pass

        elif observable.name == "H":
            circuit.ry(np.pi/4, qubit)

        else:
            circuit.unitary(observable.eigenvectors.T.conj(), qubit)


    @staticmethod
    def get_bit_strings(n_qubits):
        """
        Input:
            - n_qubits (int):

        Returns:
            - A list of all combinations of bit strings of length n_qubits
        """
        # Generate all bit strings
        bit_strings = []

        for k in range(2**n_qubits):
            bit_strings.append(("{:0%db}" % n_qubits).format(k))

        return bit_strings

    @staticmethod
    def add_measurements(circuit, qubits=None, clbits=None, add_classical_register=True):
        """
        Adds measurements operations to circuit in-place. If no argument is passed for clbits
        and no classical register was added, the measurement result for the qubit at position
        i in the qubits list will be stored in the classical bit i. If no arguement is passed
        for clbits and a classical register was added the measurent result of qubit at list
        position i will be found in classical bit
        (i + number of classical bits originally in the circuit). This prevents overwritting
        existing classical bits.

        Inputs:
            - circuit (qiskit.QuantumCircuit)
            - qubits=None (list): Qubit indices to measure. If none it is the ordered range of all qubits.
            - clbits=None (list): Classical bit indices to hold the labeled outcome of qubit measurements.
            - add_classical_register=True (boolean): If true, adds a classical register of size
            len(qubits) to circuit
        """
        if qubits == None:
            qubits = list(range(circuit.num_qubits))

        n_measurements = len(qubits)
        n_existing_clbits = circuit.num_clbits
        if add_classical_register == True:
            cr = qiskit.ClassicalRegister(n_measurements)
            circuit.add_register(cr)

        if clbits == None:
            clbits = list(range(n_existing_clbits, n_existing_clbits + n_measurements))
            circuit.measure(qubits, clbits)
        else:
            circuit.measure(qubits, clbits)

    @staticmethod
    def print_derived_classes():
        print("Probability()")
        print("Expectation()")
        print("ProbabilityThreshold()")


class Probability(Measurement):
    """
    A class to transform circuit outputs into qubit probabilites.
    """

    def __init__(self, qubits, p_zero=True, observable=None):
        """
        Attributes:
            - qubits Union(int, list, np.ndarray): qubit index or list of qubit indices
            - observable_basis (Observable): The observable corresponding the basis to measure in
            - zero=True (Boolean): If True, output returns probabilties of qubit being measured in the |0> state.
            If false, output returns probabilties of qubit being measured in the |1> state.
        """

        if isinstance(qubits, int):
            self.qubits = [qubits]
        else:
            self.qubits = qubits
        self.p_zero = p_zero
        self.observable = observable
        if observable is not None:
            requires_rotation = True
        else:
            requires_rotation = False

        super().__init__(qubits=self.qubits, rotate=requires_rotation)


    def rotate_basis(self, circuit, qubit=None, observable=None):
        """
        Call rotate basis in super class, Measurement. Modifies circuit in-place.
        Uses attributes of self for optional arguments which are not provided.
        Input:
            - circuit (qiskit.QuantumCircuit)
            - qubit=None (int or list)
            - observable (QNN-Gen Observable)
        """

        if qubit is None:
            qubit = self.qubits

        if observable is None:
            observable = self.observable

        super().rotate_basis(circuit, qubit, observable)


    def output(self, counts):
        """
        Input:
            - counts (dict):

        Returns:
            - A list (np.ndarray) of probabilities for each qubit.

        probability_list[0] is the probability of the the qubit whose index is given by
        self.qubits[0] being in the |1> state (if self.one = True)
        """

        probability_list = []

        for qubit in self.qubits:
            qubit_counts = self.counts_for_qubits(counts, qubit)
            prob_dict = self.counts_to_probability(qubit_counts)

            # Note: below might not be general enough for multi-qubit measurements
            if self.p_zero == True:
                probability_list.append(prob_dict['0'])
            else:
                probability_list.append(prob_dict['1'])

        return np.array(probability_list)


class ProbabilityThreshold(Measurement):
    """
    The output function uses a threshold on probabilities of being in the zero state (if zero=True)
    to return a binary classification result.

    E.g. If threshold=0.5 and the probability of the qubit being in the |0> state is >= 0.5 then
    the output function will return a label of [0] for this qubit. If the probability of the qubit
    being in the |0> state is < 0.5 then a label of [1] is returned by output() by default.
    """

    def __init__(self, qubits, p_zero=True, threshold=0.5, labels=None, observable=None):
        """
        Attributes:
        - qubits (int or list): qubit index  or list of qubit indices

        - observable (QNN-Gen Observable): The observable corresponding the basis to measure in

        - p_zero=True (Boolean): If True, output returns probabilties of qubit being measured in the |0> state.
        If false, output returns probabilties of qubit being measured in the |1> state.

        - threshold=0.5 (float): For binary classification. Should be between 0 and 1.

        - labels=None (np.ndarray): The lables to return from output. A 2 element list. labels[0] is the label
        corresponding to a probability that execedes the threshold, labels[1] corresponds to the probability
        being
        """

        if isinstance(qubits, int):
            self.qubits = [qubits]
        else:
            self.qubits = qubits

        self.p_zero = p_zero
        self.threshold = threshold

        if labels is None:
            if p_zero == True:
                self.labels = np.array([0, 1])
            else:
                self.labels = np.array([1, 0])
        else:
            self.labels = labels

        self.observable = observable
        if observable is not None:
            requires_rotation = True
        else:
            requires_rotation = False

        super().__init__(qubits=self.qubits, rotate=requires_rotation)


    def rotate_basis(self, circuit, qubit=None, observable=None):
        """
        Call rotate basis in super class, Measurement. Modifies circuit in-place.
        Uses attributes of self for optional arguments which are not provided.
        Input:
            - circuit (qiskit.QuantumCircuit)
            - qubit=None (int or list)
            - observable (QNN-Gen Observable)
        """
        if qubit is None:
            qubit = self.qubits

        if observable is None:
            observable = self.observable

        super().rotate_basis(circuit, qubit, observable)


    def output(self, counts):
        """
        Input:
            - counts (dict):
        Returns:
            - (np.ndarray): The element of self.label corresponding to the outcome after thresholding
        """
        prob = Probability(self.qubits, self.p_zero, self.observable)
        prob_vec = prob.output(counts)
        threshold_vec = prob_vec >= self.threshold # a boolean vector
        labels = np.where(threshold_vec, self.labels[0], self.labels[1]) # labels[0] where true

        return labels


class Expectation(Measurement):
    """
    A class to assist in computing expectation values. Currently for one qubit expectation.
    """

    def __init__(self, qubits, observable=Observable.Z()):
        """
        - qubit (int): The index of the qubit
        - observable=Observable.Z() (matrix, tuple, or QNN-Gen Observable object): The observable
        to meausre with respect to
            - If matrix it should be a list or numpy.ndarray
            - If tuple, it should be of the form ("name": matrix)
        """

        if isinstance(qubits, int):
            self.qubits = [qubits]
        else:
            self.qubits = qubits # just single-qubit for now

        if isinstance(observable, (list, np.ndarray)):
            self.observable = Observable(observable)

        elif isinstance(observable, tuple):
            self.observable = Observable(matrix=observable[1], name=observable[0])

        elif isinstance(observable, Observable):
            self.observable = observable

        else:
            raise ValueError("Observable argument is not a correct type. Check the __init__ docstring.")

        if self.observable.name == "Z":
            requires_rotation = False
        else:
            requires_rotation = True

        super().__init__(qubits=self.qubits, rotate=requires_rotation)

    def rotate_basis(self, circuit, qubit=None, observable=None):
        """
        Overwritting rotate_basis in Measurement
        """

        if qubit is None:
            qubit = self.qubits

        if observable is None:
            observable = self.observable

        super().rotate_basis(circuit, qubit, observable)

    def output(self, counts):
        """
        Input:
            - counts (dict):
        Returns:
            - (np.ndarray):
        """
        # Convert to probability
        qubit_counts = self.counts_for_qubits(counts, self.qubits)
        prob_dict = self.counts_to_probability(qubit_counts)

        # Match measurement results to corresponding eigenvalues
        prob_tuples = tuple(prob_dict.items())
        sorted_prob_tuple = sorted(prob_tuples, key=lambda k: int(k[0], 2)) # sort by bits
        # sorting is to make list positions align with self.observabale.eigenvalues

        probs = np.array([prob for (key, prob) in sorted_prob_tuple])

        return np.array([np.dot(self.observable.eigenvalues, probs)])
