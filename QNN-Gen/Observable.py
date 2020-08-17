import numpy as np

class Observable():

    def __init__(self, matrix, name="Obs", eigenvalues=None, eigenvectors=None):
        """
        Note: columns encode eigenvectors
        """

        self.matrix = matrix
        self.name = name

        if eigenvalues is None:
            set_values = True
        else:
            self.eigenvalues = eigenvalues
            set_values = False

        if eigenvectors is None:
            set_vectors = True
        else:
            self.eigenvectors = eigenvectors
            set_vectors = False

        if set_values or set_vectors:
            self.set_eigen(set_values, set_vectors)


    def set_eigen(self, set_values=False, set_vectors=False):
        """
        Performs eigen-decompositions on self.matrix.

        Sets self.eigenvalues if set_bvalues=True
        Sets self.eigenvectors if set_vectors=True
        """

        e_vals, e_vecs = np.linalg.eig(self.matrix)

        if set_values == True:
            self.eigenvalues = e_vals

        if set_vectors == True:
            #e_vecs = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
            self.eigenvectors = e_vecs


    @staticmethod
    def X():
        X_matrix = np.array([[0, 1],
                             [1, 0]])

        e_vals = np.array([1, -1])

        e_vecs = 1/np.sqrt(2) * np.array([[1, 1],
                                          [1, -1]])

        return Observable(matrix=X_matrix, name="X", eigenvalues=e_vals, eigenvectors=e_vecs)


    @staticmethod
    def Y():
        Y_matrix = np.array([[0, -1j],
                             [1j, 0]])

        e_vals = np.array([1, -1])

        e_vecs = 1/np.sqrt(2) * np.array([[-1j, 1],
                                          [1, -1j]])

        return Observable(matrix=Y_matrix, name="Y", eigenvalues=e_vals, eigenvectors=e_vecs)


    @staticmethod
    def Z():
        Z_matrix = np.array([[1, 0],
                             [0, -1]])

        e_vals = np.array([1, -1])

        e_vecs = np.array([[1, 0],
                           [0, 1]])

        return Observable(matrix=Z_matrix, name="Z", eigenvalues=e_vals, eigenvectors=e_vecs)

    @staticmethod
    def H():
        H_matrix = 1/np.sqrt(2) * np.array([[1, 1],
                                            [1, -1]])

        e_vals = np.array([1, -1])

        e_vecs = np.array([[np.cos(np.pi/8), -np.sin(np.pi/8)],
                           [np.sin(np.pi/8), np.cos(np.pi/8)]])

        return Observable(matrix=H_matrix, name="H", eigenvalues=e_vals, eigenvectors=e_vecs)
