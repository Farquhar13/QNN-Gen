# QNN-Gen: A Framework for Quantum Neural Networks 
## Beta Release

### Install 
Requirements:
- qiskit
- numpy

Install from PyPI with pip:
```
pip install QNN-Gen
```

Install an editable version with github (good if you want to change the source code):
```
git clone https://github.com/Farquhar13/QNN-Gen.git
cd QNN-Gen
pip install -e . 
```

### Design
QNN-Gen is designed to serve as a useful abstraction for understanding and implementing Quantum Neural Networks (QNNs) or Parameterized Quatum Circuits. The structure of the code is classes is intended to mirror the structure of QNNs. The choices one has to make when constructing a QNN is reflected in QNN-Gen through the use of particular classes and their attributes. 

Furthermore, QNN-Gen is designed to balance both ease-of-use and configurability.

### High-Level Abstraction
We use a high-level abstraction of QNNs to break them down into three main steps:
- Encoding input data 
- Choice of model architecture or ansatz
- Measurement and post-processing

We designed QNN-Gen to match this abstraction in code with the abstract base classes `Encode`, `Model`, and `Measurement`. Respectively, in the `encode.py`, `model.py`, and `measurement.py` functions you can find these abstract base classes as well as several derived classes. To construct a QNN, you simply need to make your choices of modeling decisions and instantiate the corresponding derived classes.

### Examples
We strive to make QNN-Gen as easy-to-use as possible. From the code snippet below you can see that it requires only 3 lines of code using QNN-Gen to create a simple QNN.
```python
import qnn_gen as qg
import numpy as np

x = np.array([1, 0, 0, 1])

encoder = qg.BasisEncoding()
model = qg.TreeTensorNetwork()
full_circuit = qg.combine(x, encoder, model)
```
Which produces the circuit:

![](/images/BasisEncode_TTN.png)

Note that the angels for the `TreeTensorNetwork` model are initialized randomly if they are not provided as an argument. 

You may wonder what happened to the measurement object. First we note that in many cases the choice of a particular model implies which measurements and outputs are sensible. In the case that no `Measurement` object is passed to `qg.combine`, in the background QNN-Gen looks to the `default_measurement` function of the `model`. For the above example, the following code is equivalent. 
```python
import qnn_gen as qg
import numpy as np

x = np.array([1, 0, 0, 1])

encoder = qg.BasisEncoding()
model = qg.TreeTensorNetwork()
measurement = qg.Expectation(qubits=2)
full_circuit = qg.combine(x, encoder, model, measurement)
```

To run the QNN and get predicitions on a toy dataset you can use `qg.run`:
```python
X = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1],
              [1, 0, 0, 0]])

predictions = qg.run(X, encoder, model, measurement)
```
Which will produce a numpy array with 5-elements corresponding to the predictions for the 5 input data points. For this initatiations of the model parameters:
![](/images/BasisEncode_TTN_predictions.png)

For more examples, checkout the `examples/` folder. Inside, you will find python files and jupyter notebooks which demonstrate both the ease-of-use and configurability of QNN-Gen.

### Contributing
QNN-Gen is designed modularly with abstract base classes. We welcome users to create their own class for a different encoding, model/ansatz, or measurement transformation and share them to be potentially added in to QNN-Gen.
