[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<img alt="Keras" src="https://img.shields.io/badge/Keras%20-%23D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/> <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" />

# Tensorflow/ Keras Model Profiler

Gives you some basic but important information about your `tf` or `keras` model like,

* Model Parameters
* Model memory requirement on GPU
* Memory required to store parameters `model weights`.
* GPU availability and GPU IDs if available

## Dependencies

```
python >= 3.6
numpy 
tabulate
tensorflow >= 2.0.0
keras >= 2.2.4
```
Built and tested on `tensorflow == 2.3.1`

## Installation 

using pip.
```
pip install model_profiler
```

## Usage

Firs load any model built using keras or tensorflow. Here for simplicity we will load model from kera applications.

```python
form tensorflow.keras.applications import VGG16

model = VGG16(include_top=True, weights="imagenet", input_tensor=None,
              input_shape=None, pooling=None, classes=1000,
              classifier_activation="softmax")
```

Now after installing `model_profiler` run

```python
from profiler import model_profiler

Batch_size = 128
profile = model_profiler(model, Batch_size)

print(profile)
```
`Batch_size` have effect on `model` memory usage so GPU memory usage need `batch_size`, it's default value if `1`.

### Output

```
| Model Profile                    | Value               | Unit    |
|----------------------------------|---------------------|---------|
| Selected GPUs                    | ['0', '1']          | GPU IDs |
| No. of FLOPs                     | 0.30932349055999997 | BFLOPs  |
| GPU Memory Requirement           | 7.4066760912537575  | GB      |
| Model Parameters                 | 138.357544          | Million |
| Memory Required by Model Weights | 527.7921447753906   | MB      |
```
Default units for the prfiler are

```
# in order 
use_units = ['GPU IDs', 'BFLOPs', 'GB', 'Million', 'MB']

```
You can change units by changing the list entry in appropriate location. For example if you want to get `model` FLOPs in million just change the list as follows.

```
# keep order 
use_units = ['GPU IDs', 'MFLOPs', 'GB', 'Million', 'MB']
```
### Availabel units are
```
    'GB':memory unit gega-byte
    'MB': memory unit mega-byte
    'MFLOPs':  FLOPs unit million-flops
    'BFLOPs':  FLOPs unit billion-flops
    'Million': paprmeter count unit millions
    'Billion': paprmeter count unit billions

```
## More Examples

For further details and more examples visit my [github](https://github.com/Mr-TalhaIlyas/Tensorflow-Keras-Model-Profiler)
