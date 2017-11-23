# Keras MinimalRNN
Keras implementation of [MinimalRNN: Toward More Interpretable and Trainable Recurrent Neural Networks](https://arxiv.org/abs/1711.06788).

## Network graph of MinimalRNNs (from the paper):

<p align="center">
<img src="https://github.com/titu1994/keras-minimal-rnn/blob/master/images/minimal-rnn.PNG?raw=true" width=50% height=50%>
</p>


# Usage
Import `minimal_rnn.py` and use either the `MinimalRNNCell` or `MinimalRNN` layer

```python
from minimal_rnn import MinimalRNN 

# this imports the layer rather than the cell
ip = Input(...)  # Rank 3 input shape
x = MinimalRNN(units=128)(ip)
...
```
