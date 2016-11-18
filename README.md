# Eleven_Line_Neural_Network
Experiments with http://iamtrask.github.io/2015/07/12/basic-python-network/

## Background

This is a simple example of a backpropagation neural network described
and explained at the website listed above. I'm using it to try and
pull together some of the basic concepts of this sort of neural
network. I've done something with this way in my past, but most of
those details are long gone now.

[These notes](http://ufldl.stanford.edu/wiki/index.php/Neural_Networks) look useful also, beyond what is discussed in Wikipedia.

### Backpropagation
[Backpropagation](https://en.wikipedia.org/wiki/Backpropagation) is a common method of training a neural network used in conjunction with an optimization method. Backpropagation takes the errors from the current step, calculates the gradient of a loss function with respect to the weights in the network, feeds that gradient into the optimization method which then updates the weights.

Requires a known, desired output for each i nput in order to calculate the loss function gradient. Given this, it is often treated as **supervised learning**.

The goal, as with other supervised learning algorithms, is to find a function that best maps a set of inputs to the correct output. An example would be a **classification** task where teh input is an image and the output is the name of what the image represents.

#### The backpropagation algorithm

There are three vectors in realspace:

- *inputs* -- x,x1,x2,...
- *outputs* -- y,y',y1,y2,...
- *weights* -- w, w0, w1, ...

A neural network corresponds to a function:

y = *f*(w,x) where *f* maps an input x to an output y given weight w. This is called an **activation function**. Activation functions take inputs and map them into a finite range. You choose an activatoin function based on the range you want your outputs to take.

An error function is selected (usually | y = y'|^2) to measure the difference between two outputs.

The general algorithm takes in a set of training examples
(x1,y1),...(xp,yp) and produces a sequence of weights (w0,...wp)
starting from some initial weight w0.

Weights are computed in each iteration, wi is computed using only (xi,yi,wi-1). The final output is the weight vector wp giving a new function y->*f*(wp,x).

w1 is calculated from (x1,y1,w0) by considering a variable weight w
and applying **gradient descent** to the function w ->
E(*f*n(w,x1),y1)) to find a local minimum wtarting at w = w0. w1 is
the minimizing weight found by gradient descent.
