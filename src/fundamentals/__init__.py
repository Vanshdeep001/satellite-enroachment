"""
Neural-network fundamentals implemented from scratch in NumPy.

Modules:
- tensor_ops: broadcasting, index arithmetic, reshaping, inner products, Jacobi SVD.
- gradcheck: central-difference numerical gradients and gradient checking.
- neurons: McCulloch-Pitts threshold neuron and single sigmoid neuron.
- mlp: dense layers, activations, losses, optimizers (GD/SGD/Momentum/Nesterov),
  training loop and a parameter-versus-hyperparameter experiment tracker.
- demo: command-line script that renders explanatory figures for all of the above.

No PyTorch autograd is used anywhere in this package; every gradient is derived by hand.
"""
