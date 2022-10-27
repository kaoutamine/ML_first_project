# Useful starting lines

import numpy as np
import matplotlib.pyplot as plt



def load_csv_data(data_path):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int64)  # check if int 64 precision was actually needed
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    return yb, input_data, ids


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for _ in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        # gradient w by descent update
        w = w - gamma * grad
    
    loss = compute_mse(y, tx, w)
    return w, loss 






def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def compute_mse(y, tx, w):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean((y - tx.dot(w)) ** 2)


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    return compute_mse(y, tx, w)
    # return calculate_mae(e)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    w = initial_w
    batch_size = 100

    for _ in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            grad, _ = compute_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
    
    loss = compute_mse(y, tx, w)
    return w, loss




def least_squares(y, tx):
    """apply the least squares 

    Args:
       y : perfect model 
       tx : data

    Returns:
        the weights and the mse loss
    """
    
    s= tx.T.dot(tx)
    t = tx.T.dot(y)
    w = np.linalg.solve(s, t)
    mse = compute_mse(y, tx, w)
    

    return w, mse


def ridge_regression(y, tx, lambda_):
    """apply ridge regression

    Args:
       y : perfect model 
       tx : data
       lambda_ : penalty coefficient

    Returns:
        the weights and the mse loss
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss
    


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    return 1/(1 + np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 

    Returns:
        a non-negative loss

    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    
    sig = sigmoid(tx@w)
    
    term1 = y.T@np.log(sig)
    term2 = (1- y).T @ np.log(1 - sig)
    loss = -(term1 +term2)/y.shape[0]
    return loss[0][0]


def compute_gradient_sig(y,tx,w):
     return tx.T@(sigmoid(tx@w)-y)/y.shape[0]

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1) 

    """
    g = compute_gradient_sig(y,tx,w)
    loss = calculate_loss(y,tx,w)
    w = w - gamma*g
    
    return loss,w



def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Does logistic regression. 
    Fits output w using gradient descent with a logistic loss function. 

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w:  shape=(D, 1) 
        max_iters : int 
        gamma: float

    Returns:
        w: shape=(D, 1) 
        loss: scalar number

    """
    y[y == -1] = 0 #because we have to have values between 0 and 1
    threshold = 1e-8
    y= y.reshape(y.shape[0],1)
    w = initial_w  
    losses = [] 
    # start the logistic regression
    for _ in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        losses.append(loss)
        #Stopping criterium
        # IMO the error should be here, we should look at losses array 
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    
    if max_iters == 0 :
        loss, _ = learning_by_gradient_descent(y, tx, w, gamma)
    else:
        loss = losses.pop()

        
    return w, loss 

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    """

    loss = calculate_loss(y, tx, w) 
    gradient = compute_gradient(y, tx, w) + 2 * lambda_ * w
    return loss,gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.  

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    """
    loss, g = penalized_logistic_regression(y,tx,w,lambda_)
    w = w - gamma*g
    return loss, w


def reg_logistic_regression(y, x, lambda_, inital_w, max_iters, gamma):
    """
    Does regulized logistic regression. 
    Fits output w using gradient descent with a logistic loss function,
    but taking into account the complexity of the model. 

    Args:
        y:  shape=(N, 1)
        x: shape=(N, D)
        lambda_ : scalar 
        initial_w:  shape=(D, 1) 
        max_iters : int 
        gamma: float

    Returns:
        w: shape=(D, 1) 
        loss: scalar number

    """
    # init parameters
    y[y == -1] = 0 #because we have to have values between 0 and 1
    y= y.reshape(y.shape[0],1)

    threshold = 1e-8
    losses = []

    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = inital_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        # IMO the error should be here, we should look at losses array 
        # Also, found weird error about np.ndarray not having exp of 0 ? Might need to check sigmoid
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    if max_iters == 0 :
        loss, _ = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
    else :
        loss = losses.pop()
    return w, loss