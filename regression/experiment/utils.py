import numpy as np

# Function to perform mini-batch SGD update
def sgd_update(theta, X_batch, Y_batch, learning_rate):
    prediction_error = Y_batch - X_batch @ theta
    gradient = -2 * X_batch.T @ prediction_error / len(X_batch)
    grad_norm = np.linalg.norm(gradient)
    if grad_norm > 1000:
        gradient = gradient * (1000.0 / grad_norm)
    return theta - learning_rate * gradient, gradient

# Optimized SGD update function
def sgd_update_optimized(theta, X_batch, Y_batch, learning_rate, dtype=np.float32):
    """Optimized SGD update with improved memory efficiency"""
    # Use in-place operations where possible
    prediction_error = Y_batch - X_batch @ theta
    
    # Vectorized gradient computation
    gradient = (-2.0 / len(X_batch)) * (X_batch.T @ prediction_error)
    grad_norm = np.linalg.norm(gradient)
    if grad_norm > 1000:
        gradient = gradient * (1000.0 / grad_norm)
    
    # In-place parameter update
    theta_new = theta - learning_rate * gradient
    
    return theta_new.astype(dtype), gradient.astype(dtype)

#compute the gradient of client 0 and client 1 at the final global theta
def compute_gradient(X, Y, theta):
    prediction_error = Y - X @ theta
    gradient = -2 * X.T @ prediction_error / len(X)
    grad_norm = np.linalg.norm(gradient)
    if grad_norm > 1000:
        gradient = gradient * (1000.0 / grad_norm)
    return gradient

# Function to compute MSE loss
def compute_loss(X, Y, theta):
    """
    Compute Mean Squared Error loss for given parameters.
    
    Args:
        X: Input features
        Y: Target values  
        theta: Model parameters
        
    Returns:
        MSE loss value
    """
    prediction_error = Y - X @ theta
    return np.mean(prediction_error ** 2)

# Optimized loss computation
def compute_loss_optimized(X, Y, theta, dtype=np.float32):
    """
    Optimized MSE loss computation with better memory efficiency.
    
    Args:
        X: Input features
        Y: Target values  
        theta: Model parameters
        dtype: Data type for computation
        
    Returns:
        MSE loss value
    """
    # Vectorized computation with specified dtype
    prediction_error = Y.astype(dtype) - (X.astype(dtype) @ theta.astype(dtype))
    return np.mean(prediction_error * prediction_error, dtype=dtype)

# Function to compute loss on all clients' data
def compute_global_loss(clients_data, theta, num_clients):
    """
    Compute total loss across all clients.
    
    Args:
        clients_data: Dictionary containing client data
        theta: Model parameters
        num_clients: Number of clients
        
    Returns:
        Total weighted loss across all clients
    """
    total_loss = 0.0
    total_samples = 0
    
    for client_id in range(num_clients):
        _, X, Y, length = clients_data[client_id]
        client_loss = compute_loss(X, Y, theta)
        total_loss += client_loss * length
        total_samples += length
    
    return total_loss / total_samples

# Optimized global loss computation
def compute_global_loss_optimized(clients_data, theta, num_clients, dtype=np.float32, use_sampling=False, sample_ratio=0.3):
    """
    Optimized global loss computation with optional client sampling.
    
    Args:
        clients_data: Dictionary containing client data
        theta: Model parameters
        num_clients: Number of clients
        dtype: Data type for computation
        use_sampling: Whether to sample subset of clients
        sample_ratio: Fraction of clients to sample
        
    Returns:
        Total weighted loss across all (or sampled) clients
    """
    if use_sampling:
        # Sample subset of clients for faster computation
        sample_size = max(1, int(num_clients * sample_ratio))
        client_ids = np.random.choice(num_clients, size=sample_size, replace=False)
    else:
        client_ids = range(num_clients)
    
    total_loss = 0.0
    total_samples = 0
    
    for client_id in client_ids:
        _, X, Y, length = clients_data[client_id]
        client_loss = compute_loss_optimized(X, Y, theta, dtype)
        total_loss += client_loss * length
        total_samples += length
    
    return (total_loss / total_samples).astype(dtype)