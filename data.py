import torch
from tqdm import tqdm


def get_data(m, n,  eig_range, prob_difficulty, sigma=10, class_balance = 0.5, seed = None):
    """
    Generates a synthetic dataset for a binary classification problem.
    
    This function creates a feature matrix A and label vector b with controlled conditioning.
    The eigenvalue decay is governed by eig_range, and noise is added via sigma.
    
    Args:
        m (int): Number of samples.
        n (int): Number of features.
        eig_range (float): Controls the exponential decay of feature importance.
        prob_difficulty (str): 'low' or 'high' difficulty, affecting the generation of base vector c.
        sigma (float): Noise standard deviation.
        class_balance (float): Threshold to determine class labels.
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        Tuple (Tensor, Tensor): Normalized feature matrix A and binary label vector b.
    """


    if not seed is None:
        torch.manual_seed(int(seed))
     

    c_bar = torch.exp(-torch.linspace(0, eig_range, n)).reshape(n, 1)

    if prob_difficulty == 'low':
        c = torch.randn(n, 1)  # distance
    elif prob_difficulty == 'high':
        c = torch.randn(n, 1) * (1 - c_bar) 

    W = sigma * torch.randn(m, n) * (torch.ones(m, 1) @ c_bar.T)

    b = (torch.randn(m, 1) > class_balance).float()

    
    A = (2*b.float()-1) @ c.T 
    A[:,:2:] = A[:,1:2:]/(2*b.float()-1)
 
    A = A + W
    A = A / torch.norm(A)
    return A,b 



  
 
 
