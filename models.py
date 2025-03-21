
import torch
import torch.nn as nn
from tqdm import tqdm



class SimpleNN(nn.Module):

    """
    A simple feedforward neural network (MLP) with Softplus activations.
    
    The network is constructed using a series of Linear layers interleaved with Softplus.
    The final layer outputs a single value for binary classification.
    """
    def __init__(self, input_size, hidden_sizes):
        super(SimpleNN, self).__init__()
        self.all_layers = nn.ModuleList()
        fc = nn.Linear(input_size, hidden_sizes[0])
        softplus  = nn.Softplus()
        self.all_layers.append(fc)
        self.all_layers.append(softplus)

        for i,h in enumerate(hidden_sizes[:-1]):
            fc = nn.Linear(hidden_sizes[i], hidden_sizes[i+1])
            softplus  = nn.Softplus()
            self.all_layers.append(fc)
            self.all_layers.append(softplus)

             
        fc = nn.Linear(hidden_sizes[-1],1) 
        self.all_layers.append(fc)
        self.num_params =   sum(p.numel() for p in self.parameters() if p.requires_grad)


    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input features.
            
        Returns:
            Tensor: Network output.
        """
        for l in self.all_layers:
            x = l(x) 
        return x





def train_model( data, targets,test_data,test_targets,model,optim, learning_rate, num_epochs):
    """
    Trains the provided model using a custom optimizer update.
    
    This function:
      - Computes the loss and its gradient.
      - Flattens parameters and gradients for use with the custom optimizer.
      - Updates the parameters manually using the optimizer's update_fn.
      - Tracks loss, gradient norm, and misclassification rates on both train and test sets.
    
    Args:
        data (Tensor): Training data.
        targets (Tensor): Training labels.
        test_data (Tensor): Test data.
        test_targets (Tensor): Test labels.
        model (nn.Module): Neural network model.
        optim: Optimizer object with an update_fn method.
        learning_rate (float): Step size for parameter updates.
        num_epochs (int): Number of epochs.
    
    Returns:
        Tuple: (trained model, loss history, gradient norm history, train misclassification, test misclassification)
    """
    track_loss = torch.full((num_epochs,), float('inf'))
    track_grad = torch.full((num_epochs,), float('inf'))
    track_misclass = torch.full((num_epochs,), float('inf'))
    if test_data is not None:
        track_test_misclass = torch.full((num_epochs,), float('inf'))
    else:
        track_test_misclass = None


    
    for epoch in tqdm(range(num_epochs), desc='Training Progress'):
        
        outputs = model(data) 
        loss_unit = model.criterion(outputs, targets)
        track_loss[epoch] = loss_unit.detach()
        
        loss_unit.backward()
    
        grads = []
        params = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
                params.append(param.view(-1))
        flat_grad = torch.cat(grads)
        track_grad[epoch] = torch.norm(flat_grad)
        
        flat_param = torch.cat(params)
    
        flat_grad = optim.update_fn(flat_param,flat_grad)
        if torch.any(torch.isnan(flat_grad)) or torch.any(torch.isinf(flat_grad)): 
            print('exiting')
            break

        
        index = 0
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    num_elements = param.numel()
                    grad = flat_grad[index:index+num_elements].view(param.size())
                    param -= learning_rate * grad
                    index += num_elements
        
        
        model.zero_grad()

       
        track_misclass[epoch] = evaluate(model, data,targets)
        if test_data is not None: track_test_misclass[epoch] = evaluate(model, test_data,test_targets)
        

    
    return model, track_loss, track_grad,track_misclass,track_test_misclass






def evaluate(model, data, labels):
    """
    Evaluates the model's performance in terms of misclassification rate.
    
    Args:
        model (nn.Module): Trained model.
        data (Tensor): Data on which to evaluate.
        labels (Tensor): True labels.
    
    Returns:
        float: Misclassification rate (1 - accuracy).
    """
    model.eval()
     
    outputs = model(data)

    if outputs.size()[1] == 1:
        predicted = (outputs > 0).float()
    else:
        _, predicted = torch.max(outputs.data, 1) 
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return  (1-(correct / total))
 