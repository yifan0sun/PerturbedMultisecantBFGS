import torch
import torch.nn as nn
import torch.optim as optim


def safe_solve(A,b, eps):
    """
    Safely solves the linear system A x = b by adding a small regularization term if A is near-singular.
    
    This function attempts to solve the system up to 1000 times, each time regularizing A by
    adding eps-scaled identity, until a solution is found.
    
    Args:
        A (Tensor): Coefficient matrix.
        b (Tensor): Right-hand side vector.
        eps (float): Regularization scaling constant.
    
    Returns:
        Tensor: Solution vector x, or a tensor filled with NaNs if unsuccessful.
    """
    nA = torch.norm(A)
    for iter in range(1000):
        try:
            x = torch.linalg.solve(A, b)
            return x
            
        except RuntimeError as e:
            A = A + nA*eps * torch.eye(A.size(0)) 
    return torch.full((A.size(1), b.size(1)), float('nan'))
        
def safe_inverse(A, eps) :
    """
    Computes the inverse of A safely by applying a small regularization when necessary.
    
    Similar to safe_solve, this function adds a small multiple of the identity to A if inversion fails.
    
    Args:
        A (Tensor): Matrix to invert.
        eps (float): Regularization parameter.
    
    Returns:
        Tensor: The inverse of A or a tensor filled with NaNs if unsuccessful.
    """
    nA = torch.norm(A)
    for iter in range(1000):
        try:
            x = torch.inverse(A)
            return x
        except RuntimeError as e:
            A = A + nA*eps * torch.eye(A.size(0)) 
    return torch.full(A.size(), float('nan'))
 
        
def getmu(W, D1, D2, iter_limit, mu, tol):
    """
    Dynamically estimates a diagonal perturbation (mu) to ensure positive semidefiniteness.
    
    Uses an SVD-based approach to compute auxiliary matrices and iteratively doubles mu
    until the constructed matrix H2 has all eigenvalues greater than tol.
    
    Args:
        W (Tensor): Matrix related to secant information.
        D1 (Tensor): First set of secant differences.
        D2 (Tensor): Second set of secant differences.
        iter_limit (int): Maximum iterations for adjusting mu.
        mu (float): Initial estimate for the diagonal perturbation.
        tol (float): Tolerance for the minimal eigenvalue of H2.
    
    Returns:
        float: The adjusted mu value ensuring H2 is PSD.
    """
    
    n, p = D1.shape
    iter = 1
    c = 0.1

    U, Sig, V = torch.svd(W)
    Sig = 1 / torch.clamp(Sig, min=0.001)
    S2 = (-1 + torch.sqrt(1 + 4 * Sig.pow(2) * (c ** 2))) / (2 * Sig)
        
    F = V @ (S2 * U.t())

    eye = torch.eye( p)
    P = torch.inverse(c * eye - (1 / c) * F @ F.t())
    Q = torch.inverse(c * eye - (1 / c) * F.t() @ F)
    invW = safe_inverse(W,0.01)
     
    B1 = torch.cat([
        torch.cat([P, -c * Q @ F.t() + invW], dim=1),
        torch.cat([-c * F @ Q + invW.t(), Q], dim=1)
    ], dim=0)
    B1 = (B1 + B1.t()) / 2
    invB1 = torch.inverse(B1)

    DTD = torch.cat([D1, D2], dim=1).t() @ torch.cat([D1, D2], dim=1)

    while iter < iter_limit:
        ww = safe_solve(invB1 * 2 * mu + DTD, DTD,0.01)
        H2 = torch.cat([
            torch.cat([c * eye, F], dim=1),
            torch.cat([F.t(), c * eye], dim=1)
        ]) - 1 / (2 * mu) * DTD + 1 / (2 * mu) * DTD @ ww
        H2 = (H2 + H2.t())/2.
        
        if torch.any(torch.isnan(H2)) or torch.any(torch.isinf(H2)):
            mu = 0.001
            break
            
        if torch.min(torch.linalg.eigvals(H2).real ) > tol:
            break
        else:
            mu *= 2.
            iter += 1

    return mu
 
 
                           
def rejection(smem, ymem, reject_type, tol):
    """
    Performs secant rejection to filter out nearly collinear update directions.
    
    Computes normalized inner products among stored s vectors and rejects those pairs
    whose absolute inner product is above (1 - tol), improving numerical conditioning.
    
    Args:
        smem (list[Tensor]): List of stored s (step difference) vectors.
        ymem (list[Tensor]): List of stored y (gradient difference) vectors.
        reject_type (str): Currently, only 'inner' is supported.
        tol (float): Tolerance threshold for rejection.
    
    Returns:
        (list[Tensor], list[Tensor]): Filtered s and y vectors.
    """
    
    smem_mat = torch.stack(smem,dim=1)
    ymem_mat = torch.stack(ymem,dim=1)
    
    
    n, p_eff = smem_mat.shape

    if p_eff <= 1:
        return smem, ymem

    if reject_type != 'inner':
        raise ValueError('only inner type is allowed right now')

    # Normalize the columns of smem
    smem_norm = torch.sqrt(torch.sum(smem_mat ** 2, dim=0))
    smem_norm = smem_mat / smem_norm

    # Compute the matrix of inner products
    STS = torch.matmul(smem_norm.t(), smem_norm)

    no_rejection = False

    while not no_rejection and smem_mat.shape[1] > 1:
        p_eff = smem_mat.shape[1]
        no_rejection = True

        for i in range(1, p_eff):
            for j in range(i):
                if torch.abs(STS[i, j]) > (1 - tol):
                    idx = [k for k in range(p_eff) if k != j]
                    smem_mat = smem_mat[:, idx]
                    ymem_mat = ymem_mat[:, idx]
                    STS = STS[:, idx]
                    STS = STS[idx, :]
                    no_rejection = False
                    break
            if not no_rejection:
                break
    smem = [smem_mat[:,i] for i in range(smem_mat.size()[1])]
    ymem = [ymem_mat[:,i] for i in range(ymem_mat.size()[1])]
    return smem, ymem




class LMSBFGSOptim():
    """
    Limited-Memory Multisecant BFGS Optimizer.
    
    Implements a quasi-Newton update that incorporates multiple secant conditions.
    Options include:
      - 'vanilla': Basic limited-memory update.
      - 'symm' or 'diagupdate': Enhanced variants with symmetrization and a diagonal (μ) perturbation.
    
    Also features secant rejection to remove nearly collinear updates and optional μ-scaling.
    """
    def __init__(self,n,p,L,secant_type='curve', Btype='vanilla', 
                 reject_type='inner', reject_tol=0, scale=False, gamma=1): 
        self.n = n              # Dimension of the parameter vector.
        self.p = p              # Number of multisecant pairs to store.
        self.L = L              # Limited-memory window size.
        self.secant_type = secant_type  # Type of secant update: 'curve' or 'anchor'.
        self.Btype = Btype              # Update variant: 'vanilla', 'symm', or 'diagupdate'.
        self.reject_type = reject_type  # Rejection method type.
        self.reject_tol = reject_tol    # Tolerance for rejecting nearly collinear vectors.
        self.scale = scale              # Flag to enable μ-scaling.
        self.gamma = gamma              # Scaling factor for the inverse Hessian approximation.

        # Memory for storing past iterates and gradients.
        self.s_msmem = []  # List of step differences.
        self.y_msmem = []  # List of gradient differences.
        self.xmem = []     # History of parameter vectors.
        self.gmem = []     # History of gradients.
        self.Y_limmem = [] # Stacked y vectors for limited memory.
        self.S_limmem = [] # Stacked s vectors for limited memory.
 

        # Internal parameter for diagonal perturbation.
        self.mu = 0.0001
        self.iter_limit = 10

        
    def update_fn(self,x,g): 
        """
        Computes the search direction using the limited-memory multisecant BFGS update.
        
        Args:
            x (Tensor): Current flattened parameter vector.
            g (Tensor): Current flattened gradient.
            
        Returns:
            Tensor: Modified gradient direction for the parameter update.
        """
        self.xmem.append(x+0.)
        self.gmem.append(g+0.)
 

        if len(self.Y_limmem) ==0:
            direction = g
        else:
            if self.Btype=='vanilla':
                direction = self.limited_update_BFGS(g,False);
            elif self.Btype=='symm' or  self.Btype=='diagupdate':
                d1 = self.limited_update_BFGS(g,False)
                d2 = self.limited_update_BFGS(g,True)
                direction = (d1+d2)/2 
            if self.Btype=='diagupdate':
                 
                smem_mat = torch.stack(self.s_msmem,dim=1)
                ymem_mat = torch.stack(self.y_msmem,dim=1)
                self.mu =  getmu(-smem_mat.T @ ymem_mat, smem_mat, smem_mat, self.iter_limit, self.mu/2,.0001)
                direction = direction - self.mu*g
                if self.scale:
                    direction = direction / max(1,self.mu);
                
        self.update_auxillary_vars()  
        return direction


    def update_auxillary_vars(self):
        """
        Updates internal storage of iterates and secant pairs.
        
        Depending on the secant type ('curve' or 'anchor'), stores the difference between successive
        iterates and gradients. Also applies secant rejection to improve conditioning.
        """
        if len(self.xmem) < 2: return
            
        if self.secant_type == 'curve':
            self.s_msmem.append(self.xmem[-1]-self.xmem[-2])
            self.y_msmem.append(self.gmem[-1]-self.gmem[-2])
            self.xmem.pop(0)
            self.gmem.pop(0)
            while len(self.s_msmem) > self.p:
                self.s_msmem.pop(0)
                self.y_msmem.pop(0)

        elif self.secant_type == 'anchor':
            while len(self.xmem) > (self.p+1):
                self.xmem.pop(0)
                self.gmem.pop(0)
            self.s_msmem = [self.xmem[i] - self.xmem[-1] for i in range(len(self.xmem)-1)]
            self.y_msmem = [self.gmem[i] - self.gmem[-1] for i in range(len(self.gmem)-1)] 
         
          
         
     
   
        if self.reject_tol > 0:
            self.s_msmem, self.y_msmem =  rejection(self.s_msmem, self.y_msmem, self.reject_type, self.reject_tol)
   
        if len(self.y_msmem) == 1:
            self.Y_limmem.append(self.y_msmem[0].view(-1,1))
            self.S_limmem.append(self.s_msmem[0].view(-1,1))
        else: 
            self.Y_limmem.append(torch.stack(self.y_msmem,dim=1))
            self.S_limmem.append(torch.stack(self.s_msmem,dim=1))
        while len(self.Y_limmem) > self.L: 
            self.S_limmem.pop(0)
            self.Y_limmem.pop(0)



    def limited_update_BFGS(self,g,transpose):
        """
        Performs the two-loop recursion to compute the inverse Hessian-vector product.
        
        Iterates over the stored limited-memory secant pairs to approximate the search direction.
        
        Args:
            g (Tensor): Flattened gradient vector.
            transpose (bool): If True, uses the transposed variant.
            
        Returns:
            Tensor: Approximated search direction.
        """
    
        L_eff = len(self.Y_limmem) 
        q = g
        a = torch.zeros((self.p, self.L))

        Rmat = [None for k in range(L_eff)]
        
        for j in range(L_eff - 1, -1, -1):
            Sj = self.S_limmem[j]
            Yj = self.Y_limmem[j]
            p_eff1 = (torch.sum(torch.abs(Yj), dim=0) > 0).sum()
            p_eff2 = (torch.sum(torch.abs(Sj), dim=0) > 0).sum()
            p_eff = min(p_eff1, p_eff2)
            
            Sj = Sj[:, :p_eff]
            Yj = Yj[:, :p_eff]

     
     
    
            Z = Sj
            a[:p_eff, j] = Z.t()@ q
    
            if not transpose:
                R = safe_solve(Sj.t() @ Yj, Sj.t(),0.01).t()
                Rmat[j] = R
            else:
                R = safe_solve(Yj.t() @ Sj, Sj.t(),0.01).t()
                Rmat[j] = R
            q = q - Yj @ (R.t() @ q)
    
        q = q * self.gamma
        u = q - R @ (Yj.t() @ q) + R @ a[:p_eff, 0]
    
    
          
        for j in range(1, L_eff):
            Sj = self.S_limmem[j]
            Yj = self.Y_limmem[j]
            p_eff1 = (torch.sum(torch.abs(Yj), dim=0) > 0).sum()
            p_eff2 = (torch.sum(torch.abs(Sj), dim=0) > 0).sum()
            p_eff = min(p_eff1, p_eff2)
            Sj = Sj[:, :p_eff]
            Yj = Yj[:, :p_eff]

            R = Rmat[j]
            u = u - R @ (Yj.t() @ u) + R @ a[:p_eff, j]
    
        d = u
        return d

 