import torch

class LinearTaskGenerator:
    def __init__(
        self, 
        x_dim: int, 
        batch_size: int, 
        seq_len: int, 
        noise_std: float = 0.0, 
        outlier_prob: float = 0.0, 
        outlier_scale: float = 10.0
    ):
        self.x_dim = x_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.noise_std = noise_std
        self.outlier_prob = outlier_prob
        self.outlier_scale = outlier_scale

    def generate(self):
        """
        Generates data according to y_i = x_i^T \theta^* + \epsilon_i + \eta_i
        where \epsilon_i ~ N(0, noise_std^2)
        and \eta_i ~ N(0, outlier_scale^2) with probability outlier_prob
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Generate optimal weights theta* for each task in the batch
        # Shape: (batch_size, x_dim, 1)
        theta_star = torch.randn(self.batch_size, self.x_dim, 1, device=device)
        
        # 2. Generate context points X_context
        # Shape: (batch_size, seq_len, x_dim)
        X_context = torch.randn(self.batch_size, self.seq_len, self.x_dim, device=device)
        
        # 3. Generate standard Gaussian noise epsilon_i
        # Shape: (batch_size, seq_len, 1)
        epsilon = torch.randn(self.batch_size, self.seq_len, 1, device=device) * self.noise_std
        
        # 4. Generate outlier noise eta_i
        # Shape: (batch_size, seq_len, 1)
        outlier_mask = (torch.rand(self.batch_size, self.seq_len, 1, device=device) < self.outlier_prob).float()
        eta = torch.randn(self.batch_size, self.seq_len, 1, device=device) * self.outlier_scale * outlier_mask
        
        # 5. Compute context targets Y_context = X_context * theta* + epsilon + eta
        # Shape: (batch_size, seq_len, 1)
        Y_context = torch.bmm(X_context, theta_star) + epsilon + eta
        
        # 6. Generate test queries X_test
        # Shape: (batch_size, 1, x_dim)
        X_test = torch.randn(self.batch_size, 1, self.x_dim, device=device)
        
        # 7. Compute true test target Y_test_true (CLEAN: no epsilon, no eta)
        # Shape: (batch_size, 1, 1)
        Y_test_true = torch.bmm(X_test, theta_star)
        
        return X_context, Y_context, X_test, Y_test_true

    def _tokenize(self, X_context: torch.Tensor, Y_context: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
        """
        Tokenizes the input sequences to match dimensions.
        Assumes token format embeds x and y into a single vector of dim (x_dim + 1).
        Test token has y set to 0.
        """
        # Context tokens: [x_i, y_i]
        context_tokens = torch.cat([X_context, Y_context], dim=-1) # (B, seq_len, x_dim + 1)
        
        # Test tokens: [x_test, 0]
        zeros = torch.zeros(self.batch_size, 1, 1, device=X_context.device)
        test_tokens = torch.cat([X_test, zeros], dim=-1) # (B, 1, x_dim + 1)
        
        # Full sequence
        full_sequence = torch.cat([context_tokens, test_tokens], dim=1) # (B, seq_len + 1, x_dim + 1)
        
        return full_sequence
