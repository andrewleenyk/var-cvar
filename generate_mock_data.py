import numpy as np
import pandas as pd

np.random.seed(42)

# Simulate daily returns (mean and std dev approximate realistic behavior)
returns = {
    'AAPL': np.random.normal(0.0005, 0.02, 250),
    'MSFT': np.random.normal(0.0004, 0.015, 250),
    'TSLA': np.random.normal(0.0008, 0.035, 250),
    'JPM':  np.random.normal(0.0003, 0.01, 250)
}

returns_df = pd.DataFrame(returns)

# Normalize weights to sum to 1
weights = np.random.dirichlet(np.ones(4), size=1).flatten()
weights_dict = dict(zip(returns_df.columns, weights))

# Compute portfolio returns
returns_df['Portfolio'] = returns_df.dot(weights)

# Save to CSV if needed
returns_df.to_csv('data/mock_portfolio_returns.csv', index=False)

returns_df.head()
