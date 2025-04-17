import torch
class HiLoBettingAgent:
    """
    Callable Hi–Lo betting agent based on already‐computed True Count.
    
    Usage:
        agent = HiLoBettingAgent(base_bet=1.0, max_units=10)
        bet   = agent(true_count)   # where true_count = running_count / decks_remaining
    """
    def __init__(self, base_bet: float = 1.0, max_units: int = 10):
        self.base_bet  = base_bet
        self.max_units = max_units

    def __call__(self, true_count: float) -> float:
        """
        Args:
            true_count: Hi–Lo running count divided by decks remaining (e.g. 2.3)
        Returns:
            dollar bet amount (base_bet × units)
        """
        # 1 unit if TC ≤ 1; otherwise floor(TC), capped at max_units
        if true_count <= 1:
            units = 1
        else:
            units = min(self.max_units, int(true_count))

        return torch.tensor([[units * self.base_bet]], dtype=torch.float32)
    
    def to(self, device):
        return self