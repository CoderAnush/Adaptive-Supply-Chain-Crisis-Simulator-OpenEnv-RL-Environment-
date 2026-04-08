from .env import SupplyChainEnv

class EasyGrader:
    def __init__(self):
        self.config = {"max_steps": 10, "enable_crises": False, "seed": 101}
    def grade(self, env: SupplyChainEnv) -> float:
        st = env.state()
        rate = st.total_fulfilled / max(1, (st.total_fulfilled + st.total_unfulfilled))
        cost_score = max(0, 1.0 - (st.total_cost / 8000.0))
        return min(1.0, max(0.0, (0.7 * rate) + (0.3 * cost_score)))

class MediumGrader:
    def __init__(self):
        self.config = {"max_steps": 15, "enable_crises": True, "seed": 202}
    def grade(self, env: SupplyChainEnv) -> float:
        st = env.state()
        rate = st.total_fulfilled / max(1, (st.total_fulfilled + st.total_unfulfilled))
        cost_score = max(0, 1.0 - (st.total_cost / 12000.0))
        return min(1.0, max(0.0, (0.6 * rate) + (0.4 * cost_score)))

class HardGrader:
    def __init__(self):
        self.config = {"max_steps": 20, "enable_crises": True, "seed": 303}
    def grade(self, env: SupplyChainEnv) -> float:
        st = env.state()
        rate = st.total_fulfilled / max(1, (st.total_fulfilled + st.total_unfulfilled))
        cost_score = max(0, 1.0 - (st.total_cost / 18000.0))
        # Hard task penalizes heavily for low fulfillment rate
        rate_penalized = max(0, (rate - 0.2) * 1.25)
        return min(1.0, max(0.0, (0.7 * rate_penalized) + (0.3 * cost_score)))

TASKS = {
    "steady_state": EasyGrader(),
    "suez_blockage": MediumGrader(),
    "black_swan": HardGrader()
}
