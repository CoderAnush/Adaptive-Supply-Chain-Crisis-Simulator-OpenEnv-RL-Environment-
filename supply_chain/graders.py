from .env import SupplyChainEnv

class EasyGrader:
    def __init__(self):
        self.config = {"max_steps": 10, "enable_crises": False, "seed": 101}
    def grade(self, env: SupplyChainEnv) -> float:
        st = env.state()
        rate = st.total_fulfilled / max(1, (st.total_fulfilled + st.total_unfulfilled))
        cost_score = max(0, 1.0 - (st.total_cost / 8000.0))
        raw_score = (0.7 * rate) + (0.3 * cost_score)
        # Scale score to be strictly within (0.01, 0.99) range as per platform requirements
        return min(0.99, max(0.01, raw_score))

class MediumGrader:
    def __init__(self):
        self.config = {"max_steps": 15, "enable_crises": True, "seed": 202}
    def grade(self, env: SupplyChainEnv) -> float:
        st = env.state()
        rate = st.total_fulfilled / max(1, (st.total_fulfilled + st.total_unfulfilled))
        cost_score = max(0, 1.0 - (st.total_cost / 12000.0))
        raw_score = (0.6 * rate) + (0.4 * cost_score)
        return min(0.99, max(0.01, raw_score))

class HardGrader:
    def __init__(self):
        self.config = {"max_steps": 20, "enable_crises": True, "seed": 303}
    def grade(self, env: SupplyChainEnv) -> float:
        st = env.state()
        rate = st.total_fulfilled / max(1, (st.total_fulfilled + st.total_unfulfilled))
        cost_score = max(0, 1.0 - (st.total_cost / 18000.0))
        # Hard task penalizes heavily for low fulfillment rate
        rate_penalized = max(0, (rate - 0.2) * 1.25)
        raw_score = (0.7 * rate_penalized) + (0.3 * cost_score)
        return min(0.99, max(0.01, raw_score))

# Grader instances
STEADY_STATE_GRADER = EasyGrader()
SUEZ_BLOCKAGE_GRADER = MediumGrader()
BLACK_SWAN_GRADER = HardGrader()

# Top-level callables for openenv.yaml
def grade_steady_state(env: SupplyChainEnv) -> float:
    return STEADY_STATE_GRADER.grade(env)

def grade_suez_blockage(env: SupplyChainEnv) -> float:
    return SUEZ_BLOCKAGE_GRADER.grade(env)

def grade_black_swan(env: SupplyChainEnv) -> float:
    return BLACK_SWAN_GRADER.grade(env)

TASKS = {
    "task_easy": STEADY_STATE_GRADER,
    "task_medium": SUEZ_BLOCKAGE_GRADER,
    "task_hard": BLACK_SWAN_GRADER
}
