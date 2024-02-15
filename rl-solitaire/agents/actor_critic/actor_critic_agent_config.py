from ..agent_config import AgentConfig


class ActorCriticConfig(AgentConfig):
    """
    A class defining the configuration for an actor-critic agent.
    """
    def __init__(self, policy_head_config, value_head_config, optimizer_config, name="ActorCriticAgent", discount=1.0):
        super().__init__(name, discount)
        self.policy_head_config = policy_head_config
        self.value_head_config = value_head_config
        self.optimizer_config = optimizer_config
