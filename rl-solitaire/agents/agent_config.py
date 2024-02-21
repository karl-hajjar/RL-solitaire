class AgentConfig:
    """
    A class defining the basic configuration for a RL agent
    """

    def __init__(self, name="Agent", discount=1.0):
        self.name = name
        self.discount = discount
