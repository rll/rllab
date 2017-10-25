class Agent():

    def runPolicy(self, state):
        raise NotImplementedError

    def trainPolicy(self, state):
        raise NotImplementedError

    def observe(self, oldstate, action, newstate, reward, terminated):
        raise NotImplementedError
