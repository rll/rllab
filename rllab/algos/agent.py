class Agent():

    def runPolicy(self, state):
        '''
            用于输出策略动作的概率
        '''
        raise NotImplementedError

    def trainPolicy(self, state):
        '''
            输出带有探索的动作
        '''
        raise NotImplementedError

    def observe(self, oldstate, action, newstate, reward, terminated):
        '''
            接受环境观测
        '''
        raise NotImplementedError
