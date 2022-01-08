import random
import numpy as np
import keras


class hDQN:
    def __init__(self,
                 n_samples=1000, meta_n_samples=1000,
                 gamma=0.975,
                 meta_epsilon=1.0, epsilon=1.0, actor_epsilon=1.0,
                 tau=0.001):
        self.n_states = 6

        self.meta_init = 'lecun_uniform'
        self.meta_nodes = [self.n_states, 30, 30, 30, self.n_states]
        self.meta_activation = 'relu' 

        self.init = 'lecun_uniform'
        self.nodes = [self.n_states * 2, 30, 30, 30, self.n_states]
        self.activation = 'relu'


        self.meta_controller = self.get_meta_controller()
        self.target_meta_controller = self.get_meta_controller()
        self.actor = self.get_actor()
        self.target_actor = self.get_actor()

        self.goal_selected = np.ones(self.n_states)
        self.goal_success = np.zeros(self.n_states)
        self.meta_epsilon = meta_epsilon
        self.actor_epsilon = actor_epsilon
        self.n_samples = n_samples
        self.meta_n_samples = meta_n_samples
        self.gamma = gamma
        self.target_tau = tau
        self.memory = []
        self.meta_memory = []

    def get_meta_controller(self):
        meta = keras.models.Sequential()
        for node in self.meta_nodes:
            meta.add(keras.layers.Dense(node, init=self.meta_init, input_shape=(node,)))
            meta.add(keras.layers.Activation(self.meta_activation))
            print('meta node: ' + str(node))
        meta.compile(loss='mean_squared_error',
                     optimizer=keras.optimizers.RMSprop(lr=0.00025, rho=0.9, epsilon=1e-06))
        return meta

    def get_actor(self):
        actor = keras.models.Sequential()
        for node in self.nodes:
            print(node)
            actor.add(keras.layers.Dense(node, init=self.init, input_shape=(node,)))
            actor.add(keras.layers.Activation(self.activation))
        actor.compile(loss='mean_squared_error',
                      optimizer=keras.optimizers.RMSprop(lr=0.00025, rho=0.9, epsilon=1e-06))
        return actor

    def select_move(self, state, goal, goal_value):
        vector = np.concatenate([state, goal], axis=1)
        if random.random() < self.actor_epsilon:
            return np.argmax(self.actor.predict(vector, verbose=0))
        return random.choice([0,1])

    def select_goal(self, state):
        if self.meta_epsilon < random.random():
            pred = self.meta_controller.predict(state, verbose=0)
            print('pred shape: ' + str(pred.shape))
            return np.argmax(pred)+1
        print('Exploring')
        return random.choice(range(self.n_states))

    def criticize(self, goal, next_state):
        return 1.0 if goal == next_state else 0.0

    def store(self, experience, meta=False):
        if meta:
            self.meta_memory.append(experience)
            if len(self.meta_memory) > 1000000:
                self.meta_memory = self.meta_memory[-100:]
        else:
            self.memory.append(experience)
            if len(self.memory) > 1000000:
                self.memory = self.memory[-1000000:]

    def _update(self):
        exps = [random.choice(self.memory) for _ in range(self.n_samples)]
        state_vectors = np.squeeze(np.asarray([np.concatenate([exp.state, exp.goal], axis=1) for exp in exps]))
        next_state_vectors = np.squeeze(np.asarray([np.concatenate([exp.next_state, exp.goal], axis=1) for exp in exps]))
        try:
            reward_vectors = self.actor.predict(state_vectors, verbose=0)
        except Exception as e:
            state_vectors = np.expand_dims(state_vectors, axis=0)
            reward_vectors = self.actor.predict(state_vectors, verbose=0)
        
        try:
            next_state_reward_vectors = self.target_actor.predict(next_state_vectors, verbose=0)
        except Exception as e:
            next_state_vectors = np.expand_dims(next_state_vectors, axis=0)
            next_state_reward_vectors = self.target_actor.predict(next_state_vectors, verbose=0)
        
        for i, exp in enumerate(exps):
            reward_vectors[i][exp.action] = exp.reward
            if not exp.done:
                reward_vectors[i][exp.action] += self.gamma * max(next_state_reward_vectors[i])
        reward_vectors = np.asarray(reward_vectors)
        self.actor.fit(state_vectors, reward_vectors, verbose=0)
        
        #Update target network
        actor_weights = self.actor.get_weights()
        actor_target_weights = self.target_actor.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.target_tau * actor_weights[i] + (1 - self.target_tau) * actor_target_weights[i]
        self.target_actor.set_weights(actor_target_weights)

    def _update_meta(self):
        if 0 < len(self.meta_memory):
            exps = [random.choice(self.meta_memory) for _ in range(self.meta_n_samples)]
            state_vectors = np.squeeze(np.asarray([exp.state for exp in exps]))
            next_state_vectors = np.squeeze(np.asarray([exp.next_state for exp in exps]))
            try:
                reward_vectors = self.meta_controller.predict(state_vectors, verbose=0)
            except Exception as e:
                state_vectors = np.expand_dims(state_vectors, axis=0)
                reward_vectors = self.meta_controller.predict(state_vectors, verbose=0)
            
            try:
                next_state_reward_vectors = self.target_meta_controller.predict(next_state_vectors, verbose=0)
            except Exception as e:
                next_state_vectors = np.expand_dims(next_state_vectors, axis=0)
                next_state_reward_vectors = self.target_meta_controller.predict(next_state_vectors, verbose=0)
            
            for i, exp in enumerate(exps):
                reward_vectors[i][np.argmax(exp.goal)] = exp.reward
                if not exp.done:
                    reward_vectors[i][np.argmax(exp.goal)] += self.gamma * max(next_state_reward_vectors[i])
            self.meta_controller.fit(state_vectors, reward_vectors, verbose=0)
            
            #Update target network
            meta_weights = self.meta_controller.get_weights()
            meta_target_weights = self.target_meta_controller.get_weights()
            for i in range(len(meta_weights)):
                meta_target_weights[i] = self.target_tau * meta_weights[i] + (1 - self.target_tau) * meta_target_weights[i]
            self.target_meta_controller.set_weights(meta_target_weights)

    def update(self, meta=False):
        if meta:
            self._update_meta()
        else:
            self._update()
