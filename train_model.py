import gym
from collections import deque
from actor_model import Actor
from critic_model import Critic
import numpy as np
import random
import tensorflow as tf

# setting random seeds for result reproducibility. This is not super important
random.seed(2212)
np.random.seed(2212)
tf.set_random_seed(2212)

# Hyperparameters
EPISODES = 1_00_000
REPLAY_MEMORY_SIZE = 1_00_000
MINIMUM_REPLAY_MEMORY = 1_000
DISCOUNT = 0.99
EPSILON = 1
EPSILON_DECAY = 0.999
MINIMUM_EPSILON = 0.001
MINIBATCH_SIZE = 32
VISUALIZATION = False

# Environment details
env = gym.make('CartPole-v1').unwrapped
action_dim = env.action_space.n
observation_dim = env.observation_space.shape

# creating own session to use across all the Keras/Tensorflow models we are using
sess = tf.Session()

# Experience replay memory for stable learning
replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

# Actor model to take actions
# state -> action
actor = Actor(sess, action_dim, observation_dim)
# Critic model to evaluate the acion taken by the actor
# state -> value of state V(s_t)
critic = Critic(sess, action_dim, observation_dim)

sess.run(tf.initialize_all_variables())

def train_advantage_actor_critic(replay_memory, actor, critic):
    minibatch = random.sample(replay_memory, MINIBATCH_SIZE)
    X = []
    y = []
    advantages = np.zeros(shape=(MINIBATCH_SIZE, action_dim))
    for index, sample in enumerate(minibatch):
        cur_state, action, reward, next_state, done = sample
        if done:
            # If last state then advatage A(s, a) = reward_t - V(s_t)
            advantages[index][action] = reward - critic.model.predict(np.expand_dims(cur_state, axis=0))[0][0]
        else:
            # If not last state the advantage A(s_t, a_t) = reward_t + DISCOUNT * V(s_(t+1)) - V(s_t)
            next_reward = critic.model.predict(np.expand_dims(next_state, axis=0))[0][0]
            advantages[index][action] = reward + DISCOUNT * next_reward - critic.model.predict(np.expand_dims(cur_state, axis=0))[0][0]
            # Updating reward to trian state value fuction V(s_t)
            reward = reward + DISCOUNT * next_reward
        X.append(cur_state)
        y.append(reward)
    X = np.array(X)
    y = np.array(y)
    y = np.expand_dims(y, axis=1)
    # Training Actor and Critic
    actor.train(X, advantages)
    critic.model.fit(X, y, batch_size=MINIBATCH_SIZE, verbose=0)

max_reward = 0
for episode in range(EPISODES):
    cur_state = env.reset()
    done = False
    episode_reward = 0
    while not done and episode_reward < 1000:
        if VISUALIZATION:
            env.render()

        action = np.zeros(shape=(action_dim))
        if(np.random.uniform(0, 1) < EPSILON):
            # Taking random actions (Exploration)
            action[np.random.randint(0, action_dim)] = 1
        else:
            # Taking optimal action suggested by the actor (Exploitation)
            action = actor.model.predict(np.expand_dims(cur_state, axis=0))

        next_state, reward, done, _ = env.step(np.argmax(action))

        episode_reward += reward
        
        if done:
            # Episode ends means we have lost the game. So, we are giving large negative reward.
            reward = -100

        # Recording experience to train the actor and critic
        replay_memory.append((cur_state, np.argmax(action), reward, next_state, done))
        cur_state = next_state
        
        if len(replay_memory) < MINIMUM_REPLAY_MEMORY:
            continue
        # Training actor and critic
        train_advantage_actor_critic(replay_memory, actor, critic)

        # Decreasing the exploration probability
        if EPSILON > MINIMUM_EPSILON and len(replay_memory) >= MINIMUM_REPLAY_MEMORY:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(EPSILON, MINIMUM_EPSILON)
    # some bookkeeping
    if(episode_reward > 400 and episode_reward > max_reward):
        actor.model.save_weights(str(episode_reward)+".h5")
    max_reward = max(max_reward, episode_reward)
    print('Episodes:', episode, 'Episodic_Reweard:', episode_reward, 'Max_Reward_Achieved:', max_reward, 'EPSILON:', EPSILON)
