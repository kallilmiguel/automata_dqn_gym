import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import gym
#import keras
from keras_K.config_K import Config
from keras_K.rl_agent_K import QAgent

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape


def build_model(action_shape):
    model = tf.keras.Sequential()
    model.add(Embedding(500, 10, input_length=1))
    model.add(Reshape((10,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(action_shape, activation='linear'))
    return model

env = gym.make("Taxi-v3").env
config = "/home/lucas/taxi_env/keras_K/config_keras.yaml"
model = build_model(env.action_space.n)
agent = QAgent(env=env, config=config, model=model)

agent.compile()

agent.adjust_lr(1e-4)
agent.fit()

agent.save()

agent.evaluate(max_steps=100)

agent.play(verbose=True, max_steps=15, sleep=0.1)


# Passenger locations:
#     - 0: R(ed)
#     - 1: G(reen)
#     - 2: Y(ellow)
#     - 3: B(lue)
#     - 4: in taxi
# Destinations:
#     - 0: R(ed)
#     - 1: G(reen)
#     - 2: Y(ellow)
#     - 3: B(lue)
# Actions:
#     There are 6 discrete deterministic actions:
#     - 0: move south
#     - 1: move north
#     - 2: move east
#     - 3: move west
#     - 4: pickup passenger
#     - 5: drop off passenger
state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
env.s = state
env.render()
state
# env.step(5)
# env.step(5)