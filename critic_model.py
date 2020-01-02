import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

# setting random seeds for result reproducibility. This is not super important
tf.set_random_seed(2212)

class Critic:
    def __init__(self, sess, action_dim, observation_dim):
        self.action_dim, self.observation_dim = action_dim, observation_dim
        # setting our created session as default session 
        K.set_session(sess)
        self.model = self.create_model()

    def create_model(self):
        state_input = Input(shape=self.observation_dim)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(24, activation='relu')(state_h1)
        state_h3 = Dense(24, activation='relu')(state_h2)
        state_h4 = Dense(24, activation='relu')(state_h3)
        output = Dense(1, activation='linear')(state_h4)
        model = Model(inputs=state_input, outputs=output)
        model.compile(loss="mse", optimizer=Adam(lr=0.005))
        return model
