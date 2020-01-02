import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

# setting random seeds for result reproducibility. This is not super important
tf.set_random_seed(2212)

class Actor:
    def __init__(self, sess, action_dim, observation_dim):
        self.action_dim, self.observation_dim = action_dim, observation_dim
        # setting the our created session as default session
        K.set_session(sess)
        self.sess = sess
        self.state_input, self.output, self.model = self.create_model()
        # Implementing 
        # grad(J(actor_weights)) = sum_(t=1, T-1)[ grad(log(pi(at | st, actor_weights)) * Advantaged(st, at), actor_weights) ]
        # Placeholder for advantage values.
        self.advantages = tf.placeholder(tf.float32, shape=[None, action_dim])
        model_weights = self.model.trainable_weights
        # Adding small number inside log to avoid log(0) = -infinity
        log_prob = tf.math.log(self.output + 10e-10)
        # Multiply log by -1 to convert the optimization problem as minimization problem.
        # This step is essential because apply_gradients always do minimization.
        neg_log_prob = tf.multiply(log_prob, -1)
        # Calulate and update the weights of the model to optimize the actor
        actor_gradients = tf.gradients(neg_log_prob, model_weights, self.advantages)
        grads = zip(actor_gradients, model_weights)
        self.optimize = tf.train.AdamOptimizer(0.001).apply_gradients(grads)

    def create_model(self):
        state_input = Input(shape=self.observation_dim)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(24, activation='relu')(state_h1)
        output = Dense(self.action_dim, activation='softmax')(state_h2)
        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam)
        return state_input, output, model

    def train(self, X, y):
        self.sess.run(self.optimize, feed_dict={self.state_input:X, self.advantages:y})
