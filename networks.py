import tensorflow as tf
import tflearn


class ActorNetwork:

    def __init__(self, sess, actor_a, tau, batch_size, state_dimensions, action_dimensions, action_bounds):
        self.sess = sess
        self.state_dimensions = state_dimensions
        self.action_dimensions = action_dimensions
        self.action_bounds = action_bounds
        self.batch_size = batch_size

        # ---------- Actor Network ----------
        with tf.variable_scope('Actor'):
            self.inputs = tf.placeholder(tf.float32, [None, state_dimensions])
            layer = tf.layers.dense(self.inputs, 400, activation=tf.nn.relu)
            layer = tf.layers.dense(layer, 300, activation=tf.nn.relu)
            self.out = tf.layers.dense(layer, action_dimensions, activation=tf.tanh,
                                       kernel_initializer=tf.initializers.random_uniform(-0.003, 0.003))
            self.scaled_out = self.out * action_bounds

        # ---------- Target Actor ----------
        with tf.variable_scope('Target_Actor'):
            self.target_inputs = tf.placeholder(tf.float32, [None, state_dimensions])
            layer = tf.layers.dense(self.target_inputs, 400, activation=tf.nn.relu)
            layer = tf.layers.dense(layer, 300, activation=tf.nn.relu)
            self.target_out = tf.layers.dense(layer, action_dimensions, activation=tf.tanh,
                                              kernel_initializer=tf.initializers.random_uniform(-0.003, 0.003))
            self.target_scaled_out = self.target_out * action_bounds

        # Get parameters of each network, used for target network updates
        actor_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        target_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Target_Actor')

        # 'Soft' update of target network values, according to tau
        self.update_target_network_params = [target_parameters[i].assign(
            (actor_parameters[i] * tau) + (target_parameters[i] * (1 - tau)))
            for i in range(len(actor_parameters))]

        # Gradient to be provided by critic
        self.gradient = tf.placeholder(tf.float32, [None, self.action_dimensions])

        # Combine gradients and normalize
        gradient = tf.gradients(self.scaled_out, actor_parameters, -self.gradient)
        gradient = [g / batch_size for g in gradient]
        self.optimize = tf.train.AdamOptimizer(actor_a).apply_gradients(zip(gradient, actor_parameters))


class CriticNetwork:

    def __init__(self, sess, critic_a, tau, state_dimensions, action_dimensions):
        self.sess = sess
        self.state_dimensions = state_dimensions
        self.action_dimensions = action_dimensions
        self.tau = tau

        # ---------- Critic Network -----------
        with tf.variable_scope('Critic'):
            self.inputs = tf.placeholder(tf.float32, [None, self.state_dimensions])
            self.action = tf.placeholder(tf.float32, [None, self.action_dimensions])
            layer = tf.layers.dense(self.inputs, 400, activation=tf.nn.relu)

            # Combine action into second layer
            temp1 = tflearn.fully_connected(layer, 300)
            temp2 = tflearn.fully_connected(self.action, 300)
            layer = tf.nn.relu(tf.matmul(layer, temp1.W) + tf.matmul(self.action, temp2.W) + temp2.b)

            self.out = tf.layers.dense(layer, 1, kernel_initializer=tf.initializers.random_uniform(-0.003, 0.003))

        # ---------- Target Critic -----------
        with tf.variable_scope('Target_Critic'):
            self.target_inputs = tf.placeholder(tf.float32, [None, self.state_dimensions])
            self.target_action = tf.placeholder(tf.float32, [None, self.action_dimensions])
            layer = tf.layers.dense(self.target_inputs, 400, activation=tf.nn.relu)

            # Combine action into second layer
            temp1 = tflearn.fully_connected(layer, 300)
            temp2 = tflearn.fully_connected(self.target_action, 300)
            layer = tf.nn.relu(tf.matmul(layer, temp1.W) + tf.matmul(self.target_action, temp2.W) + temp2.b)

            self.target_out = tf.layers.dense(layer, 1, kernel_initializer=tf.initializers.random_uniform(-0.003, 0.003))

        critic_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        target_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Target_Critic')

        self.update_target_network_params = [target_parameters[i].assign(
            (critic_parameters[i] * tau) + (target_parameters[i] * (1 - tau)))
            for i in range(len(critic_parameters))]

        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(critic_a).minimize(self.loss)

        self.critic_gradient = tf.gradients(self.out, self.action)
