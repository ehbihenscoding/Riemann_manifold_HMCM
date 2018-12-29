import tensorflow as tf
import tensorflow_probability as tfp


class VanillaHMC:
    def __init__(self, log_prob, step_size, num_leapfrog_steps, theta_0):
        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps
        self.log_prob = log_prob
        self.theta_0 = theta_0

    def sample(self, n_iter, n_burn_in):
        step_size = tf.get_variable(name='step-size', initializer=self.step_size,
                                    use_resource=True, trainable=False)
        vanilla_hmc = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=self.log_prob,
                                                     num_leapfrog_steps=self.num_leapfrog_steps,
                                                     step_size=step_size,
                                                     step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy())

        sample, kernel_results = tfp.mcmc.sample_chain(num_results=n_iter,
                                                       num_burnin_steps=n_burn_in,
                                                       current_state=self.theta_0,
                                                       kernel=vanilla_hmc)
        init_op = tf.global_variables_initializer()

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            sample_, kernel_results_ = sess.run([sample, kernel_results])

        return sample_, kernel_results_.is_accepted
