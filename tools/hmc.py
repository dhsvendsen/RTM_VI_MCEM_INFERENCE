import numpy as np
import sys

class HMC():
    def __init__(self, init_state, log_target_unormalized_dist, grad_log_target_unormalized_dist):
        
        self.samples = [ init_state ]
        self.log_target_unormalized_dist = log_target_unormalized_dist
        self.grad_log_target_unormalized_dist = grad_log_target_unormalized_dist
    
    def U(self, c):
        return -1.0 * self.log_target_unormalized_dist(c)

    def grad_U(self, c):
        return -1.0 * self.grad_log_target_unormalized_dist(c)

    def run_chain(self, n_steps, epsilon, L):
        
        for i in range(n_steps):
            self.sample(epsilon, L)
    
    def sample(self, epsilon, L):
        """
        Given current q, an energy functions and its gradient, this function samples a momentum p and follows
        it for L steps using the leapfrom algorithm.
        """
        current_q = self.samples[-1]
        q = current_q
        p = np.random.normal(size = len(q)) # independent standard normal variates
        current_p = p

        # Follow trajectory for L steps

        q, p = self.leapfrog(q, p, self.grad_U, epsilon, L)

        # Negate momentum at end of trajectory to make the proposal symmetric

        p = -p

        # Evaluate potential and kinetic energies at start and end of trajectory

        current_U = self.U(current_q)
        current_K = np.sum(current_p**2) / 2

        proposed_U = self.U(q)
        proposed_K = np.sum(p**2) / 2

        # Accept or reject the state at end of trajectory, returning either
        # the position at the end of the trajectory or the initial position

        u = np.random.uniform(size = 1)
        dH = current_U + current_K - proposed_U - proposed_K

        if u < np.exp(dH): # Accept
            self.samples = np.vstack(( self.samples, q))
        else: # Reject
            sys.stdout.write('x')
            sys.stdout.flush()
            self.samples = np.vstack(( self.samples, current_q))
    
    def leapfrog(self, q, p, grad_U, epsilon, L):

        """
        Leap frog algorithm
        """

        # Alternate full steps for position and momentum

        for i in range(L):

            # Make a half step for momentum at the beginning

            p = p - epsilon * grad_U(q).flatten() / 2

            # Make a full step for the position

            q = q + epsilon * p

            # Make a half step 

            p = p - epsilon * grad_U(q).flatten() / 2

        return (q, p)

