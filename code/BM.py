import numpy as np
#from memory_profiler import profile

class BM:
    def __init__(self, N, n_input=1):
        """ N: total number of neurons. """
        self.N = N
        self.Nin = n_input

        self.W = np.random.normal(loc=0.0, scale=0.01, size=(self.N, self.N) )
        self.W = 0.5*( self.W + self.W.T )
        self.b = np.zeros( shape=(self.N, 1) )

        self.p = np.zeros_like(self.b)
        self.s = np.random.randint( 0,high=2, size=(self.N, 1) )

    def energy(self):
        return self.s.T @ ( self.W @ self.s)

    def clamp_input(self, p):
        """ clamp inputs to values given by p """
        self.p[:self.Nin] = np.expand_dims( p, axis=1 )
        self.s[:self.Nin] = np.expand_dims( p, axis=1 ) # use prob values for input as well

    def positive_phase(self):
        # Gibbs sampling:
        for n in range(self.Nin,self.N):
            z = self.W @ self.s
            self.p[n] = 1. / (1. + np.exp( -z[n] ) )
            self.s[n] = int( np.random.random() < self.p[n] )

    def negative_phase(self):
        for n in range(self.Nin):
            z = self.W @ self.s
            self.p[n] = 1. / (1. + np.exp( -z[n] ) )
            self.s[n] = self.p[n]
        for n in range(self.Nin,self.N):
            z = self.W @ self.s
            self.p[n] = 1. / (1. + np.exp( -z[n] ) )
            self.s[n] = int( np.random.random() < self.p[n] )

    def train_with_CD(self, input_data, learning_rate ):
        """ WARNING: input_data is n_feature-by-n_samples!!! """
        Wupdates = np.zeros_like(self.W)
        bupdates =  np.zeros_like(self.b)

        counter = 0
        for col in range( input_data.shape[1] ):
            if counter % 10 == 0:
                print('iteration: ', counter)
            counter += 1

            data_sample = input_data[:,col]

            self.clamp_input( data_sample )

            # phase I of CD
            self.positive_phase()
            Wupdates += np.outer( self.s, self.s )
            bupdates += self.s

            # phase II of CD
            self.negative_phase()
            self.positive_phase()
            Wupdates -= np.outer( self.s, self.s )
            bupdates -= self.s

        n_samples = float( input_data.shape[1] )
        self.W += (learning_rate/n_samples)*Wupdates
        self.b += (learning_rate/n_samples)*bupdates


    def dream(self, hidden_state ):
        self.p[self.Nin:] = hidden_state
        for n in range(self.Nin):
            z = self.W @ self.s
            self.p[n] = 1. / (1. + np.exp( -z[n] ) )
            self.s[n] = self.p[n]
        return self.p[:self.Nin]

