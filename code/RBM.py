import numpy as np
#from memory_profiler import profile

class RBMInputLayer:
    def __init__(self, n):
        self.N = n
        self.h = np.zeros(shape=(self.N, 1), dtype=float)

#    @profile
    def clamp(self, clamping_values):
        clamping_values = clamping_values.reshape((self.N, 1))
        self.h = clamping_values

#    @profile
    def dynamics(self, h_prev, W, b):
        z = b + W @ h_prev
        self.h = z.reshape((self.N,1))

class RBMLayer:
    def __init__(self, n):
        self.N = n
        self.h = np.zeros(shape=(self.N, 1), dtype=int)

    def clamp(self, clamping_values):
        self.h = clamping_values

    def clamp_from_normalized_input(self, normalized_input):
        """requires -1 < normalized_input < 1. and sets the self.h in a valid state"""
        normalized_input = normalized_input.reshape(self.N,1)
        proba = 1. / ( 1. + np.exp(-1.*normalized_input) )
        r = np.random.random((self.N, 1))
        self.h = np.array( r < proba, dtype=int )

#    @profile
    def dynamics(self, h_prev, W, b):
        z = b + W @ h_prev
        z = z.reshape((self.N,1))
        proba = 1. / ( 1. + np.exp(-1.*z) )
        proba = proba.reshape(self.N,1)
        r = np.random.random((self.N, 1))
        self.h = np.array( r < proba, dtype=int )



class RBM:
    def __init__(self, n_input):
        self.layers = [ RBMInputLayer(n_input) ]
        self.weights = list()
        self.biases = list()
        self.n_hidden_layers = 0
        self.N = [ n_input ]
        self.Wupdates = list()
        self.bupdates = list()

        # input layer still has biases
        b = np.random.random((n_input, 1)) - 0.5
        self.biases.append(b)
        self.bupdates.append(np.zeros_like(b))

    def add_layer(self, n):
        self.layers.append( RBMLayer(n) )
        W = np.random.random((n, self.N[-1])) - 0.5
        self.weights.append(W)
        self.Wupdates.append(np.zeros_like(W))

        b = np.random.random((n, 1)) - 0.5
        self.biases.append(b)
        self.bupdates.append(np.zeros_like(b))

        self.N.append(n)
        self.n_hidden_layers += 1


    def positive_phase_inference(self, input_data):
        counter = 0
        out = np.zeros(shape=(input_data.shape[0], self.layers[-1].N), dtype=int)
        for data_sample in input_data:
            if counter % 100 == 0:
                print('iteration: ', counter)
            counter += 1

            data_sample.reshape(1,self.layers[0].N)
            # positive phase
            self.layers[0].clamp(data_sample)
            for l in range(self.n_hidden_layers):
                self.layers[l+1].dynamics(self.layers[l].h, self.weights[l], self.biases[l+1])

            out[counter-1,:] = self.layers[-1].h.reshape(1,self.layers[-1].N)

        return out

#    @profile
    def train_with_CD(self, input_data, learning_rate = 0.1):
        """requires input_data to be np.array with shape (n_samples, n_features),
           and input_data[j,i] = prob ( ith input neuron = 1 for jth sample)"""
        counter = 0

        for l in range(self.n_hidden_layers):
            self.Wupdates[l] = np.zeros_like(self.weights[l])
            self.bupdates[l] =  np.zeros_like(self.biases[l])
        self.bupdates[-1] =  np.zeros_like(self.biases[-1])

        for data_sample in input_data:
            if counter % 1000 == 0:
                print('iteration: ', counter)
            counter += 1

            # positive phase
            self.layers[0].clamp(data_sample)
            self.bupdates[0] += self.layers[0].h

            for l in range(self.n_hidden_layers):
                self.layers[l+1].dynamics(self.layers[l].h, self.weights[l], self.biases[l+1])
                self.Wupdates[l] += np.outer(self.layers[l+1].h, self.layers[l].h)
                self.bupdates[l+1] += self.layers[l+1].h

            # negative phase I
            for l in range(self.n_hidden_layers-1, -1, -1):
                self.layers[l].dynamics(self.layers[l+1].h, self.weights[l].T, self.biases[l] )

            self.bupdates[0] -= self.layers[0].h
            # negative phase II
            for l in range(self.n_hidden_layers):
                self.layers[l+1].dynamics(self.layers[l].h, self.weights[l], self.biases[l+1])
                self.Wupdates[l] -= np.outer(self.layers[l+1].h, self.layers[l].h)
                self.bupdates[l+1] -= self.layers[l+1].h

        for l in range(self.n_hidden_layers):
            self.weights[l] += learning_rate*self.Wupdates[l]/float(input_data.shape[0])
            self.biases[l] +=  learning_rate*self.bupdates[l]/float(input_data.shape[0])
        self.biases[-1] +=  learning_rate*self.bupdates[-1]/float(input_data.shape[0])

    def train_only_last_layer_with_CD(self, input_data, learning_rate = 0.1):
        """requires input_data to be np.array with shape (n_samples, n_features),
           and input_data[j,i] = prob ( ith input neuron = 1 for jth sample)"""
        counter = 0

        for l in range(self.n_hidden_layers):
            self.Wupdates[l] = np.zeros_like(self.weights[l])
            self.bupdates[l] =  np.zeros_like(self.biases[l])
        self.bupdates[-1] =  np.zeros_like(self.biases[-1])

        for data_sample in input_data:
            if counter % 1000 == 0:
                print('iteration: ', counter)
            counter += 1

            # positive phase
            self.layers[0].clamp(data_sample)

            for l in range(self.n_hidden_layers-1):
                self.layers[l+1].dynamics(self.layers[l].h, self.weights[l], self.biases[l+1])

            # last layer also computes updates
            l = self.n_hidden_layers-1
            self.layers[l+1].dynamics(self.layers[l].h, self.weights[l], self.biases[l+1])
            self.Wupdates[l] += np.outer(self.layers[l+1].h, self.layers[l].h)
            self.bupdates[l+1] += self.layers[l+1].h

            # negative phase I
            for l in range(self.n_hidden_layers-1, -1, -1):
                self.layers[l].dynamics(self.layers[l+1].h, self.weights[l].T, self.biases[l] )

            # negative phase II
            for l in range(self.n_hidden_layers-1):
                self.layers[l+1].dynamics(self.layers[l].h, self.weights[l], self.biases[l+1])

            # last layer also computes updates
            l = self.n_hidden_layers-1
            self.layers[l+1].dynamics(self.layers[l].h, self.weights[l], self.biases[l+1])
            self.Wupdates[l] -= np.outer(self.layers[l+1].h, self.layers[l].h)
            self.bupdates[l+1] -= self.layers[l+1].h

        self.weights[-1] += learning_rate*self.Wupdates[-1]/float(input_data.shape[0])
        self.biases[-1] +=  learning_rate*self.bupdates[-1]/float(input_data.shape[0])


    def dream(self, last_layer_hidden_state):
        self.layers[-1].clamp(last_layer_hidden_state)
        for l in range(self.n_hidden_layers-1, -1, -1):
            self.layers[l].dynamics(self.layers[l+1].h, self.weights[l].T, self.biases[l] )
