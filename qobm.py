import numpy as np
from scipy.optimize import minimize, fmin_bfgs
from copy import deepcopy

from grove.pyqaoa.qaoa import QAOA
from grove.pyvqe.vqe import VQE
from grove.alpha.arbitrary_state import arbitrary_state  

import pyquil.quil as pq
import pyquil.api as api
from pyquil.paulis import *
from pyquil.gates import *

def make_qvm(qvm=None):
    if qvm:
        return qvm
    else:
        return api.QVMConnection()

class QBM:

    """
    Quantum Classical Hybrid RBM implementation.
    """

    def __init__(self, qvm=None, num_visible=2, num_hidden=1, steps=3, temp=1.0, quant_meas_num=None, bias=False, reduced=False):
        """
        create an RBM with the specified number of visible and hidden units
        Params
        -------------------------------------------------------------
        qvm:                        (Rigetti QVM connection) Simulator,
        num_visible:                (int) Number of visible units,
        num_hidden:                    (int) Number of hidden units,
        steps:                        (int) Number of steps for QAOA,
        temp:                        (float) Temperature of the system,
        quant_meas_num:                (int) Number of measuremants to use for Quantum expectation estimation.
        --------------------------------------------------------------
        """
        # Initializing the Params
        self.visible_units = num_visible
        self.hidden_units = num_hidden
        self.total_units = self.visible_units + self.hidden_units
        self.qvm = make_qvm(qvm)
        self.quant_meas_num = quant_meas_num
        self.qaoa_steps = steps
        self.beta_temp = temp
        self.state_prep_angle = np.arctan(np.exp(-1/self.beta_temp)) * 2.0

        self.vqe_inst = VQE(minimizer=minimize,
                      minimizer_kwargs={'method': 'nelder-mead'})

        self.param_wb = 0.1 * np.sqrt(6. / self.total_units)
        self.WEIGHTS = np.asarray(np.random.uniform(
                    low=-self.param_wb, high=self.param_wb,
                    size=(num_visible, num_hidden)))

        # Using Reduced or Full Botlzman machines.
        if reduced:
            self.reduced = True
        else:
            self.reduced = False
        
        # Using Bias or not.
        if bias:
            self.BIAS = np.asarray(np.random.uniform(
                    low=-self.param_wb, high=self.param_wb,
                    size=(self.hidden_units)))
        else:
            self.BIAS = None

    def make_unclamped_QAOA(self):
        """
        Internal helper function for building QAOA circuit to get RBM expectation
        using Rigetti Quantum simulator
        Returns
        ---------------------------------------------------
        nus:        (list) optimal parameters for cost hamiltonians in each layer of QAOA
        gammas:        (list) optimal parameters for mixer hamiltonians in each layer of QAOA
        para_prog:  (fxn closure) fxn to return QAOA circuit for any supplied nus and gammas
        ---------------------------------------------------
        """
        
        # Indices 
        visible_indices = [i for i in range(self.visible_units)]
        hidden_indices = [i + self.visible_units for i in range(self.hidden_units)]
        total_indices = [i for i in range(self.total_units)]

        # Full Mixer and Cost Hamiltonian Operator
        full_mixer_operator = []
        for i in total_indices:
            full_mixer_operator.append(PauliSum([PauliTerm("X", i, 1.0)]))

        full_cost_operator = []
        for i in visible_indices:
            for j in hidden_indices:
                full_cost_operator.append(PauliSum([PauliTerm(
                    "Z", i, -1.0 * self.WEIGHTS[i][j - self.visible_units]) * PauliTerm("Z", j, 1.0)]))
        
        if self.BIAS is not None:
            for i in hidden_indices:
                print(i, self.visible_units, i-self.visible_units, self.BIAS[i-self.visible_units])
                full_cost_operator.append(
                    PauliSum([PauliTerm("Z", i, -1.0 * self.BIAS[i-self.visible_units])]))

        # Prepare all the units in a thermal state of the full mixer hamiltonian.         
        state_prep = pq.Program()

        for i in total_indices:
            tmp = pq.Program()
            tmp.inst(RX(self.state_prep_angle, i + self.total_units), CNOT(i + self.total_units, i))
            state_prep += tmp

        # QAOA on full mixer and full cost hamiltonian evolution
        full_QAOA = QAOA(self.qvm,
                   qubits=total_indices,
                   steps=self.qaoa_steps,
                   ref_ham=full_mixer_operator,
                   cost_ham=full_cost_operator,
                   driver_ref=state_prep,
                   store_basis=True,
                   minimizer=fmin_bfgs,
                   minimizer_kwargs={'maxiter': 100},
                   vqe_options={'samples': self.quant_meas_num},
                    rand_seed=1234)

        nus, gammas = full_QAOA.get_angles()

        program = full_QAOA.get_parameterized_program()
        return nus, gammas, program, 0

    def make_clamped_QAOA(self, data_point, iter):
        """
        Internal helper function for building QAOA circuit to get RBM expectation
        using Rigetti Quantum simulator
        Returns
        ---------------------------------------------------
        nus:        (list) optimal parameters for cost hamiltonians in each layer of QAOA
        gammas:        (list) optimal parameters for mixer hamiltonians in each layer of QAOA
        para_prog:  (fxn closure) fxn to return QAOA circuit for any supplied nus and gammas
        ---------------------------------------------------
        """
        
        # Indices
        visible_indices = [i for i in range(self.visible_units)]
        hidden_indices = [i + self.visible_units for i in range(self.hidden_units)]
        total_indices = [i for i in range(self.total_units)]

        # Partial Mixer and Partial Cost Hamiltonian
        partial_mixer_operator = []
        for i in hidden_indices:
            partial_mixer_operator.append(PauliSum([PauliTerm("X", i, 1.0)]))

        partial_cost_operator = []
        for i in visible_indices:
            for j in hidden_indices:
                partial_cost_operator.append(PauliSum([PauliTerm(
                    "Z", i, -1.0 * self.WEIGHTS[i][j - self.visible_units]) * PauliTerm("Z", j, 1.0)]))

        if self.BIAS is not None:
            for i in hidden_indices:
                partial_cost_operator.append(
                    PauliSum([PauliTerm("Z", i, -1.0 * self.BIAS[i - self.visible_units])]))

        state_prep = pq.Program()
        # state_prep = arbitrary_state.create_arbitrary_state(data_point,visible_indices)
    
        # Prepare Visible units as computational basis state corresponding to data point.
        for i, j in enumerate(data_point):
            #print(i,j)
            if j == 1:
                state_prep += X(i)

        # Prepare Hidden units in a thermal state of the partial mixer hamiltonian. 
        for i in hidden_indices:
            tmp = pq.Program()
            tmp.inst(RX(self.state_prep_angle, i + self.total_units),
                     CNOT(i + self.total_units, i))
            state_prep += tmp

        # QAOA on parital mixer and partial cost hamiltonian evolution
        partial_QAOA = QAOA(qvm=self.qvm,
                   qubits=total_indices,
                   steps=self.qaoa_steps,
                   ref_ham=partial_mixer_operator,
                   cost_ham=partial_cost_operator,
                   driver_ref=state_prep,
                   store_basis=True,
                   minimizer=fmin_bfgs,
                   minimizer_kwargs={'maxiter': 100//iter},
                   vqe_options={'samples': self.quant_meas_num},
                    rand_seed=1234)

        nus, gammas = partial_QAOA.get_angles()

        program = partial_QAOA.get_parameterized_program()
        return nus, gammas, program, 1

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def train(self, DATA, learning_rate=0.1, n_epochs=100, quantum_percentage=1.0, classical_percentage=0.0):
        """
        Train an RBM with mixture of quantum and classical update rules
        Params
        -------------------------------------------------------------------------
        DATA:                   (list) matrix with rows as data samples
        learning_rate:        (float) the learning rate used in the update rule by the rbm good value is 0.1
        n_epochs:               (int) number of weight update loops to do over RBM weights
        quantum_percentage:   (float) fraction of update rule to be dictated by quantum circuit
        classical_percentage: (float) fraction of update rule to be dictated by classical CD-1
        --------------------------------------------------------------------------
        NOTE: quantum_percentage + classical_percentage =1.0 must hold!!!
        """

        assert(quantum_percentage + classical_percentage == 1.0)

        DATA = np.asarray(DATA)

        assert(len(DATA[0]) <= self.visible_units)

        for epoch in range(n_epochs):

            print('Epoch: ', epoch)

            # Indices
            visible_indices = [i for i in range(self.visible_units)]
            hidden_indices = [i + self.visible_units for i in range(self.hidden_units)]
            total_indices = [i for i in range(self.total_units)]

            new_weights = deepcopy(self.WEIGHTS)
            if self.BIAS is not None:
                new_bias = deepcopy(self.BIAS)

            unc_nus, unc_gammas, unc_para_prog, _ = self.make_unclamped_QAOA()
            unc_mod_samp_prog = unc_para_prog(np.hstack((unc_nus, unc_gammas)))

            print('Found model expectation program')

            unc_neg_phase_quant = np.zeros_like(self.WEIGHTS)

            for i in range(self.visible_units):
                for j in range(self.hidden_units):
                    model_expectation = self.vqe_inst.expectation(unc_mod_samp_prog,
                                                   sZ(visible_indices[i]) * sZ(hidden_indices[j]),
                                                   self.quant_meas_num,
                                                   self.qvm)

                    unc_neg_phase_quant[i][j] = model_expectation

            unc_neg_phase_quant *= (1. / float(len(DATA)))

            if self.BIAS is not None:
                unc_neg_phase_quant_bias = np.zeros_like(self.BIAS)
                for i in range(self.hidden_units):
                    model_expectation = self.vqe_inst.expectation(unc_mod_samp_prog,
                                                            sZ(hidden_indices[i]),
                                                            self.quant_meas_num,
                                                            self.qvm)
                    unc_neg_phase_quant_bias[i] = model_expectation
            
                unc_neg_phase_quant_bias *= (1. / float(len(DATA)))

            pos_hidden_probs = self.sigmoid(np.dot(DATA, self.WEIGHTS))
            pos_hidden_states = pos_hidden_probs > np.random.rand(len(DATA), self.hidden_units)
            pos_phase_classical = np.dot(DATA.T, pos_hidden_probs) * 1./len(DATA)
            
            c_pos_phase_quant = np.zeros_like(self.WEIGHTS)
            if self.BIAS is not None:
                c_pos_phase_quant_bias = np.zeros_like(self.BIAS)
            
            if not self.reduced:

                iter_dat = len(DATA)

                for data in DATA:
                    c_nus, c_gammas, c_para_prog, _ = self.make_clamped_QAOA(
                        data_point=data, iter=iter_dat)
                    c_mod_samp_prog = c_para_prog(np.hstack((c_nus, c_gammas)))

                    print('Found model expectation program')

                    ct_pos_phase_quant = np.zeros_like(self.WEIGHTS)

                    for i in range(self.visible_units):
                        for j in range(self.hidden_units):
                            model_expectation = self.vqe_inst.expectation(c_mod_samp_prog,
                                                                          sZ(visible_indices[i]) * sZ(
                                                                              hidden_indices[j]),
                                                                          self.quant_meas_num,
                                                                          self.qvm)

                            ct_pos_phase_quant[i][j] = model_expectation
                    c_pos_phase_quant += ct_pos_phase_quant

                    if self.BIAS is not None:
                        ct_pos_phase_quant_bias = np.zeros_like(self.BIAS)
                        for i in range(self.hidden_units):
                            model_expectation = self.vqe_inst.expectation(c_mod_samp_prog,
                                                                          sZ(hidden_indices[j]),
                                                                          self.quant_meas_num,
                                                                          self.qvm)
                            ct_pos_phase_quant_bias[i] = model_expectation
                        c_pos_phase_quant_bias += ct_pos_phase_quant_bias

                c_pos_phase_quant *= (1. / float(len(DATA)))
                if self.BIAS is not None:
                    c_pos_phase_quant_bias *= (1. / float(len(DATA)))

            
            neg_visible_activations = np.dot(pos_hidden_states, self.WEIGHTS.T)
            neg_visible_probs = self.sigmoid(neg_visible_activations)
            
            neg_hidden_activations = np.dot(neg_visible_probs, self.WEIGHTS)
            neg_hidden_probs = self.sigmoid(neg_hidden_activations)
            
            neg_phase_classical = np.dot(
                neg_visible_probs.T, neg_hidden_probs) * 1./len(DATA)
            
            new_weights += learning_rate * \
                (classical_percentage * (pos_phase_classical - neg_phase_classical) + \
                 quantum_percentage * (c_pos_phase_quant - unc_neg_phase_quant))


            
            print(self.BIAS)
            '''
            if self.BIAS is not None:
                new_bias = new_bias + learning_rate * \
                    (classical_percentage * (pos_phase_classical - neg_phase_classical) + \
                     quantum_percentage * (c_pos_phase_quant_bias - unc_neg_phase_quant_bias))
            '''
                           
            self.WEIGHTS = deepcopy(new_weights)
            
            if self.BIAS is not None:
                self.BIAS = deepcopy(new_bias)
                print(self.BIAS)

            with open("RBM_info.txt", "w") as f:
                np.savetxt(f,self.WEIGHTS)
                if self.BIAS is not None:
                    np.savetxt(f,self.BIAS)
                
            with open("RBM_history.txt", "a") as f:
                np.savetxt(f, self.WEIGHTS)
                if self.BIAS is not None:
                    np.savetxt(f, self.BIAS)
                f.write(str('*'*72) + '\n')

        print('Training Done!')

    def transform(self, DATA):
        return self.sigmoid(np.dot(DATA, self.WEIGHTS))


    
if __name__ == "__main__":
    
    qvm = api.QVMConnection()

    r = QBM(qvm, num_visible=4, num_hidden=1,
             quant_meas_num=None, bias=False, reduced=False)

    simple_data = [[1, 1, -1, -1], [1, 1, -1, -1], [-1, -1, 1, 1], [-1, -1, 1, 1]]

    r.train(simple_data, n_epochs=100, quantum_percentage=0.7, classical_percentage=0.3)

    # transorm down to 1 dimension to see how we did.
    print(r.transform(simple_data))
