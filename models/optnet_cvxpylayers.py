import torch
import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp

class OptNet(nn.Module):

    '''
    board_size: (int) for a 9*9 sudoku grid with 3*3 subgrid, we take
                      board_size=9, and denote this variable as k

    input_tensors in forward(-) come in shape: batch*k*k*k where the
    final axis is the one-hot encoding of the numerical value in the board,
    in the forward pass though we flatten into batch*(k^3)

    We define a quadratic convex subproblem as the layer with parameters:
        - Q \in R^{(k^3)*(k^3)}
        - G \in R^{g_dim*(k^3)}
        - h \in R^{g_dim}
        - A \in R^{a_dim*(k^3)}
        - b \in R^{a_dim}
        - q \in R^{k^3} - for ease this is just the flattened board vector from input
                          or previous hidden layer (in theory all of these params
                          could be dependent on the flattened board vector though)

    With input vector z \in R^{k^3} this translates to the following quadratic
    problem:

        argmin_z (1/2)*z.T*Q*z + q.T*z
        subject to A*z = b and G*z <= h

    In the following example: https://notebook.community/locuslab/qpth/example-sudoku
    they take g_dim = k^3 and a_dim = 40 (unclear why 40 is working)
    '''
    def __init__(self, board_size, g_dim, a_dim, q_penalty=1e-3):
        flat_board_size = board_size**3

        # these definitions are lifted from the example cited above:
        self.Q_sqrt = nn.Parameter(q_penalty*torch.eye(flat_board_size, dtype=torch.float32))
        self.G = nn.Parameter(-torch.eye(flat_board_size, dtype=torch.float32))
        self.h = nn.Parameter(torch.zeros(flat_board_size, dtype=torch.float32))
        self.A = nn.Parameter(torch.rand((a_dim, flat_board_size), dtype=torch.float32))
        self.b = nn.Parameter(torch.ones(a_dim, dtype=torch.float32))

        z = cp.Variable(flat_board_size)
        Q_sqrt = cp.Parameter((flat_board_size, flat_board_size))
        G = cp.Parameter((flat_board_size, flat_board_size))
        h = cp.Parameter(flat_board_size)
        A = cp.Parameter((a_dim, flat_board_size))
        b = cp.Parameter(a_dim)
        q = cp.Parameter(flat_board_size)

        objective = cp.Minimize(0.5*cp.sum_squares(Q_sqrt@z) + q.T@z)
        constraints = [
            A@z == b,
            G@z <= h
        ]
        prob = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(prob, parameters=[Q_sqrt, q, A, b, G, h],
                                variables =[z])


    def forward(self, z_prev):
        # todo, fill in forward pass
        pass
