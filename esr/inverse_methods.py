import numpy as np
import matplotlib.pyplot as plt
import esr
from numpy import linalg as LA
from decimal import Decimal


class InverseMethod:

    def __init__(self, model, M):
        """
        Parent class for inverse methods.
        Args:
             M: Noisy measurements (M/EEG signals).
             model: An instance of class Model.
        Raises:
             TypeError: If model is none or not an instance of class Model.
                        If M is none.
             ValueError: Number of rows in the forward operator has to match
             the number of rows in M.
             NotImplementedError: If compute method is not implemented in
             the child class.
        """
        if model is None:
            raise TypeError('\'model\' cannot be None.')

        if not isinstance(model, esr.Model):
            raise TypeError("The model you provided has to be an instance of "
                            "class Model.")
        if M is None:
            raise TypeError('\'M\' cannot be None.')

        # Raise an error if model and measurements are not consistent.
        if model.forward.shape[0] != M.shape[0]:
            raise ValueError("Number of rows in forward operator must be the "
                             "same as the number of rows in M (matrix of"
                             "measurements)")

        self.model = model
        self.M = M

    def compute(self):
        raise NotImplementedError


class PseudoInverse(InverseMethod):
    """Pseudoinverse solver."""

    def compute(self):
        """ Compute the approximation of source activations using pseudoinverse
        soulution."""

        G = np.asarray(self.model.forward)
        Gp = np.linalg.pinv(G)
        Xh = np.dot(Gp, self.M)

        return Xh


def tikhonov(G, lambda_):
    Gt = np.dot(G.T, LA.pinv(np.dot(G, G.T)
                             + (lambda_ ** 2) * np.eye(G.shape[0])))
    return Gt


class TikhonovInverse(InverseMethod):

    def __init__(self, model, M, pl = 0):
        """
        Tikhonov inverse solver.
        Args:
             pl (optional): A parameter for plotting the L-curve. By default
             pl = 0 and L-curve will not be plotted. If pl = 1 L-curve will be
             plotted.
        """
        if pl!=1:
            pl = 0

        super().__init__(model, M)
        self.pl = pl

    def compute(self):
        """Compute the approximation of source activations using Tikhonov
        regularization."""

        G = np.asarray(self.model.forward)
        A = np.dot(G, G.T)
        N = 30
        lambdas = np.logspace(-5, 0.0, num=N)

        A_gcv = [np.dot(A, LA.inv(A + (lam ** 2) * np.eye(A.shape[0])))
                 for lam in lambdas]
        Ia = np.eye(A_gcv[0].shape[0])

        GCV = np.asarray([LA.norm(np.dot((Ia - a_gcv), self.M)) ** 2 /
                          np.trace(Ia - a_gcv) ** 2 for a_gcv in A_gcv])

        lam_gcv_id = np.argmin(GCV)
        optimal_lam = lambdas[lam_gcv_id]
        olam_disp = '%.2e' % Decimal(optimal_lam)

        # L-curve
        pl = self.pl
        if pl == 1:
            X_hats = [np.dot(tikhonov(G, l), self.M) for l in lambdas]
            rn = np.asarray([LA.norm(self.M - np.dot(G, x_h))
                             for x_h in X_hats])
            sn = np.asarray([LA.norm(x_h) for x_h in X_hats])

            plt.figure(figsize=(8,5))
            plt.plot(rn, sn, color='gold')
            plt.xlabel('Residual norm' r'$\ ||M - G\hatX||_2$')
            plt.ylabel('Solution norm' r'$\  ||\hatX||_2$')
            plt.scatter(rn, sn, color='mediumvioletred', marker='o', s=15)
            plt.annotate(r'$\lambda$* = ' + str(olam_disp),
                         xy=(rn[lam_gcv_id], sn[lam_gcv_id]), xycoords='data',
                         xytext=(rn[lam_gcv_id] * 1.5, sn[lam_gcv_id]),
                         arrowprops=dict(arrowstyle='->'))
            tit = 'Optimal ' r'$\lambda$ ' 'obtained with GCV'
            plt.title(tit)
            plt.show()
        else:
            pass

        Gt = tikhonov(G, optimal_lam)
        Xht = np.dot(Gt, self.M)

        return Xht
