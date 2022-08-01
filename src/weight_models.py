import sys
import subprocess
import numpy as np
import pandas as pd
from contextlib import suppress
import math
from scipy.stats import norm
from sklearn.preprocessing import *
import timeit

# Functions for common mathematical operations
def sigmoid_array(x):
   return 1 / (1 + np.exp(-x))

def std_norm_exp(x):
   return np.exp(-np.square(x)/2)

def zs(yhat, xs, hs):
    # Returns ndim(yhat) * ndim(xs)
    return np.divide(np.expand_dims(yhat,yhat.ndim) - xs, hs)

def get_encoding(encoding, condition, metadata):
    if encoding == "contbin":
        return Contbin(condition, metadata)
    elif encoding == "cont":
        return Cont(condition, metadata)
    elif encoding == "bicontbin":
        return Bicontbin(condition, metadata)
    elif encoding == "cat":
        return Cat(condition, metadata)
    else:
        raise ValueError("Encoding not allowed: " + encoding)

# Categorical variables
class Cat:
    # Initalize weights and store variables in a useful way
    def __init__(self, condition, metadata, alpha = 0.1, beta = 0.01, w_0 = -1, w_1 = 3, done_updating = False):
        self.w_0 = w_0
        self.w_1 = w_1
        self.alpha = alpha
        self.beta = beta

        vec = OneHotEncoder(handle_unknown='ignore')
        vec.fit([["microdialysis"],["vapor diffusion"],["batch"],["lipidic cubic phase"],["evaporation"],["counter diffusion"],["liquid diffusion"],["microfluidic"],["cooling"],["cell"],["other"],["soaking"],["dialysis"],["seeding"]])

        self.condition_dict = dict(zip([seq_id for seq_id in metadata['seq_ID'].tolist()], [vec.transform([[item]]).toarray() if pd.notnull(item) else None for item in metadata[condition].tolist()]))
        self.done_updating = done_updating
        self.n = (len(metadata.index) - metadata[condition].isna().sum()) - 1

        # Overall mean term in each update
        self.mean_block = np.mean(np.array([value for _, value in self.condition_dict.items() if value is not None]), axis=0) * (self.n + 1) / self.n

    def set_alpha(self, alpha):
        self.alpha = float(alpha)

    # Update the weights with gradient descent using ID as the protein of interest, IDs as the similar proteins, ss as their sequence similarity, and n_p as the number of similar proteins
    def update(self, ID, IDs, ss, n_p):
        if self.condition_dict[ID] is None:
            return None
        else:
            y = np.argmax(self.condition_dict[ID]) # Only the index of the correct one matters here

        xs = [self.condition_dict[item][0][y] if self.condition_dict[item] is not None else None for item in IDs] # Get values for similar proteins
        if sum(x is not None for x in xs) > 0:
            xs, ss = zip(*[(x, s) for x, s in zip(xs, ss) if x is not None])
            xs, ss = np.array(xs), np.array(ss)
            n_p = len(xs)
        else:
            n_p = 0

        if n_p > 0:
            # Pieces of gradient
            sigmoids = sigmoid_array(self.w_1 * ss + self.w_0)
            sigminus = np.ones(n_p) - sigmoids
            sigmoids_x = np.multiply(xs, sigmoids)
            sigmoids_x_sigminus = np.multiply(sigmoids_x, sigminus)
            sigmoids_sigminus = np.multiply(sigmoids, sigminus)
            sum_sigmoids = np.sum(sigmoids)
            sum_sigmoids_x = np.sum(sigmoids_x)
            yhat = 1/(n_p + 1) * (self.mean_block[0][y] - 1/self.n) + n_p/(n_p + 1) * sum_sigmoids_x/sum_sigmoids
            first_chunk = 1/yhat

            # Compute gradient
            dldw0 = -first_chunk * (sum_sigmoids * np.sum(sigmoids_x_sigminus) - sum_sigmoids_x * np.sum(sigmoids_sigminus)) / np.square(sum_sigmoids)
            dldw1 = -first_chunk * (sum_sigmoids * np.sum(np.multiply(sigmoids_x_sigminus, ss)) - sum_sigmoids_x * np.sum(np.multiply(sigmoids_sigminus, ss))) / np.square(sum_sigmoids)

            # Update weights
            self.w_0 = self.w_0 - self.alpha * dldw0 - 2 * self.w_0 * self.beta
            self.w_1 = self.w_1 - self.alpha * dldw1 - 2 * self.w_1 * self.beta

            # Compute and return loss
            loss = -np.log(yhat)
            if loss is np.nan:
                raise ValueError("Invalid loss from categorical predictor")
            else:
                return loss
        else:
            loss = -np.log(self.mean_block[0][y] - 1/self.n)
            return loss

# Continuous variables
class Cont:
    # Initalize weights and store variables in a useful way
    def __init__(self, condition, metadata, alpha = 0.1, beta = 0.01, w_0 = -1, w_1 = -2, c = 5, done_updating = False):
        self.w_0 = w_0
        self.w_1 = w_1
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.condition_dict = dict(zip([seq_id for seq_id in metadata['seq_ID'].tolist()], [float(item) if pd.notnull(item) else None for item in metadata[condition].tolist()]))
        self.done_updating = done_updating
        self.n = (len(metadata.index) - metadata[condition].isna().sum()) - 1

        all_x = [value for _, value in self.condition_dict.items() if value is not None]
        self.delta = np.std(np.array(all_x)) / 5

        all_x = np.array([value for key, value in self.condition_dict.items() if value is not None])
        self.xbar = np.mean(all_x)
        self.eta = np.std(all_x)
        del(all_x)

    def set_alpha(self, alpha):
        self.alpha = float(alpha)

    # Update the weights with gradient descent using ID as the protein of interest, IDs as the similar proteins, ss as their sequence similarity, and n_p as the number of similar proteins
    def update(self, ID, IDs, ss, n_p):
        if self.condition_dict[ID] is None:
            return None
        else:
            y = self.condition_dict[ID]

        xs = [self.condition_dict[item] for item in IDs] # Get values for similar proteins
        if sum(x is not None for x in xs) > 0:
            xs, ss = zip(*[(x, s) for x, s in zip(xs, ss) if x is not None])
            xs, ss = np.array(xs), np.array(ss)
            n_p = len(xs)
        else:
            n_p = 0

        xbar = self.xbar
        eta = self.eta

        if n_p > 0:
            # Pieces of yhats
            sigmoids = sigmoid_array(self.w_1 * ss + self.w_0)
            U = (np.log(np.exp(-self.w_1) + np.exp(self.w_0)) - np.log(1 + np.exp(self.w_0))) / self.w_1 + 1
            hs = self.c * sigmoids / U
            norm_c = 1/(2 * math.pi * eta * (n_p + 1))

            # Calculate loss with integral approximation
            int_space = np.arange(y-self.delta, y+self.delta * 101/100, self.delta / 100)
            std_norm_piece = std_norm_exp((int_space - xbar)/eta)
            std_phis = np.multiply(1/hs, std_norm_exp(zs(int_space, xs, hs)))
            loss = 1 - np.sum(norm_c * std_norm_piece + 1/((n_p + 1) * 2 * math.pi) * np.sum(std_phis, axis=1)) * self.delta / 100

            # Calculate gradients
            norm_d = 1/((n_p + 1) * 2 * math.pi)
            m = np.exp(self.w_0) + np.exp(-self.w_1)

            dfydh = np.multiply(1 / np.square(hs), np.multiply(std_norm_exp(zs(int_space, xs, hs)), np.square(zs(int_space, xs, hs)) - 1)) # ndim(yhat) * ndim(xs)
            dhdw0 = np.add(self.c/U * np.multiply(sigmoids, 1-sigmoids), -np.multiply(self.c/np.multiply(self.w_1, np.square(U)) * sigmoids, np.exp(self.w_0) / m - (np.exp(self.w_0) / (1+np.exp(self.w_0))))) # ndim(xs)
            dhdw1 = np.add(np.multiply(ss, self.c/U * np.multiply(sigmoids, 1-sigmoids)), -np.multiply(self.c/np.square(np.multiply(self.w_1, U)) * sigmoids, -self.w_1 * np.exp(-self.w_1) / m - (np.log(m) - np.log(1 + np.exp(self.w_0))))) # ndim(xs)
            dhdc = hs/self.c # ndim(xs)

            dldw0 = - np.sum(norm_d * np.multiply(dfydh, dhdw0)) * self.delta / 100
            dldw1 = - np.sum(norm_d * np.multiply(dfydh, dhdw1)) * self.delta / 100
            dldc = - np.sum(norm_d * np.multiply(dfydh, dhdc)) * self.delta / 100 + 2 * (self.c - eta) * self.beta

            self.w_0 = self.w_0 - self.alpha * dldw0
            self.w_1 = self.w_1 - self.alpha * dldw1
            self.c = self.c - self.alpha * dldc

        else:
            loss = 1 - np.sum(1/(2 * math.pi * eta) * std_norm_exp(zs(np.arange(y-self.delta, y+self.delta * 101/100, self.delta / 100), xbar, eta)[0])) * self.delta / 100

        if loss > 1 or loss < 0:
            raise ValueError("Loss error: out of bounds")
        if loss is np.nan:
            raise ValueError("Invalid loss from continuous predictor")

        return loss

class Contbin:
    # Initalize weights and store variables in a useful way
    def __init__(self, condition, metadata, alpha = 0.1, beta_cont = 0.01, beta_bin = 0.01, w_0_cont = -1, w_1_cont = -2, c = 5, w_0_bin = -1, w_1_bin = 3, done_updating = False):
        # Initialize continuous parameters
        self.w_0_cont = w_0_cont
        self.w_1_cont = w_1_cont
        self.c = c
        self.alpha = alpha
        self.beta_cont = beta_cont
        self.condition_dict = dict(zip([seq_id for seq_id in metadata['seq_ID'].tolist()], [(float(item), float(item)!=0) if pd.notnull(item) else None for item in metadata[condition].tolist()]))
        self.done_updating = done_updating
        self.n = (len(metadata.index) - metadata[condition].isna().sum()) - 1

        all_x = [value[0] for key, value in self.condition_dict.items() if value is not None]
        self.delta = np.std(np.array(all_x)) / 5

        # Initialize binary parameters
        self.w_0_bin = w_0_bin
        self.w_1_bin = w_1_bin
        self.beta_bin = beta_bin

        # Overall mean term in each update
        self.mean_block = np.mean(np.array([value[1] for _, value in self.condition_dict.items() if value is not None]), axis=0) * (self.n + 1) / self.n
        all_x = np.array([value[0] for key, value in self.condition_dict.items() if value is not None])
        self.xbar = np.mean(all_x)
        self.eta = np.std(all_x)
        del(all_x)

    def set_alpha(self, alpha):
        self.alpha = float(alpha)

    # Update the weights with gradient descent using ID as the protein of interest, IDs as the similar proteins, ss as their sequence similarity, and n_p as the number of similar proteins
    def update(self, ID, IDs, ss, n_p):
        if self.condition_dict[ID] is None:
            return None
        else:
            y = self.condition_dict[ID]

        xbar = self.xbar
        eta = self.eta

        xs = [self.condition_dict[item] for item in IDs] # Get values for similar proteins
        if sum(x is not None for x in xs) > 0:
            xs_0, ss = zip(*[(x[0], s) for x, s in zip(xs, ss) if x is not None])
            xs_0, ss = np.array(xs_0), np.array(ss)
            n_p = len(xs_0)
        else:
            n_p = 0

        if n_p > 0:

            # Continuous updates if the condition is present
            if y[1]:
                # Pieces of yhats
                sigmoids = sigmoid_array(self.w_1_cont * ss + self.w_0_cont)
                U = (np.log(np.exp(-self.w_1_cont) + np.exp(self.w_0_cont)) - np.log(1 + np.exp(self.w_0_cont))) / self.w_1_cont + 1
                hs = self.c * sigmoids / U
                norm_c = 1/(2 * math.pi * eta * (n_p + 1))

                # Calculate loss with integral approximation
                int_space = np.arange(y[0] * 0.899, y[0] * 1.101, y[0] / 500)
                std_norm_piece = std_norm_exp((int_space - xbar)/eta)
                std_phis = np.multiply(1/hs, std_norm_exp(zs(int_space, xs_0, hs)))
                loss_cont = 1 - np.sum(norm_c * std_norm_piece + 1/((n_p + 1) * 2 * math.pi) * np.sum(std_phis, axis=1)) * y[0] / 501

                # Calculate gradients
                norm_d = 1/((n_p + 1) * 2 * math.pi)
                m = np.exp(self.w_0_cont) + np.exp(-self.w_1_cont)

                dfydh = np.multiply(1 / np.square(hs), np.multiply(std_norm_exp(zs(int_space, xs_0, hs)), np.square(zs(int_space, xs_0, hs)) - 1)) # ndim(yhat) * ndim(xs)
                dhdw0 = np.add(self.c/U * np.multiply(sigmoids, 1-sigmoids), -np.multiply(self.c/np.multiply(self.w_1_cont, np.square(U)) * sigmoids, np.exp(self.w_0_cont) / m - (np.exp(self.w_0_cont) / (1+np.exp(self.w_0_cont))))) # ndim(xs)
                dhdw1 = np.add(np.multiply(ss, self.c/U * np.multiply(sigmoids, 1-sigmoids)), -np.multiply(self.c/np.square(np.multiply(self.w_1_cont, U)) * sigmoids, -self.w_1_cont * np.exp(-self.w_1_cont) / m - (np.log(m) - np.log(1 + np.exp(self.w_0_cont))))) # ndim(xs)
                dhdc = hs/self.c # ndim(xs)

                dldw0 = - np.sum(norm_d * np.multiply(dfydh, dhdw0)) * y[0] / 501
                dldw1 = - np.sum(norm_d * np.multiply(dfydh, dhdw1)) * y[0] / 501
                dldc = - np.sum(norm_d * np.multiply(dfydh, dhdc)) * y[0] / 501 + 2 * (self.c - eta) * self.beta_cont

                self.w_0_cont = self.w_0_cont - self.alpha * dldw0
                self.w_1_cont = self.w_1_cont - self.alpha * dldw1
                self.c = self.c - self.alpha * dldc
            else:
                loss_cont = 0

            # Binary updates
            xs_1 = np.array([x[1] for x in xs if x is not None])

            # Pieces of gradient
            sigmoids = sigmoid_array(self.w_1_bin * ss + self.w_0_bin)
            sigminus = np.ones(n_p) - sigmoids
            sigmoids_x = np.multiply(xs_1, sigmoids)
            sigmoids_x_sigminus = np.multiply(sigmoids_x, sigminus)
            sigmoids_sigminus = np.multiply(sigmoids, sigminus)
            sum_sigmoids = np.sum(sigmoids)
            sum_sigmoids_x = np.sum(sigmoids_x)
            yhat = 1/(n_p + 1) * (self.mean_block - y[1]/self.n) + n_p/(n_p + 1) * sum_sigmoids_x/sum_sigmoids
            first_chunk = 1/yhat

            # Compute gradient
            dldw0 = -first_chunk * (sum_sigmoids * np.sum(sigmoids_x_sigminus) - sum_sigmoids_x * np.sum(sigmoids_sigminus)) / np.square(sum_sigmoids)
            dldw1 = -first_chunk * (sum_sigmoids * np.sum(np.multiply(sigmoids_x_sigminus, ss)) - sum_sigmoids_x * np.sum(np.multiply(sigmoids_sigminus, ss))) / np.square(sum_sigmoids)

            # Update weights
            self.w_0_bin = self.w_0_bin - self.alpha * dldw0 - 2 * self.w_0_bin * self.beta_bin
            self.w_1_bin = self.w_1_bin - self.alpha * dldw1 - 2 * self.w_1_bin * self.beta_bin

            # Compute and return loss
            loss_bin = -np.log(yhat)

        else:
            if y[1]:
                loss_cont = 1 - np.sum(1/(2 * math.pi * eta) * std_norm_exp(zs(np.arange(y[0] * 0.899, y[0] * 1.101, y[0] / 500), xbar, eta)[0])) * y[0] / 501
            else:
                loss_cont = 0
            loss_bin = -np.log(self.mean_block - y[1]/self.n)

        if loss_cont > 1.01 or loss_cont < -0.01:
            raise ValueError("Loss error: out of bounds")
        if loss_cont is np.nan:
            raise ValueError("Invalid loss from continuous predictor")
        if loss_bin is np.nan:
            raise ValueError("Invalid loss from categorical predictor")

        return (loss_cont, loss_bin)

def zs2d(yhat, xs_1, xs_2, hs_1, hs_2):
    return np.divide(np.expand_dims(yhat,2) - [xs_1,xs_2], [hs_1,hs_2])

def std_norm_exp2d(x):
   return np.exp(-np.sum(np.square(x), axis=1)/2)

class Bicontbin:
    # Initalize weights and store variables in a useful way
    def __init__(self, condition, metadata, alpha = 0.01, beta_cont = 0.01, beta_bin = 0.01, w_10_cont = -1, w_11_cont = -2, c_1 = 5, w_20_cont = -1, w_21_cont = -2, c_2 = 5, w_0_bin = -1, w_1_bin = 3, done_updating = False):
        # Initialize continuous parameters
        self.w_10_cont = w_10_cont
        self.w_11_cont = w_11_cont
        self.c_1 = c_1
        self.w_20_cont = w_20_cont
        self.w_21_cont = w_21_cont
        self.c_2 = c_2
        self.alpha = alpha
        self.beta_cont = beta_cont
        self.condition_dict = dict(zip([seq_id for seq_id in metadata['seq_ID'].tolist()],
            [None if pd.isnull(item) else (float(item.split(" , ")[0]), float(item.split(" , ")[1]), 1) if item != "0" else (0, 0, 0) for item in metadata[condition].tolist()]))
        self.done_updating = done_updating
        self.n = (len(metadata.index) - metadata[condition].isna().sum()) - 1

        # Initialize binary parameters
        self.w_0_bin = w_0_bin
        self.w_1_bin = w_1_bin
        self.beta_bin = beta_bin

        # Overall mean term in each update
        self.mean_block = np.mean(np.array([value[2] for _, value in self.condition_dict.items() if value is not None]), axis=0) * (self.n + 1) / self.n

        # Set delta range for fitting
        all_x_1, all_x_2 = zip(*[(value[0], value[1]) for _, value in self.condition_dict.items() if value is not None])
        self.delta_1 = np.std(np.array(all_x_1)) / 5
        self.delta_2 = np.std(np.array(all_x_2)) / 5

        self.xbar_1 = np.mean(np.array(all_x_1))
        self.eta_1 = np.std(np.array(all_x_1))
        del(all_x_1)

        self.xbar_2 = np.mean(np.array(all_x_2))
        self.eta_2 = np.std(np.array(all_x_2))
        del(all_x_2)

    def set_alpha(self, alpha):
        self.alpha = float(alpha)

    # Update the weights with gradient descent using ID as the protein of interest, IDs as the similar proteins, ss as their sequence similarity, and n_p as the number of similar proteins
    def update(self, ID, IDs, ss, n_p):

        if self.condition_dict[ID] is None:
            return None
        else:
            y = self.condition_dict[ID]

        xbar_1 = self.xbar_1
        xbar_2 = self.xbar_2
        eta_1 = self.eta_1
        eta_2 = self.eta_2

        xs = [self.condition_dict[item] for item in IDs] # Get values for similar proteins
        if sum(x is not None for x in xs) > 0:
            xs_1, xs_2, ss = zip(*[(x[0], x[1], s) for x, s in zip(xs, ss) if x is not None])
            xs_1, xs_2, ss = np.array(xs_1), np.array(xs_2), np.array(ss)
            n_p = len(xs_1)
        else:
            n_p = 0

        if n_p > 0:
            # Continuous 1 updates if present
            if y[2]:
                sigmoids_1 = sigmoid_array(self.w_11_cont * ss + self.w_10_cont)
                U_1 = (np.log(np.exp(-self.w_11_cont) + np.exp(self.w_10_cont)) - np.log(1 + np.exp(self.w_10_cont))) / self.w_11_cont + 1
                hs_1 = self.c_1 * sigmoids_1 / U_1

                sigmoids_2 = sigmoid_array(self.w_21_cont * ss + self.w_20_cont)
                U_2 = (np.log(np.exp(-self.w_21_cont) + np.exp(self.w_20_cont)) - np.log(1 + np.exp(self.w_20_cont))) / self.w_21_cont + 1
                hs_2 = self.c_2 * sigmoids_2 / U_2

                norm_c = 1/(2 * math.pi * eta_1 * eta_2 * (n_p + 1))

                # Calculate loss with integral approximation
                int_space = np.mgrid[(y[0] * 0.899):(y[0] * 1.101 + y[0]/1000):(y[0] / 500), (y[1] * 0.899):(y[1] * 1.101 + y[0]/1000):(y[1] / 500)].reshape(2,-1).T
                std_norm_piece = std_norm_exp2d(zs2d(int_space, [xbar_1], [xbar_2], [eta_1], [eta_2]))
                std_phis = np.divide(std_norm_exp2d(zs2d(int_space, xs_1, xs_2, hs_1, hs_2)), np.multiply(hs_1, hs_2))
                loss_cont = 1 - np.sum(norm_c * np.sum(std_norm_piece, axis=1) + 1/((n_p + 1) * 2 * math.pi) * np.sum(std_phis, axis=1)) * y[0] / 501 * y[1] / 501

                # Calculate gradients
                norm_d = 1/((n_p + 1) * 2 * math.pi)
                m_1 = np.exp(self.w_10_cont) + np.exp(-self.w_11_cont)

                dfydh1 = 1 / hs_2 * 1 / np.square(hs_1) * np.multiply(std_norm_exp(zs(int_space, xs_2, hs_2)), np.multiply(std_norm_exp(zs(int_space, xs_1, hs_1)), np.square(zs(int_space, xs_1, hs_1)) - 1))
                dhdw10 = np.add(self.c_1/U_1 * np.multiply(sigmoids_1, 1-sigmoids_1), -np.multiply(self.c_1/np.multiply(self.w_11_cont, np.square(U_1)) * sigmoids_1, np.exp(self.w_10_cont) / m_1 - (np.exp(self.w_10_cont) / (1+np.exp(self.w_10_cont)))))
                dhdw11 = np.add(np.multiply(ss, self.c_1/U_1 * np.multiply(sigmoids_1, 1-sigmoids_1)), -np.multiply(self.c_1/np.square(np.multiply(self.w_11_cont, U_1)) * sigmoids_1, -self.w_11_cont * np.exp(-self.w_11_cont) / m_1 - (np.log(m_1) - np.log(1 + np.exp(self.w_10_cont)))))
                dhdc1 = hs_1/self.c_1 # ndim(xs)

                dldw10 = - np.sum(norm_d * np.multiply(dfydh1, dhdw10)) * y[0] / 501 * y[1] / 501
                dldw11 = - np.sum(norm_d * np.multiply(dfydh1, dhdw11)) * y[0] / 501 * y[1] / 501
                dldc1 = - np.sum(norm_d * np.multiply(dfydh1, dhdc1)) * y[0] / 501 * y[1] / 501 + 2 * (self.c_1 - eta_1) * self.beta_cont

                self.w_10_cont = self.w_10_cont - self.alpha * dldw10
                self.w_11_cont = self.w_11_cont - self.alpha * dldw11
                self.c_1 = self.c_1 - self.alpha * dldc1

                # Continuous 2 updates

                # Calculate gradients
                m_2 = np.exp(self.w_20_cont) + np.exp(-self.w_21_cont)

                dfydh2 = 1 / hs_1 * 1 / np.square(hs_2) * np.multiply(std_norm_exp(zs(int_space, xs_1, hs_1)), np.multiply(std_norm_exp(zs(int_space, xs_2, hs_2)), np.square(zs(int_space, xs_2, hs_2)) - 1))
                dhdw20 = np.add(self.c_2/U_2 * np.multiply(sigmoids_2, 1-sigmoids_2), -np.multiply(self.c_2/np.multiply(self.w_21_cont, np.square(U_2)) * sigmoids_2, np.exp(self.w_20_cont) / m_2 - (np.exp(self.w_20_cont) / (1+np.exp(self.w_20_cont)))))
                dhdw21 = np.add(np.multiply(ss, self.c_2/U_2 * np.multiply(sigmoids_2, 1-sigmoids_2)), -np.multiply(self.c_2/np.square(np.multiply(self.w_21_cont, U_2)) * sigmoids_2, -self.w_21_cont * np.exp(-self.w_21_cont) / m_2 - (np.log(m_2) - np.log(1 + np.exp(self.w_20_cont)))))
                dhdc2 = hs_2/self.c_2 # ndim(xs)

                dldw20 = - np.sum(norm_d * np.multiply(dfydh2, dhdw20)) * y[0] / 501 * y[1] / 501
                dldw21 = - np.sum(norm_d * np.multiply(dfydh2, dhdw21)) * y[0] / 501 * y[1] / 501
                dldc2 = - np.sum(norm_d * np.multiply(dfydh2, dhdc2)) * y[0] / 501 * y[1] / 501 * (self.c_2 - eta_2) * self.beta_cont

                self.w_20_cont = self.w_20_cont - self.alpha * dldw20
                self.w_21_cont = self.w_21_cont - self.alpha * dldw21
                self.c_2 = self.c_2 - self.alpha * dldc2
            else:
                loss_cont = 0


            # Binary updates
            xs = [x[2] for x in xs if x is not None]

            # Pieces of gradient
            sigmoids = sigmoid_array(self.w_1_bin * ss + self.w_0_bin)
            sigminus = np.ones(n_p) - sigmoids
            sigmoids_x = np.multiply(xs, sigmoids)
            sigmoids_x_sigminus = np.multiply(sigmoids_x, sigminus)
            sigmoids_sigminus = np.multiply(sigmoids, sigminus)
            sum_sigmoids = np.sum(sigmoids)
            sum_sigmoids_x = np.sum(sigmoids_x)
            yhat = 1/(n_p + 1) * (self.mean_block - y[2]/self.n) + n_p/(n_p + 1) * sum_sigmoids_x/sum_sigmoids
            first_chunk = 1/yhat

            # Compute gradient
            dldw0 = -first_chunk * (sum_sigmoids * np.sum(sigmoids_x_sigminus) - sum_sigmoids_x * np.sum(sigmoids_sigminus)) / np.square(sum_sigmoids)
            dldw1 = -first_chunk * (sum_sigmoids * np.sum(np.multiply(sigmoids_x_sigminus, ss)) - sum_sigmoids_x * np.sum(np.multiply(sigmoids_sigminus, ss))) / np.square(sum_sigmoids)

            # Update weights
            self.w_0_bin = self.w_0_bin - self.alpha * dldw0 - 2 * self.w_0_bin * self.beta_bin
            self.w_1_bin = self.w_1_bin - self.alpha * dldw1 - 2 * self.w_1_bin * self.beta_bin

            # Compute and return loss
            loss_bin = -np.log(yhat)

        else:
            if y[2]:
                norm_c = 1/(2 * math.pi * eta_1 * eta_2 * (n_p + 1))
                int_space = np.mgrid[(y[0] * 0.899):(y[0] * 1.101 + y[0]/1000):(y[0] / 500), (y[1] * 0.899):(y[1] * 1.101 + y[0]/1000):(y[1] / 500)].reshape(2,-1).T
                std_norm_piece = std_norm_exp2d(zs2d(int_space, [xbar_1], [xbar_2], [eta_1], [eta_2]))
                loss_cont = 1 - np.sum(norm_c * np.sum(std_norm_piece, axis=1)) * y[0] / 501 * y[1] / 501
            else:
                loss_cont = 0
            loss_bin = -np.log(self.mean_block - y[2]/self.n)

        if loss_cont is np.nan:
            raise ValueError("Invalid loss from continuous predictor")
        if loss_cont > 1.01 or loss_cont < -0.01:
            raise ValueError("Loss error: out of bounds: " + str(loss_cont))
        if loss_bin is np.nan:
            raise ValueError("Invalid loss from categorical predictor")

        return (loss_cont, loss_bin)
