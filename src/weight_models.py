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

def get_encoding(encoding, condition, metadata, beta_cont, beta_bin, convergence_radius=0.1, convergence_length=20000, alpha=0.1):
    if encoding == "contbin":
        return Contbin(condition, metadata, beta_cont=beta_cont, beta_bin=beta_bin, convergence_radius=convergence_radius, convergence_length=convergence_length, alpha=alpha)
    elif encoding == "cont":
        return Cont(condition, metadata, beta=beta_cont, convergence_radius=convergence_radius, convergence_length=convergence_length, alpha=alpha)
    elif encoding == "bicontbin":
        return Bicontbin(condition, metadata, beta_cont=beta_cont, beta_bin=beta_bin, convergence_radius=convergence_radius, convergence_length=convergence_length, alpha=alpha)
    elif encoding == "cat":
        return Cat(condition, metadata, beta=beta_bin, convergence_radius=convergence_radius, convergence_length=convergence_length, alpha=alpha)
    else:
        raise ValueError("Encoding not allowed: " + encoding)

# Categorical variables
class Cat:
    # Initalize weights and store variables in a useful way
    def __init__(self, condition, metadata, alpha = 0.1, beta = 0.00, w_0 = -1, w_1 = 3, convergence_radius=0.1, convergence_length=20000, done_updating = False):
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

        # Store previous weights for checking convergence
        self.w_0_history = np.linspace(self.w_0 + convergence_radius * convergence_length, self.w_0, num=convergence_length)
        self.w_1_history = np.linspace(self.w_1 + convergence_radius * convergence_length, self.w_1, num=convergence_length)
        self.convergence_radius = convergence_radius
        self.convergence_length = convergence_length

        self.condition = condition

    def set_alpha(self, alpha):
        self.alpha = float(alpha)

    # Update the weights with gradient descent using ID as the protein of interest, IDs as the similar proteins, ss as their sequence similarity, and n_p as the number of similar proteins
    def update(self, ID, IDs, ss, n_p):

        if not self.done_updating:
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
                self.w_0 = self.w_0 - self.alpha * (dldw0 + 2 * self.w_0 * self.beta)
                self.w_1 = self.w_1 - self.alpha * (dldw1 + 2 * self.w_1 * self.beta)

                # Check for convergence
                self.w_0_history = np.roll(self.w_0_history, -1)
                self.w_0_history[self.convergence_length - 1] = self.w_0
                self.w_1_history = np.roll(self.w_1_history, -1)
                self.w_1_history[self.convergence_length - 1] = self.w_1

                if np.max(self.w_0_history) - np.min(self.w_0_history) < self.convergence_radius and np.max(self.w_1_history) - np.min(self.w_1_history) < self.convergence_radius:
                    self.done_updating = True
                    print("Done updating weights for " + self.condition)

                # Compute and return loss
                loss = -np.log(yhat)
                if loss is np.nan:
                    raise ValueError("Invalid loss from categorical predictor")
                else:
                    return loss
            else:
                loss = -np.log(self.mean_block[0][y] - 1/self.n)
                return loss
        else:
            # If done training
            return None

    def evaluate(self, ID, IDs, ss, n_p, CI): # CI not used for anything but allows general calling of the evaluation function
        if self.condition_dict[ID] is None:
            y = np.nan
        else:
            y = np.argmax(self.condition_dict[ID]) # Only the index of the correct one matters here

        xs = [self.condition_dict[item][0] if self.condition_dict[item] is not None else None for item in IDs] # Get values for similar proteins
        if sum(x is not None for x in xs) > 0:
            xs, ss = zip(*[(x, s) for x, s in zip(xs, ss) if x is not None])
            xs, ss = np.array(xs), np.array(ss)
            n_p = len(xs)
        else:
            n_p = 0

        if n_p > 0:
            sigmoids = sigmoid_array(self.w_1 * ss + self.w_0)
            sigmoids_x = np.multiply(xs.T, sigmoids).T
            sum_sigmoids = np.sum(sigmoids)
            sum_sigmoids_x = np.sum(sigmoids_x, axis=0)
            tmp_mean_block = self.mean_block[0]
            tmp_mean_block[y]  = tmp_mean_block[y] - 1/self.n
            yhat = 1/(n_p + 1) * (tmp_mean_block) + n_p/(n_p + 1) * sum_sigmoids_x/sum_sigmoids
            return y, yhat, n_p
        else:
            tmp_mean_block = self.mean_block[0]
            tmp_mean_block[y]  = tmp_mean_block[y] - 1/self.n
            return y, tmp_mean_block, n_p

# Continuous variables
class Cont:
    # Initalize weights and store variables in a useful way
    def __init__(self, condition, metadata, alpha = 0.1, beta = 0.1, w_0 = -1, w_1 = -2, convergence_radius=0.1, convergence_length=20000, done_updating = False):
        self.condition = condition
        self.w_0 = w_0
        self.w_1 = w_1
        #self.c = c
        self.alpha = alpha
        self.beta = beta
        in_list = np.array([float(item) for item in metadata[condition].tolist() if pd.notnull(item)])
        self.true_mean = np.mean(in_list)
        del(in_list)
        self.condition_dict = dict(zip([seq_id for seq_id in metadata['seq_ID'].tolist()], [float(item)/self.true_mean if pd.notnull(item) else None for item in metadata[condition].tolist()]))
        self.done_updating = done_updating
        self.n = (len(metadata.index) - metadata[condition].isna().sum()) - 1

        all_x = np.array([value for key, value in self.condition_dict.items() if value is not None])
        self.xbar = np.mean(all_x) # Should be 1
        self.eta = np.std(all_x)
        self.c = self.eta
        del(all_x)

        # Store previous weights for checking convergence
        self.w_0_history = np.linspace(self.w_0 + convergence_radius * convergence_length, self.w_0, num=convergence_length)
        self.w_1_history = np.linspace(self.w_1 + convergence_radius * convergence_length, self.w_1, num=convergence_length)
        #self.c_history = np.linspace(self.c + convergence_radius * convergence_length, self.c, num=convergence_length)
        self.convergence_radius = convergence_radius
        self.convergence_length = convergence_length

        self.condition = condition

    def set_alpha(self, alpha):
        self.alpha = float(alpha)

    # Update the weights with gradient descent using ID as the protein of interest, IDs as the similar proteins, ss as their sequence similarity, and n_p as the number of similar proteins
    def update(self, ID, IDs, ss, n_p):
        if not self.done_updating:
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
                norm_c = 1/(np.sqrt(2 * math.pi) * eta * (n_p + 1))

                # Calculate loss with integral approximation
                int_space = np.arange(y * 0.9, y * 1.102, y / 100)
                std_norm_piece = std_norm_exp((int_space - xbar)/eta)
                std_phis = np.multiply(1/hs, std_norm_exp(zs(int_space, xs, hs)))
                loss = 1 - np.sum(norm_c * std_norm_piece + 1/((n_p + 1) * np.sqrt(2 * math.pi)) * np.sum(std_phis, axis=1)) * y / 100

                # Calculate gradients
                norm_d = 1/((n_p + 1) * np.sqrt(2 * math.pi))
                m = np.exp(self.w_0) + np.exp(-self.w_1)

                dfydh = np.multiply(1 / np.square(hs), np.multiply(std_norm_exp(zs(int_space, xs, hs)), np.square(zs(int_space, xs, hs)) - 1)) # ndim(yhat) * ndim(xs)
                dhdw0 = np.add(self.c/U * np.multiply(sigmoids, 1-sigmoids), -np.multiply(self.c/np.multiply(self.w_1, np.square(U)) * sigmoids, np.exp(self.w_0) / m - (np.exp(self.w_0) / (1+np.exp(self.w_0))))) # ndim(xs)
                dhdw1 = np.add(np.multiply(ss, self.c/U * np.multiply(sigmoids, 1-sigmoids)), -np.multiply(self.c/np.square(np.multiply(self.w_1, U)) * sigmoids, -self.w_1 * np.exp(-self.w_1) / m - (np.log(m) - np.log(1 + np.exp(self.w_0))))) # ndim(xs)
                #dhdc = hs/self.c # ndim(xs)

                dldw0 = - np.sum(norm_d * np.multiply(dfydh, dhdw0)) * y / 100
                dldw1 = - np.sum(norm_d * np.multiply(dfydh, dhdw1)) * y / 100
                #dldc = - np.sum(norm_d * np.multiply(dfydh, dhdc)) * y / 100 + 2 * (self.c - eta) * self.beta

                # Ensure c doesn't jump over 0 become problematic
                #if np.abs(self.alpha * dldc) > np.abs(self.c):
                #    dldc = np.sign(dldc) * np.abs(self.c) / 2 / self.alpha

                self.w_0 = self.w_0 - self.alpha * dldw0
                self.w_1 = self.w_1 - self.alpha * dldw1
                #self.c = self.c - self.alpha * dldc

                # Check for convergence
                self.w_0_history = np.roll(self.w_0_history, -1)
                self.w_0_history[self.convergence_length - 1] = self.w_0
                self.w_1_history = np.roll(self.w_1_history, -1)
                self.w_1_history[self.convergence_length - 1] = self.w_1
                #self.c_history = np.roll(self.c_history, -1)
                #self.c_history[self.convergence_length - 1] = self.c

                if np.max(self.w_0_history) - np.min(self.w_0_history) < self.convergence_radius and np.max(self.w_1_history) - np.min(self.w_1_history) < self.convergence_radius:# and np.max(self.c_history) - np.min(self.c_history) < self.convergence_radius:
                    self.done_updating = True
                    print("Done updating weights for " + self.condition)

            else:
                loss = 1 - np.sum(1/(np.sqrt(2 * math.pi) * eta) * std_norm_exp(zs(np.arange(y * 0.9, y * 1.102, y / 100), xbar, eta)[0])) * y / 100

            if loss > 1.01 or loss < -1:
                raise ValueError("Loss out of bounds for " + self.condition + " at " + str(loss) + " with c=" + str(self.c))
            if loss is np.nan:
                raise ValueError("Invalid loss from continuous predictor")

            return loss
        else:
            # If done training
            return None

    def evaluate(self, ID, IDs, ss, n_p, CI):
        if self.condition_dict[ID] is None:
            y = np.nan
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
            norm_c = 1/(np.sqrt(2 * math.pi) * eta * (n_p + 1))

            def fyhat(yhat):
                std_norm_piece = std_norm_exp((yhat - xbar)/eta)
                std_phis = np.multiply(1/hs, std_norm_exp(zs(yhat, xs, hs)))
                return norm_c * std_norm_piece + 1/((n_p + 1) * np.sqrt(2 * math.pi)) * np.sum(std_phis, axis=1)

            # Get mean from linearity of expectation
            mean = 1/(n_p + 1) * xbar + n_p/(n_p + 1) * np.mean(xs)

            # Get mode
            for_search = np.append(xbar, xs)
            mode_start_search = min(for_search)
            mode_stop_search = max(for_search)
            search_dx = (mode_stop_search - mode_start_search)/1000
            search_space = np.arange(mode_start_search, mode_stop_search * (1 + search_dx), search_dx)
            searched = fyhat(search_space)
            mode = search_space[np.argmax(searched)]

            # Get NN% confidence interval
            # Need to include option for what percent CI to report
            lower = norm.ppf(0.5 - CI/2)
            search_val = max(min(all_x), min(lower * eta + xbar, min(lower*hs + xs)))
            while 1/(n_p + 1) * (norm.cdf((search_val - xbar)/eta) + np.sum(norm.cdf(np.divide(search_val - xs,hs)))) < 0.5 - CI/2:
                search_val += search_dx
            lb = search_val
            while 1/(n_p + 1) * (norm.cdf((search_val - xbar)/eta) + np.sum(norm.cdf(np.divide(search_val - xs,hs)))) < 0.5:
                search_val += search_dx
            median = search_val
            upper = norm.ppf(0.5 + CI/2)
            search_val = min(max(all_x), max(upper * eta + xbar, max(upper*hs + xs)))
            while 1/(n_p + 1) * (norm.cdf((search_val - xbar)/eta) + np.sum(norm.cdf(np.divide(search_val - xs,hs)))) >= 0.5 + CI/2:
                search_val -= search_dx
            ub = search_val + search_dx

        else:
            def fyhat(yhat):
                return 1/(np.sqrt(2 * math.pi) * eta) * std_norm_exp(zs(yhat, xbar, eta)[0])
            mean = xbar
            mode = xbar
            lb = norm.ppf(0.5 - CI/2) * eta + xbar
            ub = norm.ppf(0.5 + CI/2) * eta + xbar

        return y, mean, mode, median, lb, ub

# Continuous variables with binary presence
class Contbin:
    # Initalize weights and store variables in a useful way
    def __init__(self, condition, metadata, alpha = 0.1, beta_cont = 0.1, beta_bin = 0, w_0_cont = -1, w_1_cont = -2, w_0_bin = -1, w_1_bin = 3, convergence_radius=0.1, convergence_length=20000, done_updating = False):
        # Initialize continuous parameters
        self.condition = condition
        self.w_0_cont = w_0_cont
        self.w_1_cont = w_1_cont
        #self.c = c
        self.alpha = alpha
        self.beta_cont = beta_cont
        in_list = np.array([float(item) for item in metadata[condition].tolist() if pd.notnull(item) and item!=0])
        self.true_mean = np.mean(in_list)
        del(in_list)
        self.condition_dict = dict(zip([seq_id for seq_id in metadata['seq_ID'].tolist()], [(float(item)/self.true_mean, float(item)!=0) if pd.notnull(item) else None for item in metadata[condition].tolist()]))
        self.done_updating = done_updating
        self.n = (len(metadata.index) - metadata[condition].isna().sum()) - 1

        # Initialize binary parameters
        self.w_0_bin = w_0_bin
        self.w_1_bin = w_1_bin
        self.beta_bin = beta_bin

        # Overall mean term in each update
        self.mean_block = np.mean(np.array([value[1] for _, value in self.condition_dict.items() if value is not None]), axis=0) * (self.n + 1) / self.n
        all_x = np.array([value[0] for key, value in self.condition_dict.items() if value is not None and value[1] != 0])
        self.xbar = np.mean(all_x)
        self.eta = np.std(all_x)
        self.c = self.eta
        del(all_x)

        # Store previous weights for checking convergence
        self.w_0_cont_history = np.linspace(self.w_0_cont + convergence_radius * convergence_length, self.w_0_cont, num=convergence_length)
        self.w_1_cont_history = np.linspace(self.w_1_cont + convergence_radius * convergence_length, self.w_1_cont, num=convergence_length)
        #self.c_history = np.linspace(self.c + convergence_radius * convergence_length, self.c, num=convergence_length)
        self.w_0_bin_history = np.linspace(self.w_0_bin + convergence_radius * convergence_length, self.w_0_bin, num=convergence_length)
        self.w_1_bin_history = np.linspace(self.w_1_bin + convergence_radius * convergence_length, self.w_1_bin, num=convergence_length)
        self.convergence_radius = convergence_radius
        self.convergence_length = convergence_length

        self.condition = condition

    def set_alpha(self, alpha):
        self.alpha = float(alpha)

    # Update the weights with gradient descent using ID as the protein of interest, IDs as the similar proteins, ss as their sequence similarity, and n_p as the number of similar proteins
    def update(self, ID, IDs, ss_org, n_p):
        if not self.done_updating:
            if self.condition_dict[ID] is None:
                return None
            else:
                y = self.condition_dict[ID]

            xbar = self.xbar
            eta = self.eta

            xs = [self.condition_dict[item] for item in IDs] # Get values for similar proteins
            if sum(x is not None for x in xs) > 0:
                try:
                    xs_0, ss = zip(*[(x[0], s) for x, s in zip(xs, ss_org) if x is not None and x[1] != 0])
                    xs_0, ss = np.array(xs_0), np.array(ss)
                    n_p_0 = len(xs_0)
                    n_p = n_p_0
                except:
                    n_p_0 = 0
                    n_p = 1
            else:
                n_p = 0

            if n_p > 0:
                # Continuous updates if the condition is present
                if n_p_0 > 0 and y[1]:
                    # Pieces of yhats
                    sigmoids = sigmoid_array(self.w_1_cont * ss + self.w_0_cont)
                    U = (np.log(np.exp(-self.w_1_cont) + np.exp(self.w_0_cont)) - np.log(1 + np.exp(self.w_0_cont))) / self.w_1_cont + 1
                    hs = self.c * sigmoids / U
                    norm_c = 1/(np.sqrt(2 * math.pi) * eta * (n_p + 1))

                    # Calculate loss with integral approximation
                    int_space = np.arange(y[0] * 0.9, y[0] * 1.102, y[0] / 100)
                    std_norm_piece = std_norm_exp((int_space - xbar)/eta)
                    std_phis = np.multiply(1/hs, std_norm_exp(zs(int_space, xs_0, hs)))
                    loss_cont = 1 - np.sum(norm_c * std_norm_piece + 1/((n_p + 1) * np.sqrt(2 * math.pi)) * np.sum(std_phis, axis=1)) * y[0] * 0.2 / 101

                    # Calculate gradients
                    norm_d = 1/((n_p + 1) * np.sqrt(2 * math.pi))
                    m = np.exp(self.w_0_cont) + np.exp(-self.w_1_cont)

                    dfydh = np.multiply(1 / np.square(hs), np.multiply(std_norm_exp(zs(int_space, xs_0, hs)), np.square(zs(int_space, xs_0, hs)) - 1)) # ndim(yhat) * ndim(xs)
                    dhdw0 = np.add(self.c/U * np.multiply(sigmoids, 1-sigmoids), -np.multiply(self.c/np.multiply(self.w_1_cont, np.square(U)) * sigmoids, np.exp(self.w_0_cont) / m - (np.exp(self.w_0_cont) / (1+np.exp(self.w_0_cont))))) # ndim(xs)
                    dhdw1 = np.add(np.multiply(ss, self.c/U * np.multiply(sigmoids, 1-sigmoids)), -np.multiply(self.c/np.square(np.multiply(self.w_1_cont, U)) * sigmoids, -self.w_1_cont * np.exp(-self.w_1_cont) / m - (np.log(m) - np.log(1 + np.exp(self.w_0_cont))))) # ndim(xs)
                    #dhdc = hs/self.c # ndim(xs)

                    dldw0 = - np.sum(norm_d * np.multiply(dfydh, dhdw0)) * y[0] * 0.2 / 101
                    dldw1 = - np.sum(norm_d * np.multiply(dfydh, dhdw1)) * y[0] * 0.2 / 101
                    #dldc = - np.sum(norm_d * np.multiply(dfydh, dhdc)) * y[0] * 0.2 / 101 + 2 * (self.c - eta) * self.beta_cont

                    # Ensure c doesn't jump over 0 become problematic
                    #if np.abs(self.alpha * dldc) > np.abs(self.c):
                    #    dldc = np.sign(dldc) * np.abs(self.c) / 2 / self.alpha

                    self.w_0_cont = self.w_0_cont - self.alpha * dldw0
                    self.w_1_cont = self.w_1_cont - self.alpha * dldw1
                    #self.c = self.c - self.alpha * dldc

                    # Check for convergence
                    self.w_0_cont_history = np.roll(self.w_0_cont_history, -1)
                    self.w_0_cont_history[self.convergence_length - 1] = self.w_0_cont
                    self.w_1_cont_history = np.roll(self.w_1_cont_history, -1)
                    self.w_1_cont_history[self.convergence_length - 1] = self.w_1_cont
                    #self.c_history = np.roll(self.c_history, -1)
                    #self.c_history[self.convergence_length - 1] = self.c


                else:
                    loss_cont = 0

                xs_1, ss = zip(*[(x[1], s) for x, s in zip(xs, ss_org) if x is not None])
                xs_1, ss = np.array(xs_1), np.array(ss)
                n_p = len(xs_1)

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
                self.w_0_bin = self.w_0_bin - self.alpha * (dldw0 + 2 * self.w_0_bin * self.beta_bin)
                self.w_1_bin = self.w_1_bin - self.alpha * (dldw1 + 2 * self.w_1_bin * self.beta_bin)

                # Check for convergence
                self.w_0_bin_history = np.roll(self.w_0_bin_history, -1)
                self.w_0_bin_history[self.convergence_length - 1] = self.w_0_bin
                self.w_1_bin_history = np.roll(self.w_1_bin_history, -1)
                self.w_1_bin_history[self.convergence_length - 1] = self.w_1_bin

                if np.max(self.w_0_bin_history) - np.min(self.w_0_bin_history) < self.convergence_radius and np.max(self.w_1_bin_history) - np.min(self.w_1_bin_history) < self.convergence_radius and np.max(self.w_0_cont_history) - np.min(self.w_0_cont_history) < self.convergence_radius and np.max(self.w_1_cont_history) - np.min(self.w_1_cont_history) < self.convergence_radius:# and np.max(self.c_history) - np.min(self.c_history) < self.convergence_radius:
                    self.done_updating = True
                    print("Done updating weights for " + self.condition)

                # Compute and return loss
                loss_bin = -np.log(yhat)

            else:
                if y[1]:
                    loss_cont = 1 - np.sum(1/(np.sqrt(2 * math.pi) * eta) * std_norm_exp(zs(np.arange(y[0] * 0.9, y[0] * 1.102, y[0] / 100), xbar, eta)[0])) * y[0] * 0.2 / 101
                else:
                    loss_cont = 0
                loss_bin = -np.log(self.mean_block - y[1]/self.n)

            if loss_cont > 1.01 or loss_cont < -1:
                raise ValueError("Loss out of bounds for " + self.condition + " at " + str(loss_cont) + " with w_0_bin:" + str(self.w_0_bin) + ", w_1_bin: " + str(self.w_1_bin) + ", w_0_cont: " + str(self.w_0_cont) + ", w_1_cont: " + str(self.w_1_cont) + ", c: " + str(self.c))
            if loss_cont is np.nan:
                raise ValueError("Invalid loss from continuous predictor")
            if loss_bin is np.nan:
                raise ValueError("Invalid loss from categorical predictor")

            return (loss_cont, loss_bin)
        else:
            # If done training
            return None

def zs2d(yhat, xs_1, xs_2, hs_1, hs_2):
    return np.divide(np.expand_dims(yhat,2) - [xs_1,xs_2], [hs_1,hs_2])

def std_norm_exp2d(x):
   return np.exp(-np.sum(np.square(x), axis=1)/2)

# Continuous variables on two axes with binary presence
class Bicontbin:
    # Initalize weights and store variables in a useful way
    def __init__(self, condition, metadata, alpha = 0.1, beta_cont = 0.1, beta_bin = 0, w_10_cont = -1, w_11_cont = -2, w_20_cont = -1, w_21_cont = -2, w_0_bin = -1, w_1_bin = 3, convergence_radius=0.1, convergence_length=20000, done_updating = False):
        # Initialize continuous parameters
        self.condition = condition
        self.w_10_cont = w_10_cont
        self.w_11_cont = w_11_cont
        #self.c_1 = c_1
        self.w_20_cont = w_20_cont
        self.w_21_cont = w_21_cont
        #self.c_2 = c_2
        self.alpha = alpha
        self.beta_cont = beta_cont
        self.true_mean_1 = np.mean([float(item.split(" , ")[0]) for item in metadata[condition].tolist() if str(item) != "0" and not pd.isnull(item)])
        self.true_mean_2 = np.mean([float(item.split(" , ")[1]) for item in metadata[condition].tolist() if str(item) != "0" and not pd.isnull(item)])
        self.condition_dict = dict(zip([seq_id for seq_id in metadata['seq_ID'].tolist()],
            [None if pd.isnull(item) else (float(item.split(" , ")[0])/self.true_mean_1, float(item.split(" , ")[1])/self.true_mean_2, 1) if str(item) != "0" else (0, 0, 0) for item in metadata[condition].tolist()]))
        self.done_updating = done_updating
        self.n = (len(metadata.index) - metadata[condition].isna().sum()) - 1

        # Initialize binary parameters
        self.w_0_bin = w_0_bin
        self.w_1_bin = w_1_bin
        self.beta_bin = beta_bin

        # Overall mean term in each update
        self.mean_block = np.mean(np.array([value[2] for _, value in self.condition_dict.items() if value is not None]), axis=0) * (self.n + 1) / self.n

        all_x_1, all_x_2 = zip(*[(value[0], value[1]) for _, value in self.condition_dict.items() if value is not None])
        self.xbar_1 = np.mean(np.array(all_x_1))
        self.eta_1 = np.std(np.array(all_x_1))
        self.c_1 = self.eta_1
        del(all_x_1)

        self.xbar_2 = np.mean(np.array(all_x_2))
        self.eta_2 = np.std(np.array(all_x_2))
        self.c_2 = self.eta_2
        del(all_x_2)

        # Store previous weights for checking convergence
        self.w_10_cont_history = np.linspace(self.w_10_cont + convergence_radius * convergence_length, self.w_10_cont, num=convergence_length)
        self.w_11_cont_history = np.linspace(self.w_11_cont + convergence_radius * convergence_length, self.w_11_cont, num=convergence_length)
        #self.c_1_history = np.linspace(self.c_1 + convergence_radius * convergence_length, self.c_1, num=convergence_length)
        self.w_20_cont_history = np.linspace(self.w_20_cont + convergence_radius * convergence_length, self.w_20_cont, num=convergence_length)
        self.w_21_cont_history = np.linspace(self.w_21_cont + convergence_radius * convergence_length, self.w_21_cont, num=convergence_length)
        #self.c_2_history = np.linspace(self.c_2 + convergence_radius * convergence_length, self.c_2, num=convergence_length)
        self.w_0_bin_history = np.linspace(self.w_0_bin + convergence_radius * convergence_length, self.w_0_bin, num=convergence_length)
        self.w_1_bin_history = np.linspace(self.w_1_bin + convergence_radius * convergence_length, self.w_1_bin, num=convergence_length)
        self.convergence_radius = convergence_radius
        self.convergence_length = convergence_length

        self.condition = condition

    def set_alpha(self, alpha):
        self.alpha = float(alpha)

    # Update the weights with gradient descent using ID as the protein of interest, IDs as the similar proteins, ss as their sequence similarity, and n_p as the number of similar proteins
    def update(self, ID, IDs, ss_org, n_p):
        if not self.done_updating:
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
                try:
                    xs_1, xs_2, ss = zip(*[(x[0], x[1], s) for x, s in zip(xs, ss_org) if x is not None and x[2] != 0])
                    xs_1, xs_2, ss = np.array(xs_1), np.array(xs_2), np.array(ss)
                    n_p_0 = len(xs_1)
                    n_p = n_p_0
                except:
                    n_p_0 = 0
                    n_p = 1
            else:
                n_p = 0

            if n_p > 0:
                # Continuous 1 updates if present
                if n_p_0 > 0 and y[2]:
                    sigmoids_1 = sigmoid_array(self.w_11_cont * ss + self.w_10_cont)
                    U_1 = (np.log(np.exp(-self.w_11_cont) + np.exp(self.w_10_cont)) - np.log(1 + np.exp(self.w_10_cont))) / self.w_11_cont + 1
                    hs_1 = self.c_1 * sigmoids_1 / U_1

                    sigmoids_2 = sigmoid_array(self.w_21_cont * ss + self.w_20_cont)
                    U_2 = (np.log(np.exp(-self.w_21_cont) + np.exp(self.w_20_cont)) - np.log(1 + np.exp(self.w_20_cont))) / self.w_21_cont + 1
                    hs_2 = self.c_2 * sigmoids_2 / U_2

                    norm_c = 1/(2 * math.pi * eta_1 * eta_2 * (n_p + 1))

                    # Calculate loss with integral approximation
                    int_space = np.mgrid[(y[0] * 0.9):(y[0] * 1.102 + y[0]/200):(y[0] / 100), (y[1] * 0.9):(y[1] * 1.102 + y[1]/200):(y[1] / 100)].reshape(2,-1).T
                    std_norm_piece = std_norm_exp2d(zs2d(int_space, [xbar_1], [xbar_2], [eta_1], [eta_2]))
                    std_phis = np.divide(std_norm_exp2d(zs2d(int_space, xs_1, xs_2, hs_1, hs_2)), np.multiply(hs_1, hs_2))
                    loss_cont = 1 - np.sum(norm_c * np.sum(std_norm_piece, axis=1) + 1/((n_p + 1) * 2 * math.pi) * np.sum(std_phis, axis=1)) * y[0] * 0.2 / 101 * y[1] * 0.2 / 101

                    # Calculate gradients
                    norm_d = 1/((n_p + 1) * 2 * math.pi)
                    m_1 = np.exp(self.w_10_cont) + np.exp(-self.w_11_cont)

                    dfydh1 = 1 / hs_2 * 1 / np.square(hs_1) * np.multiply(std_norm_exp(zs(int_space, xs_2, hs_2)), np.multiply(std_norm_exp(zs(int_space, xs_1, hs_1)), np.square(zs(int_space, xs_1, hs_1)) - 1))
                    dhdw10 = np.add(self.c_1/U_1 * np.multiply(sigmoids_1, 1-sigmoids_1), -np.multiply(self.c_1/np.multiply(self.w_11_cont, np.square(U_1)) * sigmoids_1, np.exp(self.w_10_cont) / m_1 - (np.exp(self.w_10_cont) / (1+np.exp(self.w_10_cont)))))
                    dhdw11 = np.add(np.multiply(ss, self.c_1/U_1 * np.multiply(sigmoids_1, 1-sigmoids_1)), -np.multiply(self.c_1/np.square(np.multiply(self.w_11_cont, U_1)) * sigmoids_1, -self.w_11_cont * np.exp(-self.w_11_cont) / m_1 - (np.log(m_1) - np.log(1 + np.exp(self.w_10_cont)))))
                    #dhdc1 = hs_1/self.c_1 # ndim(xs)

                    dldw10 = - np.sum(norm_d * np.multiply(dfydh1, dhdw10)) * y[0] * 0.2 / 101 * y[1] * 0.2 / 101
                    dldw11 = - np.sum(norm_d * np.multiply(dfydh1, dhdw11)) * y[0] * 0.2 / 101 * y[1] * 0.2 / 101
                    #dldc1 = - np.sum(norm_d * np.multiply(dfydh1, dhdc1)) * y[0] * 0.2 / 101 * y[1] * 0.2 / 101 + 2 * (self.c_1 - eta_1) * self.beta_cont

                    # Ensure c doesn't jump over 0 become problematic
                    #if np.abs(self.alpha * dldc1) > np.abs(self.c_1):
                    #    dldc1 = np.sign(dldc1) * np.abs(self.c_1) / 2 / self.alpha

                    self.w_10_cont = self.w_10_cont - self.alpha * dldw10
                    self.w_11_cont = self.w_11_cont - self.alpha * dldw11
                    #self.c_1 = self.c_1 - self.alpha * dldc1

                    # Continuous 2 updates

                    # Calculate gradients
                    m_2 = np.exp(self.w_20_cont) + np.exp(-self.w_21_cont)

                    dfydh2 = 1 / hs_1 * 1 / np.square(hs_2) * np.multiply(std_norm_exp(zs(int_space, xs_1, hs_1)), np.multiply(std_norm_exp(zs(int_space, xs_2, hs_2)), np.square(zs(int_space, xs_2, hs_2)) - 1))
                    dhdw20 = np.add(self.c_2/U_2 * np.multiply(sigmoids_2, 1-sigmoids_2), -np.multiply(self.c_2/np.multiply(self.w_21_cont, np.square(U_2)) * sigmoids_2, np.exp(self.w_20_cont) / m_2 - (np.exp(self.w_20_cont) / (1+np.exp(self.w_20_cont)))))
                    dhdw21 = np.add(np.multiply(ss, self.c_2/U_2 * np.multiply(sigmoids_2, 1-sigmoids_2)), -np.multiply(self.c_2/np.square(np.multiply(self.w_21_cont, U_2)) * sigmoids_2, -self.w_21_cont * np.exp(-self.w_21_cont) / m_2 - (np.log(m_2) - np.log(1 + np.exp(self.w_20_cont)))))
                    #dhdc2 = hs_2/self.c_2 # ndim(xs)

                    dldw20 = - np.sum(norm_d * np.multiply(dfydh2, dhdw20)) * y[0] * 0.2 / 101 * y[1] * 0.2 / 101
                    dldw21 = - np.sum(norm_d * np.multiply(dfydh2, dhdw21)) * y[0] * 0.2 / 101 * y[1] * 0.2 / 101
                    #dldc2 = - np.sum(norm_d * np.multiply(dfydh2, dhdc2)) * y[0] * 0.2 / 101 * y[1] * 0.2 / 101 + 2 * (self.c_2 - eta_2) * self.beta_cont

                    # Ensure c doesn't jump over 0 become problematic
                    #if np.abs(self.alpha * dldc2) > np.abs(self.c_2):
                    #    dldc2 = np.sign(dldc2) * np.abs(self.c_2) / 2 / self.alpha

                    self.w_20_cont = self.w_20_cont - self.alpha * dldw20
                    self.w_21_cont = self.w_21_cont - self.alpha * dldw21
                    #self.c_2 = self.c_2 - self.alpha * dldc2

                    # Check for convergence
                    self.w_10_cont_history = np.roll(self.w_10_cont_history, -1)
                    self.w_10_cont_history[self.convergence_length - 1] = self.w_10_cont
                    self.w_11_cont_history = np.roll(self.w_11_cont_history, -1)
                    self.w_11_cont_history[self.convergence_length - 1] = self.w_11_cont
                    #self.c_1_history = np.roll(self.c_1_history, -1)
                    #self.c_1_history[self.convergence_length - 1] = self.c_1
                    self.w_20_cont_history = np.roll(self.w_20_cont_history, -1)
                    self.w_20_cont_history[self.convergence_length - 1] = self.w_20_cont
                    self.w_21_cont_history = np.roll(self.w_21_cont_history, -1)
                    self.w_21_cont_history[self.convergence_length - 1] = self.w_21_cont
                    #self.c_2_history = np.roll(self.c_2_history, -1)
                    #self.c_2_history[self.convergence_length - 1] = self.c_2

                else:
                    loss_cont = 0


                # Binary updates
                xs, ss = zip(*[(x[2], s) for x, s in zip(xs, ss_org) if x is not None])
                xs, ss = np.array(xs), np.array(ss)
                n_p = len(xs)

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
                self.w_0_bin = self.w_0_bin - self.alpha * (dldw0 + 2 * self.w_0_bin * self.beta_bin)
                self.w_1_bin = self.w_1_bin - self.alpha * (dldw1 + 2 * self.w_1_bin * self.beta_bin)

                # Check for convergence
                self.w_0_bin_history = np.roll(self.w_0_bin_history, -1)
                self.w_0_bin_history[self.convergence_length - 1] = self.w_0_bin
                self.w_1_bin_history = np.roll(self.w_1_bin_history, -1)
                self.w_1_bin_history[self.convergence_length - 1] = self.w_1_bin

                if np.max(self.w_0_bin_history) - np.min(self.w_0_bin_history) < self.convergence_radius and np.max(self.w_1_bin_history) - np.min(self.w_1_bin_history) < self.convergence_radius and np.max(self.w_10_cont_history) - np.min(self.w_10_cont_history) < self.convergence_radius and np.max(self.w_11_cont_history) - np.min(self.w_11_cont_history) < self.convergence_radius and np.max(self.w_20_cont_history) - np.min(self.w_20_cont_history) < self.convergence_radius and np.max(self.w_21_cont_history) - np.min(self.w_21_cont_history) < self.convergence_radius: #and np.max(self.c_2_history) - np.min(self.c_2_history) < self.convergence_radius and np.max(self.c_1_history) - np.min(self.c_1_history) < self.convergence_radius:
                    self.done_updating = True
                    print("Done updating weights for " + self.condition)

                # Compute and return loss
                loss_bin = -np.log(yhat)

            else:
                if y[2]:
                    norm_c = 1/(2 * math.pi * eta_1 * eta_2 * (n_p + 1))
                    int_space = np.mgrid[(y[0] * 0.9):(y[0] * 1.102 + y[0]/200):(y[0] / 100), (y[1] * 0.9):(y[1] * 1.102 + y[1]/200):(y[1] / 100)].reshape(2,-1).T
                    std_norm_piece = std_norm_exp2d(zs2d(int_space, [xbar_1], [xbar_2], [eta_1], [eta_2]))
                    loss_cont = 1 - np.sum(norm_c * np.sum(std_norm_piece, axis=1)) * y[0] * 0.2 / 101 * y[1] * 0.2 / 101
                else:
                    loss_cont = 0
                loss_bin = -np.log(self.mean_block - y[2]/self.n)

            if loss_cont is np.nan:
                raise ValueError("Invalid loss from continuous predictor")
            if loss_cont > 1.01 or loss_cont < -1:
                raise ValueError("Loss out of bounds for " + self.condition + " at " + str(loss_cont) + " with w_0_bin:" + str(self.w_0_bin) + ", w_1_bin: " + str(self.w_1_bin) + ", w_10: " + str(self.w_10_cont) + ", w_11: " + str(self.w_11_cont) + ", c_1: " + str(self.c_1) + ", w_20: " + str(self.w_20_cont) + ", w_21: " + str(self.w_21_cont) + ", c_2: " + str(self.c_2))
            if loss_bin is np.nan:
                raise ValueError("Invalid loss from categorical predictor")

            return (loss_cont, loss_bin)
        else:
            # If done updating
            return None
