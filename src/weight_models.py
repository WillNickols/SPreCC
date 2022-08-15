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

def get_encoding(encoding, condition, metadata, beta_cont, beta_bin, w_0_bin, w_1_bin, w_0_cont, w_1_cont, convergence_radius=0.1, convergence_length=20000, alpha=0.1):
    if encoding == "contbin":
        return Contbin(condition, metadata, beta_cont=beta_cont, beta_bin=beta_bin, convergence_radius=convergence_radius, convergence_length=convergence_length, alpha=alpha, w_0_cont=w_0_cont, w_1_cont=w_1_cont, w_0_bin=w_0_bin, w_1_bin=w_1_bin)
    elif encoding == "cont":
        return Cont(condition, metadata, beta=beta_cont, convergence_radius=convergence_radius, convergence_length=convergence_length, alpha=alpha, w_0=w_0_cont, w_1=w_1_cont)
    elif encoding == "bicontbin":
        return Bicontbin(condition, metadata, beta_cont=beta_cont, beta_bin=beta_bin, convergence_radius=convergence_radius, convergence_length=convergence_length, alpha=alpha, w_10_cont=w_0_cont, w_11_cont=w_1_cont, w_20_cont=w_0_cont, w_21_cont=w_1_cont, w_0_bin=w_0_bin, w_1_bin=w_1_bin)
    elif encoding == "cat":
        return Cat(condition, metadata, beta=beta_bin, convergence_radius=convergence_radius, convergence_length=convergence_length, alpha=alpha, w_0=w_0_bin, w_1=w_1_bin)
    else:
        raise ValueError("Encoding not allowed: " + encoding)

def get_test_encoding(encoding, condition, metadata_train, metadata_validation, w_0_bin, w_1_bin, w_0_cont, w_1_cont, old_weight):
    if old_weight is not None:
        if type(old_weight) is Cat:
            return Cat_test(old_weight.condition, metadata_train, metadata_validation, old_weight.w_0, old_weight.w_1)
        if type(old_weight) is Cont:
            return Cont_test(old_weight.condition, metadata_train, metadata_validation, old_weight.w_0, old_weight.w_1)
        if type(old_weight) is Contbin:
            return Contbin_test(old_weight.condition, metadata_train, metadata_validation, old_weight.w_0_cont, old_weight.w_1_cont, old_weight.w_0_bin, old_weight.w_1_bin)
        if type(old_weight) is Bicontbin:
            return Bicontbin_test(old_weight.condition, metadata_train, metadata_validation, old_weight.w_10_cont, old_weight.w_11_cont, old_weight.w_20_cont, old_weight.w_21_cont, old_weight.w_0_bin, old_weight.w_1_bin)
    else:
        if encoding == "cat":
            return Cat_test(condition, metadata_train, metadata_validation, w_0_bin, w_1_bin)
        if encoding == "cont":
            return Cont_test(condition, metadata_train, metadata_validation, w_0_cont, w_1_cont)
        if encoding == "contbin":
            return Contbin_test(condition, metadata_train, metadata_validation, w_0_cont, w_1_cont, w_0_bin, w_1_bin)
        if encoding == "bicontbin":
            return Bicontbin_test(condition, metadata_train, metadata_validation, w_0_cont, w_1_cont, w_0_cont, w_1_cont, w_0_bin, w_1_bin)
        else:
            raise ValueError("Encoding not allowed: " + encoding)

# Categorical variables
class Cat:
    # Initalize weights and store variables in a useful way
    def __init__(self, condition, metadata, alpha = 0.1, beta = 0, w_0 = -1, w_1 = 3, convergence_radius=0.1, convergence_length=20000, done_updating = False):
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
                return f'{(loss):.8f}'
            else:
                loss = -np.log(self.mean_block[0][y] - 1/self.n)
                return f'{(loss):.8f}'
        else:
            # If done training
            return None

    def evaluate(self, ID, IDs, ss, n_p, CI): # CI not used for anything but allows general calling of the evaluation function
        if self.condition_dict[ID] is None:
            return np.nan, np.nan, n_p
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
            return y, tuple([f'{(item):.8f}' for item in yhat]), n_p
        else:
            tmp_mean_block = self.mean_block[0]
            tmp_mean_block[y]  = tmp_mean_block[y] - 1/self.n
            return y, tuple([f'{(item):.8f}' for item in tmp_mean_block]), n_p

class Cat_test:
    # Initalize weights and store variables in a useful way
    def __init__(self, condition, metadata_train, metadata_test, w_0, w_1):
        self.w_0 = w_0
        self.w_1 = w_1

        vec = OneHotEncoder(handle_unknown='ignore')
        vec.fit([["microdialysis"],["vapor diffusion"],["batch"],["lipidic cubic phase"],["evaporation"],["counter diffusion"],["liquid diffusion"],["microfluidic"],["cooling"],["cell"],["other"],["soaking"],["dialysis"],["seeding"]])

        self.condition_dict_train = dict(zip([seq_id for seq_id in metadata_train['seq_ID'].tolist()], [vec.transform([[item]]).toarray() if pd.notnull(item) else None for item in metadata_train[condition].tolist()]))
        self.condition_dict_test = dict(zip([seq_id for seq_id in metadata_test['seq_ID'].tolist()], [vec.transform([[item]]).toarray() if pd.notnull(item) else None for item in metadata_test[condition].tolist()]))
        self.n = (len(metadata_train.index) - metadata_train[condition].isna().sum())

        # Overall mean term in each update
        self.mean_block = np.mean(np.array([value for _, value in self.condition_dict_train.items() if value is not None]), axis=0)

        self.condition = condition

    def evaluate(self, ID, IDs, ss, n_p, CI): # CI not used for anything but allows general calling of the evaluation function
        if self.condition_dict_test[ID] is None:
            return np.nan, np.nan, n_p
        else:
            y = np.argmax(self.condition_dict_test[ID]) # Only the index of the correct one matters here

        xs = [self.condition_dict_train[item][0] if self.condition_dict_train[item] is not None else None for item in IDs] # Get values for similar proteins
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
            yhat = 1/(n_p + 1) * (self.mean_block[0]) + n_p/(n_p + 1) * sum_sigmoids_x/sum_sigmoids
            return y, tuple([f'{(item):.8f}' for item in yhat]), n_p
        else:
            return y, tuple([f'{(item):.8f}' for item in self.mean_block[0]]), n_p

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
        self.condition_dict = dict(zip([seq_id for seq_id in metadata['seq_ID'].tolist()], [float(item) if pd.notnull(item) else None for item in metadata[condition].tolist()]))
        self.done_updating = done_updating
        self.n = (len(metadata.index) - metadata[condition].isna().sum()) - 1

        all_x = np.array([value for key, value in self.condition_dict.items() if value is not None])
        self.xbar = np.mean(all_x)
        self.eta = np.std(all_x)
        self.c = self.eta
        self.max_x = np.max(np.array(all_x))
        self.min_x = np.min(np.array(all_x))
        del(all_x)

        # Store previous weights for checking convergence
        self.w_0_history = np.linspace(self.w_0 + convergence_radius * convergence_length, self.w_0, num=convergence_length)
        self.w_1_history = np.linspace(self.w_1 + convergence_radius * convergence_length, self.w_1, num=convergence_length)
        #self.c_history = np.linspace(self.c + convergence_radius * convergence_length, self.c, num=convergence_length)
        self.convergence_radius = convergence_radius
        self.convergence_length = convergence_length

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
                try:
                    int_space = np.arange(y - 0.2 * self.eta, y + 0.2 * eta, eta * 0.4 / 100)
                except:
                    raise ValueError("Y: " + str(y) + ", ID: " + ID + ", n_p: " + str(n_p) + ", condition: " + self.condition)
                std_norm_piece = std_norm_exp((int_space - xbar)/eta)
                std_phis = np.multiply(1/hs, std_norm_exp(zs(int_space, xs, hs)))
                loss = 1 - np.sum(norm_c * std_norm_piece + 1/((n_p + 1) * np.sqrt(2 * math.pi)) * np.sum(std_phis, axis=1)) * eta * 0.4 / 100

                # Calculate gradients
                norm_d = 1/((n_p + 1) * np.sqrt(2 * math.pi))
                m = np.exp(self.w_0) + np.exp(-self.w_1)

                dfydh = np.multiply(1 / np.square(hs), np.multiply(std_norm_exp(zs(int_space, xs, hs)), np.square(zs(int_space, xs, hs)) - 1)) # ndim(yhat) * ndim(xs)
                dhdw0 = np.add(self.c/U * np.multiply(sigmoids, 1-sigmoids), -np.multiply(self.c/np.multiply(self.w_1, np.square(U)) * sigmoids, np.exp(self.w_0) / m - (np.exp(self.w_0) / (1+np.exp(self.w_0))))) # ndim(xs)
                dhdw1 = np.add(np.multiply(ss, self.c/U * np.multiply(sigmoids, 1-sigmoids)), -np.multiply(self.c/np.square(np.multiply(self.w_1, U)) * sigmoids, -self.w_1 * np.exp(-self.w_1) / m - (np.log(m) - np.log(1 + np.exp(self.w_0))))) # ndim(xs)
                #dhdc = hs/self.c # ndim(xs)

                dldw0 = - np.sum(norm_d * np.multiply(dfydh, dhdw0)) * eta * 0.4 / 100 + 2 * self.w_0 * self.beta
                dldw1 = - np.sum(norm_d * np.multiply(dfydh, dhdw1)) * eta * 0.4 / 100 + 2 * self.w_1 * self.beta
                #dldc = - np.sum(norm_d * np.multiply(dfydh, dhdc)) * self.eta * 0.4 / 100 + 2 * (self.c - eta) * self.beta

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
                loss = 1 - np.sum(1/(np.sqrt(2 * math.pi) * eta) * std_norm_exp(zs(np.arange(y - 0.2 * eta, y + 0.2 * eta, eta * 0.4 / 100), xbar, eta))) * eta * 0.4 / 100

            if loss > 1.01 or loss < -1:
                raise ValueError("Loss out of bounds for " + self.condition + " at " + str(loss) + " with c=" + str(self.c))

            return f'{(loss):.8f}'
        else:
            # If done training
            return None

    def evaluate(self, ID, IDs, ss, n_p, CI):
        if self.condition_dict[ID] is None:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
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
            lower = norm.ppf(0.5 - CI/2)
            upper = norm.ppf(0.5 + CI/2)
            search_lower = max(self.min_x, min(lower * eta + xbar, min(lower*hs + xs)), 0)
            search_upper = min(self.max_x, max(upper * eta + xbar, max(upper*hs + xs)))
            search_space = np.arange(search_lower, search_upper * (1 + search_dx), search_dx)
            eval_space = 1/(n_p + 1) * (norm.cdf((search_space - xbar)/eta) + np.sum(norm.cdf(zs(search_space, xs, hs)), axis=1))
            lb = search_space[np.argmax(eval_space >= 0.5 - CI/2)]
            median = search_space[np.argmax(eval_space >= 0.5)]
            ub_index = np.argmax(eval_space >= 0.5 + CI/2)
            if ub_index == 0: # Deal with case where nothing contains the top end of the probability distribution
                ub_index = len(eval_space) - 1
            ub = search_space[ub_index]

        else:
            mean = xbar
            mode = xbar
            median = xbar
            lb = max(norm.ppf(0.5 - CI/2) * eta + xbar, self.min_x)
            ub = min(norm.ppf(0.5 + CI/2) * eta + xbar, self.max_x)

        return f'{(y):.8f}', f'{(mean):.8f}', f'{(mode):.8f}', f'{(median):.8f}', f'{(lb):.8f}', f'{(ub):.8f}'

class Cont_test:
    # Initalize weights and store variables in a useful way
    def __init__(self, condition, metadata_train, metadata_test, w_0, w_1):
        self.condition = condition
        self.w_0 = w_0
        self.w_1 = w_1
        #self.c = c
        self.condition_dict_train = dict(zip([seq_id for seq_id in metadata_train['seq_ID'].tolist()], [float(item) if pd.notnull(item) else None for item in metadata_train[condition].tolist()]))
        self.condition_dict_test = dict(zip([seq_id for seq_id in metadata_test['seq_ID'].tolist()], [float(item) if pd.notnull(item) else None for item in metadata_test[condition].tolist()]))
        self.n = (len(metadata_train.index) - metadata_train[condition].isna().sum())

        all_x = np.array([value for key, value in self.condition_dict_train.items() if value is not None])
        self.xbar = np.mean(all_x)
        self.eta = np.std(all_x)
        self.c = self.eta
        self.max_x = np.max(np.array(all_x))
        self.min_x = np.min(np.array(all_x))
        del(all_x)

    def evaluate(self, ID, IDs, ss, n_p, CI):
        if self.condition_dict_test[ID] is None:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            y = self.condition_dict_test[ID]

        xs = [self.condition_dict_train[item] for item in IDs] # Get values for similar proteins
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
            lower = norm.ppf(0.5 - CI/2)
            upper = norm.ppf(0.5 + CI/2)
            search_lower = max(self.min_x, min(lower * eta + xbar, min(lower*hs + xs)), 0)
            search_upper = min(self.max_x, max(upper * eta + xbar, max(upper*hs + xs)))
            search_space = np.arange(search_lower, search_upper * (1 + search_dx), search_dx)
            eval_space = 1/(n_p + 1) * (norm.cdf((search_space - xbar)/eta) + np.sum(norm.cdf(zs(search_space, xs, hs)), axis=1))
            lb = search_space[np.argmax(eval_space >= 0.5 - CI/2)]
            median = search_space[np.argmax(eval_space >= 0.5)]
            ub_index = np.argmax(eval_space >= 0.5 + CI/2)
            if ub_index == 0: # Deal with case where nothing contains the top end of the probability distribution
                ub_index = len(eval_space) - 1
            ub = search_space[ub_index]

        else:
            mean = xbar
            mode = xbar
            median = xbar
            lb = max(norm.ppf(0.5 - CI/2) * eta + xbar, self.min_x)
            ub = min(norm.ppf(0.5 + CI/2) * eta + xbar, self.max_x)

        return f'{(y):.8f}', f'{(mean):.8f}', f'{(mode):.8f}', f'{(median):.8f}', f'{(lb):.8f}', f'{(ub):.8f}'

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
        self.max_x = np.max(np.array(all_x))
        self.min_x = np.min(np.array(all_x))
        del(all_x)

        # Store previous weights for checking convergence
        self.w_0_cont_history = np.linspace(self.w_0_cont + convergence_radius * convergence_length, self.w_0_cont, num=convergence_length)
        self.w_1_cont_history = np.linspace(self.w_1_cont + convergence_radius * convergence_length, self.w_1_cont, num=convergence_length)
        #self.c_history = np.linspace(self.c + convergence_radius * convergence_length, self.c, num=convergence_length)
        self.w_0_bin_history = np.linspace(self.w_0_bin + convergence_radius * convergence_length, self.w_0_bin, num=convergence_length)
        self.w_1_bin_history = np.linspace(self.w_1_bin + convergence_radius * convergence_length, self.w_1_bin, num=convergence_length)
        self.convergence_radius = convergence_radius
        self.convergence_length = convergence_length

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
                    norm_c = 1/(np.sqrt(2 * math.pi) * eta * (n_p_0 + 1))

                    # Calculate loss with integral approximation
                    int_space = np.arange(y[0] * 0.9, y[0] * 1.102, y[0] / 100)
                    std_norm_piece = std_norm_exp((int_space - xbar)/eta)
                    std_phis = np.multiply(1/hs, std_norm_exp(zs(int_space, xs_0, hs)))
                    loss_cont = 1 - np.sum(norm_c * std_norm_piece + 1/((n_p_0 + 1) * np.sqrt(2 * math.pi)) * np.sum(std_phis, axis=1)) * y[0] * 0.2 / 101

                    # Calculate gradients
                    norm_d = 1/((n_p_0 + 1) * np.sqrt(2 * math.pi))
                    m = np.exp(self.w_0_cont) + np.exp(-self.w_1_cont)

                    dfydh = np.multiply(1 / np.square(hs), np.multiply(std_norm_exp(zs(int_space, xs_0, hs)), np.square(zs(int_space, xs_0, hs)) - 1)) # ndim(yhat) * ndim(xs)
                    dhdw0 = np.add(self.c/U * np.multiply(sigmoids, 1-sigmoids), -np.multiply(self.c/np.multiply(self.w_1_cont, np.square(U)) * sigmoids, np.exp(self.w_0_cont) / m - (np.exp(self.w_0_cont) / (1+np.exp(self.w_0_cont))))) # ndim(xs)
                    dhdw1 = np.add(np.multiply(ss, self.c/U * np.multiply(sigmoids, 1-sigmoids)), -np.multiply(self.c/np.square(np.multiply(self.w_1_cont, U)) * sigmoids, -self.w_1_cont * np.exp(-self.w_1_cont) / m - (np.log(m) - np.log(1 + np.exp(self.w_0_cont))))) # ndim(xs)
                    #dhdc = hs/self.c # ndim(xs)

                    dldw0 = - np.sum(norm_d * np.multiply(dfydh, dhdw0)) * y[0] * 0.2 / 101 + 2 * self.w_0_cont * self.beta_cont
                    dldw1 = - np.sum(norm_d * np.multiply(dfydh, dhdw1)) * y[0] * 0.2 / 101 + 2 * self.w_1_cont * self.beta_cont
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
                    loss_cont = np.nan

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
                    loss_cont = 1 - np.sum(1/(np.sqrt(2 * math.pi) * eta) * std_norm_exp(zs(np.arange(y[0] * 0.9, y[0] * 1.102, y[0] / 100), xbar, eta))) * y[0] * 0.2 / 101
                else:
                    loss_cont = np.nan
                loss_bin = -np.log(self.mean_block - y[1]/self.n)

            if loss_cont > 1.01 or loss_cont < -1:
                raise ValueError("Loss out of bounds for " + self.condition + " at " + str(loss_cont) + " with w_0_bin:" + str(self.w_0_bin) + ", w_1_bin: " + str(self.w_1_bin) + ", w_0_cont: " + str(self.w_0_cont) + ", w_1_cont: " + str(self.w_1_cont) + ", c: " + str(self.c))

            return (f'{(loss_cont):.8f}', f'{(loss_bin):.8f}')
        else:
            # If done training
            return None

    def evaluate(self, ID, IDs, ss_org, n_p, CI):
        if self.condition_dict[ID] is None:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
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
            if y[1]:
                # Continuous updates if the condition is present
                if n_p_0 > 0:
                    # Pieces of yhats
                    sigmoids = sigmoid_array(self.w_1_cont * ss + self.w_0_cont)
                    U = (np.log(np.exp(-self.w_1_cont) + np.exp(self.w_0_cont)) - np.log(1 + np.exp(self.w_0_cont))) / self.w_1_cont + 1
                    hs = self.c * sigmoids / U
                    norm_c = 1/(np.sqrt(2 * math.pi) * eta * (n_p_0 + 1))

                    def fyhat(yhat):
                        std_norm_piece = std_norm_exp((yhat - xbar)/eta)
                        std_phis = np.multiply(1/hs, std_norm_exp(zs(yhat, xs_0, hs)))
                        return norm_c * std_norm_piece + 1/((n_p_0 + 1) * np.sqrt(2 * math.pi)) * np.sum(std_phis, axis=1)

                    # Get mean from linearity of expectation
                    mean = 1/(n_p_0 + 1) * xbar + n_p_0/(n_p_0 + 1) * np.mean(xs_0)

                    # Get mode
                    for_search = np.append(xbar, xs_0)
                    mode_start_search = min(for_search)
                    mode_stop_search = max(for_search)
                    search_dx = (mode_stop_search - mode_start_search)/1000
                    search_space = np.arange(mode_start_search, mode_stop_search * (1 + search_dx), search_dx)
                    searched = fyhat(search_space)
                    mode = search_space[np.argmax(searched)]

                    # Get NN% confidence interval
                    lower = norm.ppf(0.5 - CI/2)
                    upper = norm.ppf(0.5 + CI/2)
                    search_lower = max(self.min_x, min(lower * eta + xbar, min(lower*hs + xs_0)), 0)
                    search_upper = min(self.max_x, max(upper * eta + xbar, max(upper*hs + xs_0)))
                    search_space = np.arange(search_lower, search_upper * (1 + search_dx), search_dx)
                    eval_space = 1/(n_p_0 + 1) * (norm.cdf((search_space - xbar)/eta) + np.sum(norm.cdf(zs(search_space, xs_0, hs)), axis=1))
                    lb = search_space[np.argmax(eval_space >= 0.5 - CI/2)]
                    median = search_space[np.argmax(eval_space >= 0.5)]
                    ub_index = np.argmax(eval_space >= 0.5 + CI/2)
                    if ub_index == 0: # Deal with case where nothing contains the top end of the probability distribution
                        ub_index = len(eval_space) - 1
                    ub = search_space[ub_index]

                else:
                    mean = xbar
                    mode = xbar
                    median = xbar
                    lb = max(norm.ppf(0.5 - CI/2) * eta + xbar, self.min_x)
                    ub = min(norm.ppf(0.5 + CI/2) * eta + xbar, self.max_x)
            else:
                mean = np.nan
                mode = np.nan
                median = np.nan
                lb = np.nan
                ub = np.nan

            xs_1, ss = zip(*[(x[1], s) for x, s in zip(xs, ss_org) if x is not None])
            xs_1, ss = np.array(xs_1), np.array(ss)
            n_p = len(xs_1)

            # Binary updates
            xs_1 = np.array([x[1] for x in xs if x is not None])

            if n_p > 0:
                sigmoids = sigmoid_array(self.w_1_bin * ss + self.w_0_bin)
                sigmoids_x = np.multiply(xs_1, sigmoids)
                sum_sigmoids = np.sum(sigmoids)
                sum_sigmoids_x = np.sum(sigmoids_x)
                yhat = 1/(n_p + 1) * (self.mean_block - y[1]/self.n) + n_p/(n_p + 1) * sum_sigmoids_x/sum_sigmoids
            else:
                yhat = self.mean_block

        else:
            if y[1]:
                mean = xbar
                mode = xbar
                median = xbar
                lb = max(norm.ppf(0.5 - CI/2) * eta + xbar, self.min_x)
                ub = min(norm.ppf(0.5 + CI/2) * eta + xbar, self.max_x)
            else:
                mean = np.nan
                mode = np.nan
                median = np.nan
                lb = np.nan
                ub = np.nan

            yhat = self.mean_block

        return (f'{(y[0] * self.true_mean):.8f}', f'{(y[1]):.8f}'), f'{(mean * self.true_mean):.8f}', f'{(mode * self.true_mean):.8f}', f'{(median * self.true_mean):.8f}', f'{(lb * self.true_mean):.8f}', f'{(ub * self.true_mean):.8f}', f'{(yhat):.8f}'

class Contbin_test:
    # Initalize weights and store variables in a useful way
    def __init__(self, condition, metadata_train, metadata_test, w_0_cont, w_1_cont, w_0_bin, w_1_bin):
        # Initialize continuous parameters
        self.condition = condition
        self.w_0_cont = w_0_cont
        self.w_1_cont = w_1_cont
        #self.c = c
        in_list = np.array([float(item) for item in metadata_train[condition].tolist() if pd.notnull(item) and item!=0])
        self.true_mean = np.mean(in_list)
        del(in_list)
        self.condition_dict_train = dict(zip([seq_id for seq_id in metadata_train['seq_ID'].tolist()], [(float(item)/self.true_mean, float(item)!=0) if pd.notnull(item) else None for item in metadata_train[condition].tolist()]))
        self.condition_dict_test = dict(zip([seq_id for seq_id in metadata_test['seq_ID'].tolist()], [(float(item)/self.true_mean, float(item)!=0) if pd.notnull(item) else None for item in metadata_test[condition].tolist()]))
        self.n = (len(metadata_train.index) - metadata_train[condition].isna().sum())

        # Initialize binary parameters
        self.w_0_bin = w_0_bin
        self.w_1_bin = w_1_bin

        # Overall mean term in each update
        self.mean_block = np.mean(np.array([value[1] for _, value in self.condition_dict_train.items() if value is not None]), axis=0)
        all_x = np.array([value[0] for key, value in self.condition_dict_train.items() if value is not None and value[1] != 0])
        self.xbar = np.mean(all_x)
        self.eta = np.std(all_x)
        self.c = self.eta
        self.max_x = np.max(np.array(all_x))
        self.min_x = np.min(np.array(all_x))
        del(all_x)

    def evaluate(self, ID, IDs, ss_org, n_p, CI):
        if self.condition_dict_test[ID] is None:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            y = self.condition_dict_test[ID]

        xbar = self.xbar
        eta = self.eta

        xs = [self.condition_dict_train[item] for item in IDs] # Get values for similar proteins
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
            if y[1]:
                # Continuous updates if the condition is present
                if n_p_0 > 0:
                    # Pieces of yhats
                    sigmoids = sigmoid_array(self.w_1_cont * ss + self.w_0_cont)
                    U = (np.log(np.exp(-self.w_1_cont) + np.exp(self.w_0_cont)) - np.log(1 + np.exp(self.w_0_cont))) / self.w_1_cont + 1
                    hs = self.c * sigmoids / U
                    norm_c = 1/(np.sqrt(2 * math.pi) * eta * (n_p_0 + 1))

                    def fyhat(yhat):
                        std_norm_piece = std_norm_exp((yhat - xbar)/eta)
                        std_phis = np.multiply(1/hs, std_norm_exp(zs(yhat, xs_0, hs)))
                        return norm_c * std_norm_piece + 1/((n_p_0 + 1) * np.sqrt(2 * math.pi)) * np.sum(std_phis, axis=1)

                    # Get mean from linearity of expectation
                    mean = 1/(n_p_0 + 1) * xbar + n_p_0/(n_p_0 + 1) * np.mean(xs_0)

                    # Get mode
                    for_search = np.append(xbar, xs_0)
                    mode_start_search = min(for_search)
                    mode_stop_search = max(for_search)
                    search_dx = (mode_stop_search - mode_start_search)/1000
                    search_space = np.arange(mode_start_search, mode_stop_search * (1 + search_dx), search_dx)
                    searched = fyhat(search_space)
                    mode = search_space[np.argmax(searched)]

                    # Get NN% confidence interval
                    lower = norm.ppf(0.5 - CI/2)
                    upper = norm.ppf(0.5 + CI/2)
                    search_lower = max(self.min_x, min(lower * eta + xbar, min(lower*hs + xs_0)), 0)
                    search_upper = min(self.max_x, max(upper * eta + xbar, max(upper*hs + xs_0)))
                    search_space = np.arange(search_lower, search_upper * (1 + search_dx), search_dx)
                    eval_space = 1/(n_p_0 + 1) * (norm.cdf((search_space - xbar)/eta) + np.sum(norm.cdf(zs(search_space, xs_0, hs)), axis=1))
                    lb = search_space[np.argmax(eval_space >= 0.5 - CI/2)]
                    median = search_space[np.argmax(eval_space >= 0.5)]
                    ub_index = np.argmax(eval_space >= 0.5 + CI/2)
                    if ub_index == 0: # Deal with case where nothing contains the top end of the probability distribution
                        ub_index = len(eval_space) - 1
                    ub = search_space[ub_index]

                else:
                    mean = xbar
                    mode = xbar
                    median = xbar
                    lb = max(norm.ppf(0.5 - CI/2) * eta + xbar, self.min_x)
                    ub = min(norm.ppf(0.5 + CI/2) * eta + xbar, self.max_x)
            else:
                mean = np.nan
                mode = np.nan
                median = np.nan
                lb = np.nan
                ub = np.nan

            xs_1, ss = zip(*[(x[1], s) for x, s in zip(xs, ss_org) if x is not None])
            xs_1, ss = np.array(xs_1), np.array(ss)
            n_p = len(xs_1)

            # Binary updates
            xs_1 = np.array([x[1] for x in xs if x is not None])

            if n_p > 0:
                sigmoids = sigmoid_array(self.w_1_bin * ss + self.w_0_bin)
                sigmoids_x = np.multiply(xs_1, sigmoids)
                sum_sigmoids = np.sum(sigmoids)
                sum_sigmoids_x = np.sum(sigmoids_x)
                yhat = 1/(n_p + 1) * (self.mean_block) + n_p/(n_p + 1) * sum_sigmoids_x/sum_sigmoids
            else:
                yhat = self.mean_block

        else:
            if y[1]:
                mean = xbar
                mode = xbar
                median = xbar
                lb = max(norm.ppf(0.5 - CI/2) * eta + xbar, self.min_x)
                ub = min(norm.ppf(0.5 + CI/2) * eta + xbar, self.max_x)
            else:
                mean = np.nan
                mode = np.nan
                median = np.nan
                lb = np.nan
                ub = np.nan

            yhat = self.mean_block

        return (f'{(y[0] * self.true_mean):.8f}', f'{(y[1]):.8f}'), f'{(mean * self.true_mean):.8f}', f'{(mode * self.true_mean):.8f}', f'{(median * self.true_mean):.8f}', f'{(lb * self.true_mean):.8f}', f'{(ub * self.true_mean):.8f}', f'{(yhat):.8f}'

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
        self.min_x_1 = np.min(np.array(all_x_1))
        self.max_x_1 = np.max(np.array(all_x_1))
        del(all_x_1)

        self.xbar_2 = np.mean(np.array(all_x_2))
        self.eta_2 = np.std(np.array(all_x_2))
        self.c_2 = self.eta_2
        self.min_x_2 = np.min(np.array(all_x_2))
        self.max_x_2 = np.max(np.array(all_x_2))
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

                    norm_c = 1/(2 * math.pi * eta_1 * eta_2 * (n_p_0 + 1))

                    # Calculate loss with integral approximation
                    int_space = np.mgrid[(y[0] * 0.9):(y[0] * 1.102 + y[0]/200):(y[0] / 100), (y[1] * 0.9):(y[1] * 1.102 + y[1]/200):(y[1] / 100)].reshape(2,-1).T
                    std_norm_piece = std_norm_exp2d(zs2d(int_space, [xbar_1], [xbar_2], [eta_1], [eta_2]))
                    std_phis = np.divide(std_norm_exp2d(zs2d(int_space, xs_1, xs_2, hs_1, hs_2)), np.multiply(hs_1, hs_2))
                    loss_cont = 1 - np.sum(norm_c * np.sum(std_norm_piece, axis=1) + 1/((n_p_0 + 1) * 2 * math.pi) * np.sum(std_phis, axis=1)) * y[0] * 0.2 / 101 * y[1] * 0.2 / 101

                    # Calculate gradients
                    norm_d = 1/((n_p_0 + 1) * 2 * math.pi)
                    m_1 = np.exp(self.w_10_cont) + np.exp(-self.w_11_cont)

                    dfydh1 = 1 / hs_2 * 1 / np.square(hs_1) * np.multiply(std_norm_exp(zs(int_space, xs_2, hs_2)), np.multiply(std_norm_exp(zs(int_space, xs_1, hs_1)), np.square(zs(int_space, xs_1, hs_1)) - 1))
                    dhdw10 = np.add(self.c_1/U_1 * np.multiply(sigmoids_1, 1-sigmoids_1), -np.multiply(self.c_1/np.multiply(self.w_11_cont, np.square(U_1)) * sigmoids_1, np.exp(self.w_10_cont) / m_1 - (np.exp(self.w_10_cont) / (1+np.exp(self.w_10_cont)))))
                    dhdw11 = np.add(np.multiply(ss, self.c_1/U_1 * np.multiply(sigmoids_1, 1-sigmoids_1)), -np.multiply(self.c_1/np.square(np.multiply(self.w_11_cont, U_1)) * sigmoids_1, -self.w_11_cont * np.exp(-self.w_11_cont) / m_1 - (np.log(m_1) - np.log(1 + np.exp(self.w_10_cont)))))
                    #dhdc1 = hs_1/self.c_1 # ndim(xs)

                    dldw10 = - np.sum(norm_d * np.multiply(dfydh1, dhdw10)) * y[0] * 0.2 / 101 * y[1] * 0.2 / 101 + 2 * self.w_10_cont * self.beta_cont
                    dldw11 = - np.sum(norm_d * np.multiply(dfydh1, dhdw11)) * y[0] * 0.2 / 101 * y[1] * 0.2 / 101 + 2 * self.w_11_cont * self.beta_cont
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

                    dldw20 = - np.sum(norm_d * np.multiply(dfydh2, dhdw20)) * y[0] * 0.2 / 101 * y[1] * 0.2 / 101 + 2 * self.w_20_cont * self.beta_cont
                    dldw21 = - np.sum(norm_d * np.multiply(dfydh2, dhdw21)) * y[0] * 0.2 / 101 * y[1] * 0.2 / 101 + 2 * self.w_21_cont * self.beta_cont
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
                    loss_cont = np.nan


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
                    loss_cont = np.nan
                loss_bin = -np.log(self.mean_block - y[2]/self.n)

            if loss_cont > 1.01 or loss_cont < -1:
                raise ValueError("Loss out of bounds for " + self.condition + " at " + str(loss_cont) + " with w_0_bin:" + str(self.w_0_bin) + ", w_1_bin: " + str(self.w_1_bin) + ", w_10: " + str(self.w_10_cont) + ", w_11: " + str(self.w_11_cont) + ", c_1: " + str(self.c_1) + ", w_20: " + str(self.w_20_cont) + ", w_21: " + str(self.w_21_cont) + ", c_2: " + str(self.c_2))

            return (f'{loss_cont:.8f}', f'{loss_bin:.8f}')
        else:
            # If done updating
            return None

    def evaluate(self, ID, IDs, ss_org, n_p, CI):
        if self.condition_dict[ID] is None:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
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
            if y[2]:
                if n_p_0 > 0:
                    sigmoids_1 = sigmoid_array(self.w_11_cont * ss + self.w_10_cont)
                    U_1 = (np.log(np.exp(-self.w_11_cont) + np.exp(self.w_10_cont)) - np.log(1 + np.exp(self.w_10_cont))) / self.w_11_cont + 1
                    hs_1 = self.c_1 * sigmoids_1 / U_1

                    sigmoids_2 = sigmoid_array(self.w_21_cont * ss + self.w_20_cont)
                    U_2 = (np.log(np.exp(-self.w_21_cont) + np.exp(self.w_20_cont)) - np.log(1 + np.exp(self.w_20_cont))) / self.w_21_cont + 1
                    hs_2 = self.c_2 * sigmoids_2 / U_2

                    norm_c = 1/(2 * math.pi * eta_1 * eta_2 * (n_p_0 + 1))

                    def fyhat(yhat):
                        std_norm_piece = std_norm_exp2d(zs2d(yhat, [xbar_1], [xbar_2], [eta_1], [eta_2]))
                        std_phis = np.divide(std_norm_exp2d(zs2d(yhat, xs_1, xs_2, hs_1, hs_2)), np.multiply(hs_1, hs_2))
                        return norm_c * np.sum(std_norm_piece, axis=1) + 1/((n_p_0 + 1) * 2 * math.pi) * np.sum(std_phis, axis=1)

                    # Get mean from linearity of expectation
                    mean_1 = 1/(n_p_0 + 1) * xbar_1 + n_p_0/(n_p_0 + 1) * np.mean(xs_1)
                    mean_2 = 1/(n_p_0 + 1) * xbar_2 + n_p_0/(n_p_0 + 1) * np.mean(xs_2)
                    mean = np.array([mean_1, mean_2])

                    # Get modes
                    for_search_1 = np.append(xbar_1, xs_1)
                    for_search_2 = np.append(xbar_2, xs_2)
                    mode_start_search = np.array([np.min(for_search_1), np.min(for_search_2)])
                    mode_stop_search = np.array([np.max(for_search_1), np.max(for_search_2)])
                    search_dx = (mode_stop_search - mode_start_search)/1000
                    search_dx_1 = search_dx[0]
                    search_dx_2 = search_dx[1]
                    search_space = np.mgrid[mode_start_search[0]:mode_stop_search[0] * (1 + search_dx_1 * 10):search_dx_1 * 10, mode_start_search[1]:mode_stop_search[1] * (1 + search_dx_2 * 10):search_dx_2 * 10].reshape(2,-1).T
                    searched = fyhat(search_space)
                    mode = search_space[np.argmax(searched)]

                    # Get NN% confidence interval and median
                    lower = norm.ppf(0.5 - CI/2)
                    upper = norm.ppf(0.5 + CI/2)
                    search_lower = max(self.min_x_1, min(lower * eta_1 + xbar_1, min(lower*hs_1 + xs_1)), 0)
                    search_upper = min(self.max_x_1, max(upper * eta_1 + xbar_1, max(upper*hs_1 + xs_1)))
                    search_space = np.arange(search_lower, search_upper * (1 + search_dx_1), search_dx_1)
                    eval_space = 1/(n_p_0 + 1) * (norm.cdf((search_space - xbar_1)/eta_1) + np.sum(norm.cdf(zs(search_space, xs_1, hs_1)), axis=1))
                    lb_1 = search_space[np.argmax(eval_space >= 0.5 - CI/2)]
                    median_1 = search_space[np.argmax(eval_space >= 0.5)]
                    ub_index = np.argmax(eval_space >= 0.5 + CI/2)
                    if ub_index == 0: # Deal with case where nothing contains the top end of the probability distribution
                        ub_index = len(eval_space) - 1
                    ub_1 = search_space[ub_index]

                    search_lower = max(self.min_x_2, min(lower * eta_2 + xbar_2, min(lower*hs_2 + xs_2)), 0)
                    search_upper = min(self.max_x_2, max(upper * eta_2 + xbar_2, max(upper*hs_2 + xs_2)))
                    search_space = np.arange(search_lower, search_upper * (1 + search_dx_2), search_dx_2)
                    eval_space = 1/(n_p_0 + 1) * (norm.cdf((search_space - xbar_2)/eta_2) + np.sum(norm.cdf(zs(search_space, xs_2, hs_2)), axis=1))
                    lb_2 = search_space[np.argmax(eval_space >= 0.5 - CI/2)]
                    median_2 = search_space[np.argmax(eval_space >= 0.5)]
                    ub_index = np.argmax(eval_space >= 0.5 + CI/2)
                    if ub_index == 0: # Deal with case where nothing contains the top end of the probability distribution
                        ub_index = len(eval_space) - 1
                    ub_2 = search_space[ub_index]

                    median = np.array([median_1, median_2])
                    lb = np.array([lb_1, lb_2])
                    ub = np.array([ub_1, ub_2])

                else:
                    mean = np.array([xbar_1, xbar_2])
                    mode = np.array([xbar_1, xbar_2])
                    median = np.array([xbar_1, xbar_2])
                    lb = np.array([max(norm.ppf(0.5 - CI/2) * eta_1 + xbar_1, self.min_x_1), max(norm.ppf(0.5 - CI/2) * eta_2 + xbar_2, self.min_x_2)])
                    ub = np.array([min(norm.ppf(0.5 + CI/2) * eta_1 + xbar_1, self.max_x_1), min(norm.ppf(0.5 + CI/2) * eta_2 + xbar_2, self.max_x_2)])
            else:
                mean = np.nan
                mode = np.nan
                median = np.nan
                lb = np.nan
                ub = np.nan

            # Binary updates
            xs, ss = zip(*[(x[2], s) for x, s in zip(xs, ss_org) if x is not None])
            xs, ss = np.array(xs), np.array(ss)
            n_p = len(xs)

            sigmoids = sigmoid_array(self.w_1_bin * ss + self.w_0_bin)
            sigmoids_x = np.multiply(xs, sigmoids)
            sum_sigmoids = np.sum(sigmoids)
            sum_sigmoids_x = np.sum(sigmoids_x)
            yhat = 1/(n_p + 1) * (self.mean_block - y[1]/self.n) + n_p/(n_p + 1) * sum_sigmoids_x/sum_sigmoids

        else:
            if y[2]:
                mean = np.array([xbar_1, xbar_2])
                mode = np.array([xbar_1, xbar_2])
                median = np.array([xbar_1, xbar_2])
                lb = np.array([max(norm.ppf(0.5 - CI/2) * eta_1 + xbar_1, self.min_x_1), max(norm.ppf(0.5 - CI/2) * eta_2 + xbar_2, self.min_x_2)])
                ub = np.array([min(norm.ppf(0.5 + CI/2) * eta_1 + xbar_1, self.max_x_1), min(norm.ppf(0.5 + CI/2) * eta_2 + xbar_2, self.max_x_2)])
            else:
                mean = np.nan
                mode = np.nan
                median = np.nan
                lb = np.nan
                ub = np.nan
            yhat = self.mean_block

        return (round(y[0] * self.true_mean_1, 2), round(y[1] * self.true_mean_2, 2), f'{(y[2]):.8f}'), tuple((mean * np.array([self.true_mean_1, self.true_mean_2])).round(2)), tuple((mode * np.array([self.true_mean_1, self.true_mean_2])).round(2)), tuple((median * np.array([self.true_mean_1, self.true_mean_2])).round(2)), tuple((lb * np.array([self.true_mean_1, self.true_mean_2])).round(2)), tuple((ub * np.array([self.true_mean_1, self.true_mean_2])).round(2)), f'{(yhat):.8f}'

class Bicontbin_test:
    # Initalize weights and store variables in a useful way
    def __init__(self, condition, metadata_train, metadata_test, w_10_cont, w_11_cont, w_20_cont, w_21_cont, w_0_bin, w_1_bin):
        # Initialize continuous parameters
        self.condition = condition
        self.w_10_cont = w_10_cont
        self.w_11_cont = w_11_cont
        #self.c_1 = c_1
        self.w_20_cont = w_20_cont
        self.w_21_cont = w_21_cont
        #self.c_2 = c_2
        self.true_mean_1 = np.mean([float(item.split(" , ")[0]) for item in metadata_train[condition].tolist() if str(item) != "0" and not pd.isnull(item)])
        self.true_mean_2 = np.mean([float(item.split(" , ")[1]) for item in metadata_train[condition].tolist() if str(item) != "0" and not pd.isnull(item)])
        self.condition_dict_train = dict(zip([seq_id for seq_id in metadata_train['seq_ID'].tolist()],
            [None if pd.isnull(item) else (float(item.split(" , ")[0])/self.true_mean_1, float(item.split(" , ")[1])/self.true_mean_2, 1) if str(item) != "0" else (0, 0, 0) for item in metadata_train[condition].tolist()]))
        self.condition_dict_test = dict(zip([seq_id for seq_id in metadata_test['seq_ID'].tolist()],
            [None if pd.isnull(item) else (float(item.split(" , ")[0])/self.true_mean_1, float(item.split(" , ")[1])/self.true_mean_2, 1) if str(item) != "0" else (0, 0, 0) for item in metadata_test[condition].tolist()]))
        self.n = (len(metadata_train.index) - metadata_train[condition].isna().sum())

        # Initialize binary parameters
        self.w_0_bin = w_0_bin
        self.w_1_bin = w_1_bin

        # Overall mean term in each update
        self.mean_block = np.mean(np.array([value[2] for _, value in self.condition_dict_train.items() if value is not None]), axis=0)

        all_x_1, all_x_2 = zip(*[(value[0], value[1]) for _, value in self.condition_dict_train.items() if value is not None])
        self.xbar_1 = np.mean(np.array(all_x_1))
        self.eta_1 = np.std(np.array(all_x_1))
        self.c_1 = self.eta_1
        self.min_x_1 = np.min(np.array(all_x_1))
        self.max_x_1 = np.max(np.array(all_x_1))
        del(all_x_1)

        self.xbar_2 = np.mean(np.array(all_x_2))
        self.eta_2 = np.std(np.array(all_x_2))
        self.c_2 = self.eta_2
        self.min_x_2 = np.min(np.array(all_x_2))
        self.max_x_2 = np.max(np.array(all_x_2))
        del(all_x_2)

    def evaluate(self, ID, IDs, ss_org, n_p, CI):
        if self.condition_dict_test[ID] is None:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            y = self.condition_dict_test[ID]

        xbar_1 = self.xbar_1
        xbar_2 = self.xbar_2
        eta_1 = self.eta_1
        eta_2 = self.eta_2

        xs = [self.condition_dict_train[item] for item in IDs] # Get values for similar proteins
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
            if y[2]:
                if n_p_0 > 0:
                    sigmoids_1 = sigmoid_array(self.w_11_cont * ss + self.w_10_cont)
                    U_1 = (np.log(np.exp(-self.w_11_cont) + np.exp(self.w_10_cont)) - np.log(1 + np.exp(self.w_10_cont))) / self.w_11_cont + 1
                    hs_1 = self.c_1 * sigmoids_1 / U_1

                    sigmoids_2 = sigmoid_array(self.w_21_cont * ss + self.w_20_cont)
                    U_2 = (np.log(np.exp(-self.w_21_cont) + np.exp(self.w_20_cont)) - np.log(1 + np.exp(self.w_20_cont))) / self.w_21_cont + 1
                    hs_2 = self.c_2 * sigmoids_2 / U_2

                    norm_c = 1/(2 * math.pi * eta_1 * eta_2 * (n_p_0 + 1))

                    def fyhat(yhat):
                        std_norm_piece = std_norm_exp2d(zs2d(yhat, [xbar_1], [xbar_2], [eta_1], [eta_2]))
                        std_phis = np.divide(std_norm_exp2d(zs2d(yhat, xs_1, xs_2, hs_1, hs_2)), np.multiply(hs_1, hs_2))
                        return norm_c * np.sum(std_norm_piece, axis=1) + 1/((n_p_0 + 1) * 2 * math.pi) * np.sum(std_phis, axis=1)

                    # Get mean from linearity of expectation
                    mean_1 = 1/(n_p_0 + 1) * xbar_1 + n_p_0/(n_p_0 + 1) * np.mean(xs_1)
                    mean_2 = 1/(n_p_0 + 1) * xbar_2 + n_p_0/(n_p_0 + 1) * np.mean(xs_2)
                    mean = np.array([mean_1, mean_2])

                    # Get modes
                    for_search_1 = np.append(xbar_1, xs_1)
                    for_search_2 = np.append(xbar_2, xs_2)
                    mode_start_search = np.array([np.min(for_search_1), np.min(for_search_2)])
                    mode_stop_search = np.array([np.max(for_search_1), np.max(for_search_2)])
                    search_dx = (mode_stop_search - mode_start_search)/1000
                    search_dx_1 = search_dx[0]
                    search_dx_2 = search_dx[1]
                    search_space = np.mgrid[mode_start_search[0]:mode_stop_search[0] * (1 + search_dx_1 * 10):search_dx_1 * 10, mode_start_search[1]:mode_stop_search[1] * (1 + search_dx_2 * 10):search_dx_2 * 10].reshape(2,-1).T
                    searched = fyhat(search_space)
                    mode = search_space[np.argmax(searched)]

                    # Get NN% confidence interval and median
                    lower = norm.ppf(0.5 - CI/2)
                    upper = norm.ppf(0.5 + CI/2)
                    search_lower = max(self.min_x_1, min(lower * eta_1 + xbar_1, min(lower*hs_1 + xs_1)), 0)
                    search_upper = min(self.max_x_1, max(upper * eta_1 + xbar_1, max(upper*hs_1 + xs_1)))
                    search_space = np.arange(search_lower, search_upper * (1 + search_dx_1), search_dx_1)
                    eval_space = 1/(n_p_0 + 1) * (norm.cdf((search_space - xbar_1)/eta_1) + np.sum(norm.cdf(zs(search_space, xs_1, hs_1)), axis=1))
                    lb_1 = search_space[np.argmax(eval_space >= 0.5 - CI/2)]
                    median_1 = search_space[np.argmax(eval_space >= 0.5)]
                    ub_index = np.argmax(eval_space >= 0.5 + CI/2)
                    if ub_index == 0: # Deal with case where nothing contains the top end of the probability distribution
                        ub_index = len(eval_space) - 1
                    ub_1 = search_space[ub_index]

                    search_lower = max(self.min_x_2, min(lower * eta_2 + xbar_2, min(lower*hs_2 + xs_2)), 0)
                    search_upper = min(self.max_x_2, max(upper * eta_2 + xbar_2, max(upper*hs_2 + xs_2)))
                    search_space = np.arange(search_lower, search_upper * (1 + search_dx_2), search_dx_2)
                    eval_space = 1/(n_p_0 + 1) * (norm.cdf((search_space - xbar_2)/eta_2) + np.sum(norm.cdf(zs(search_space, xs_2, hs_2)), axis=1))
                    lb_2 = search_space[np.argmax(eval_space >= 0.5 - CI/2)]
                    median_2 = search_space[np.argmax(eval_space >= 0.5)]
                    ub_index = np.argmax(eval_space >= 0.5 + CI/2)
                    if ub_index == 0: # Deal with case where nothing contains the top end of the probability distribution
                        ub_index = len(eval_space) - 1
                    ub_2 = search_space[ub_index]

                    median = np.array([median_1, median_2])
                    lb = np.array([lb_1, lb_2])
                    ub = np.array([ub_1, ub_2])

                else:
                    mean = np.array([xbar_1, xbar_2])
                    mode = np.array([xbar_1, xbar_2])
                    median = np.array([xbar_1, xbar_2])
                    lb = np.array([max(norm.ppf(0.5 - CI/2) * eta_1 + xbar_1, self.min_x_1), max(norm.ppf(0.5 - CI/2) * eta_2 + xbar_2, self.min_x_2)])
                    ub = np.array([min(norm.ppf(0.5 + CI/2) * eta_1 + xbar_1, self.max_x_1), min(norm.ppf(0.5 + CI/2) * eta_2 + xbar_2, self.max_x_2)])
            else:
                mean = np.nan
                mode = np.nan
                median = np.nan
                lb = np.nan
                ub = np.nan

            # Binary updates
            xs, ss = zip(*[(x[2], s) for x, s in zip(xs, ss_org) if x is not None])
            xs, ss = np.array(xs), np.array(ss)
            n_p = len(xs)

            sigmoids = sigmoid_array(self.w_1_bin * ss + self.w_0_bin)
            sigmoids_x = np.multiply(xs, sigmoids)
            sum_sigmoids = np.sum(sigmoids)
            sum_sigmoids_x = np.sum(sigmoids_x)
            yhat = 1/(n_p + 1) * (self.mean_block) + n_p/(n_p + 1) * sum_sigmoids_x/sum_sigmoids

        else:
            if y[2]:
                mean = np.array([xbar_1, xbar_2])
                mode = np.array([xbar_1, xbar_2])
                median = np.array([xbar_1, xbar_2])
                lb = np.array([max(norm.ppf(0.5 - CI/2) * eta_1 + xbar_1, self.min_x_1), max(norm.ppf(0.5 - CI/2) * eta_2 + xbar_2, self.min_x_2)])
                ub = np.array([min(norm.ppf(0.5 + CI/2) * eta_1 + xbar_1, self.max_x_1), min(norm.ppf(0.5 + CI/2) * eta_2 + xbar_2, self.max_x_2)])
            else:
                mean = np.nan
                mode = np.nan
                median = np.nan
                lb = np.nan
                ub = np.nan
            yhat = self.mean_block

        return (round(y[0] * self.true_mean_1, 2), round(y[1] * self.true_mean_2, 2), f'{(y[2]):.8f}'), tuple((mean * np.array([self.true_mean_1, self.true_mean_2])).round(2)), tuple((mode * np.array([self.true_mean_1, self.true_mean_2])).round(2)), tuple((median * np.array([self.true_mean_1, self.true_mean_2])).round(2)), tuple((lb * np.array([self.true_mean_1, self.true_mean_2])).round(2)), tuple((ub * np.array([self.true_mean_1, self.true_mean_2])).round(2)), f'{(yhat):.8f}'
