#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:11:20 2023

@author: connorrice
"""

import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds

class data_extract:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, delimiter='\t')
        
        self.column_mapping = None
        self.user_mapping = None
        
    def order_names(self):
        all_order_names = self.df['MSW05_Order'].unique().tolist()
        
        return all_order_names
    
    def order_extract(self, order):
        order_data = self.df[self.df['MSW05_Order'] == order]
        order_data = order_data.drop(
            columns=[
                'MSW05_Order', 
                'MSW05_Family', 
                'MSW05_Genus', 
                'MSW05_Species'
                ]
            )
        order_data = order_data.loc[:, (order_data != -999.00).any(axis=0)]

        order_data['Users'] = range(len(order_data))
        order_data = order_data.rename(columns={'MSW05_Binomial': 'Users'}).set_index('Users')

        order_data.columns = [str(i) for i in range(len(order_data.columns))]
        order_data['1'] = order_data['1'].astype(int)

        extracted_data = pd.DataFrame(columns=['Users', 'Items', 'Values'])

        for column in order_data.columns:
            items_col = pd.DataFrame(
                {'Users': order_data.index, 'Items': [column] * len(order_data), 'Values': order_data[column]})
            extracted_data = pd.concat([extracted_data, items_col], ignore_index=True)

        extracted_data = extracted_data[extracted_data['Values'] != -999.00]

        extracted_data['Values'] = extracted_data['Values'].apply(lambda x: sum(map(int, str(x).split(';'))) if ';' in str(x) else int(x))
        extracted_data['Users'] = extracted_data['Users'].apply(lambda x: x[1] if isinstance(x, tuple) else int(x))
        extracted_data['Items'] = extracted_data['Items'].astype(int)
        
        print(extracted_data)
        
        vector_size = [len(order_data), len(order_data.columns)]
        
        return extracted_data, vector_size
    
    def data_split(self, data):
        num_observations = len(data)

        num_train = int(0.80 * num_observations)
        num_test = num_observations - num_train

        perm = np.random.permutation(data.shape[0])
        train = np.array(data[['Users', 'Items', 'Values']].iloc[perm[0:num_train]])
        test = np.array(data[['Users', 'Items', 'Values']].iloc[perm[num_train:]])

        datasets = {'train': train, 'test': test}

        return datasets

        
class ALS_Matrix_Completion:
    def __init__(self, datasets, vector_size, params, step_sizes):
        self.datasets = datasets
        self.vector_size = vector_size
        self.folds = []

        self.unique_U = []  # unique movies
        self.unique_V = []  # unique users
        self.U_to_V = {} # movies to users
        self.V_to_U = {} # users to movie

        self.hyperparams = params['hyperparams']
        self.best_hyperparams = self.hyperparams.copy()
        self.params = params['other']
        self.step_sizes = step_sizes

        self.U = []
        self.V = []
        
        self.best_errors = {'train': [], 'val': [], 'test': []}
    
    def mse(self, U, V, data):
        estimates = np.zeros(len(data))
        
        for i in range(len(estimates)):
            estimates[i] = np.dot(U[data[i, 1], :], V[data[i, 0], :])
        
        return np.sum((estimates - data[:, 2]) ** 2) / len(estimates)
        
    def k_folds(self):
        fold_size = len(self.datasets['train']) // self.params['k_folds']

        # Split the dataset into k folds
        for i in range(self.params['k_folds']):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < self.params['k_folds'] - 1 else None  # Last fold may be smaller
            fold = self.datasets['train'][start_idx:end_idx]
            self.folds.append(fold)
        
        return
    
    def closed_form_prep(self, k = 0, k_fold = False):
        if k_fold == True:
            self.unique_U = set()
            self.unique_V = set()
            train = np.array([])
            
            for i, fold in enumerate(self.folds):
                if i != k:
                    if len(train) == 0:
                      train = fold
                    else:
                      train = np.concatenate((train, fold), axis=0)
            
            self.unique_U = set(np.unique(train[:,1]))
            self.unique_V = set(np.unique(train[:,0]))
            
                    
            self.U_to_V = {i: train[train[:, 1] == i] for i in self.unique_U} # movies to users
            self.V_to_U = {i: train[train[:, 0] == i] for i in self.unique_V} # users to movie
            
            R_twiddle = np.zeros((self.vector_size[1], self.vector_size[0]))
            R_twiddle[train[:, 1].astype(int), train[:, 0].astype(int)] = train[:, 2]
            self.U, _, self.V = svds(R_twiddle, self.params['d'])
            self.V = self.V.T
        else:
            self.unique_U = set(np.unique(self.datasets['train'][:,1]))
            self.unique_V = set(np.unique(self.datasets['train'][:,0]))
            
            self.U_to_V = {i: self.datasets['train'][self.datasets['train'][:, 1] == i] for i in self.unique_U} # movies to users
            self.V_to_U = {i: self.datasets['train'][self.datasets['train'][:, 0] == i] for i in self.unique_V} # users to movie
            
            R_twiddle = np.zeros((self.vector_size[1], self.vector_size[0]))
            R_twiddle[self.datasets['train'][:, 1], self.datasets['train'][:, 0]] = self.datasets['train'][:, 2]
            self.U, _, self.V = svds(R_twiddle, self.params['d'])
            self.V = self.V.T
        
        return
    
    def closed_form_u(self):
        
        new_U, I_size = np.zeros_like(self.U), self.U.shape[1]

        for _, u in enumerate(self.unique_U):
            V_j = self.V[self.U_to_V[u][:, 0], :]
            new_U[u, :] = np.linalg.solve(
                np.sum(V_j[:, :, None] @ V_j[:, None, :] + self.hyperparams['lam_U'] * np.eye(I_size), axis=0),
                np.sum(V_j * self.U_to_V[u][:, 2][:, None], axis=0)
            )

        return new_U

    def closed_form_v(self):
        
        new_V, I_size = np.zeros_like(self.V), self.V.shape[1]

        for _, u in enumerate(self.unique_V):
            U_j = self.U[self.V_to_U[u][:, 1], :]
            new_V[u, :] = np.linalg.solve(
                np.sum(U_j[:, :, None] @ U_j[:, None, :] + self.hyperparams['lam_V'] * np.eye(I_size), axis=0),
                np.sum(U_j * self.V_to_U[u][:, 2][:, None], axis=0)
            )

        return new_V

    def get_UV(self): 
        k = 0
        
        while True:
            k += 1
            old_U, old_V = (np.copy(self.U), np.copy(self.V))
            self.U = self.closed_form_u()
            self.V = self.closed_form_v()
            U_delta, V_delta = (np.max(np.abs(self.U - old_U)), np.max(np.abs(self.V - old_V)))
            if k % 100 == 0:
              print(f"\rIteration {k} Hyperparamers: {self.hyperparams}, U delta: {np.round(U_delta, 5)} V delta: {np.round(V_delta, 5)}", ' ', ' ', end='')
            if max(U_delta, V_delta) <= self.params['delta']:
                break

        return self.U, self.V
    
    def cross_validation(self):
        self.k_folds()
        errors = []
        
        for k in range(self.params['k_folds']):
            self.closed_form_prep(k, k_fold = True)
            U, V = self.get_UV()
            error = self.mse(U, V, self.folds[k])
            errors.append(error)         
            
        return np.mean(errors)
    
    def new_hyperparams(self):
        
        new_params = {
            param: np.abs(np.round(self.hyperparams[param] + random.uniform(-self.step_sizes[param], self.step_sizes[param]), 5))
            for param in self.hyperparams
        }
        
        for key in new_params:
            if new_params[key] < 0.05:
                new_params[key] = self.hyperparams[key]
        
        return new_params
    
    def drunken_sailor_search(self):
        best_error = np.inf
        count = 0
    
        while True:
            old_params = self.hyperparams.copy()
            self.hyperparams = self.new_hyperparams()
    
            cross_val_error = self.cross_validation()
    
            if cross_val_error < best_error:
                self.best_errors['val'].append(cross_val_error)
    
                self.closed_form_prep(0)
                U, V = self.get_UV()
    
                for key in self.datasets:
                    error = self.mse(U, V, self.datasets[key])
                    self.best_errors[key].append(error)
    
                print(f"Best Errors: {[f'{key}: {self.best_errors[key][-1]}' for key in self.best_errors.keys()]}")
    
            else:
                count += 1
                self.hyperparams = old_params
    
                if count == 10:
                    break
    
        return self.params, self.best_errors


def main():
    # Example usage
    # primate_data, primate_info = get_primate_data('/Users/connorrice/Downloads/PanTHERIA_WR05_mammals.txt')

    data = data_extract('PanTHERIA_WR05_mammals.txt')

    order_names = data.order_names()

    for order in order_names:

        order_data, order_info = data.order_extract(order)

        order_datasets = data.data_split(order_data)

        params = {
            'hyperparams': {
                'lam_U': .1,
                'lam_V': .1,
            },
            'other': {
                'd': 5,
                'delta': 1e-5,
                'k_folds': 10
            }
        }

        # Define the step sizes for the random walk
        step_sizes = {
            'lam_U': .01,
            'lam_V': .01,
        }

        param_search = ALS_Matrix_Completion(order_datasets, order_info, params, step_sizes)

        params, best_errors = param_search.drunken_sailor_search()

        # Plotting both train and test error as a function of d on the same plot.
        for dataset, error_values in best_errors.items():
            plt.plot(range(1, len(error_values) + 1), error_values, label=dataset)

        plt.xlabel('Succesful Random Step')
        plt.ylabel('Error attained')
        plt.title('Error with modified U, V estimator')
        plt.legend(loc='best')
        plt.show()


if __name__ == "__main__":
    main()