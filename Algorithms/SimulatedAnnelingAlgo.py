from logging import warning
from typing import functools, List
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
import copy

class SimulatedAnneling:
    def __init__(self, 
            Quality: functools,
            epochs: int =100,
            initial_temp: int = 99999 , 
            dimensions: int =2, 
            bounds: List = [-10,10], 
            stepsize: float = 0.1,
            precision: int = 6
        ):
        self.epochs = epochs
        self.initial_temp = initial_temp
        self.dimensions = dimensions
        self.bounds = bounds
        self.stepsize = stepsize
        self.Quality = Quality
        self.precision = precision
        
        self.history = None
        self.best = None
        self.time = None
        
    def train(self, write_txt = True):
        current_temp = self.initial_temp
        current_solution = np.around(np.random.uniform(self.bounds[0],self.bounds[1],self.dimensions),self.precision)
        current_quality = np.around(self.Quality(current_solution),self.precision)
        best_solution= current_solution.copy() 
        best_quality = round(self.Quality(best_solution),self.precision)
        self.history = []
        self.best = []
        self.time = datetime.now().strftime('%Y_%m%d_%H%M')
        for i in range(self.epochs):
            candidate = np.around(current_solution + np.random.randn(len(current_solution)) *self.stepsize,self.precision)
            quality_candidate = round(self.Quality(candidate),self.precision)
            self.history.append(quality_candidate)
            if quality_candidate < best_quality:
                best_solution = candidate.copy()
                best_quality = quality_candidate
            self.best.append(best_quality)
            
            
            if write_txt == True:
                with open(f'./record/SA_{self.dimensions}_{self.time}.txt','a') as f:
                    f.write(f'epochs: {i+1}| curr quality: {quality_candidate}|best quality:{best_quality}| best_solution: {best_solution} \n')
            
            diff = quality_candidate - current_quality
            t = current_temp *0.95
            current_temp = t
            mc = np.exp(-diff/t)
            if diff < 0 or random.random() < mc:
                current_solution = candidate.copy()
                current_quality = quality_candidate
            
        
        print(f'iter: {i+1} |best quality: {best_quality} \nbest_solution: {best_solution}')
    
    def plot_converge(self, print_record=False):             
        plt.figure(figsize= (15,3))
        plt.plot(self.best, label = 'best')
        if print_record == True:
            plt.plot(self.history, label = 'records')
        plt.title(f'SA_Convergence_D={self.dimensions}')
        plt.ylabel('Quality')
        plt.xlabel('epochs')
        plt.legend()
        plt.savefig(f'./plots/SA_{self.time}_D{self.dimensions}.png')
        plt.show()