import numpy as np
import matplotlib.pyplot as plt
from typing import functools, List
from datetime import datetime


class SOS:
    
    def __init__(self, Quality: functools, pop_size: int = 30, dimension: int=2, epochs: int=100, bounds: list=[-100,100], precision: int = 6):
        # 初始化所有參數
        self.Quality =  Quality
        self.pop_size = pop_size
        self.dimension = dimension
        self.epochs = epochs
        self.up_bound = bounds[1]
        self.low_bound = bounds[0]
        self.precision = precision
        
        self.population = None # 初始母體的儲存
        self.best = None # 最佳答案儲存
        self.best_solution = None # 最佳解的儲存
        self.history_best = None #歷史最佳答案的儲存
        self.time = None
        
    def tweak(self,xi):
        # 調整演化後的解
        for i in range(len(xi)):
            if xi[i] >= self.up_bound:
                xi[i] = self.up_bound
            elif xi[i] <= self.low_bound:
                xi[i] = self.low_bound
        return xi
    
    def generate_population(self):
        
        #生成一個初始母體
        self.population = np.around(np.random.uniform(self.low_bound, self.up_bound, (self.pop_size,self.dimension)),self.precision)
        #生成一個初始Xbext
        self.best_solution = sorted(self.population, key=lambda x: self.Quality(x))[0]
        self.best = self.Quality(self.best_solution)
        
    def change_pop(self, x, x_new):
        #比較大小
        if self.Quality(x) >= self.Quality(x_new):
            return x_new
        else:
            return x
    
    def mutualism(self, index_i):
        # 目前的 i 的個體
        xi = self.population[index_i].copy()
        # 隨機亂數 index 一個非 i 的個體
        xj_index = np.random.permutation(np.delete(np.arange(self.pop_size),index_i))[0]
        xj = self.population[xj_index].copy()
        # 計算 bf1,bf2
        bf1, bf2 = np.random.randint(1,3,2)
        # 計算 mutual vec
        mutual_vec = (xi+xj)/2
        # 計算新的解
        xi_new = np.around(self.tweak(xi + np.random.uniform(0,1) * (self.best_solution-mutual_vec*bf1)),self.precision)
        xj_new = np.around(self.tweak(xj + np.random.uniform(0,1) * (self.best_solution-mutual_vec*bf2)),self.precision)
        # 若比較好,則各自取代各自 index 所代表的個體
        self.population[index_i] = self.change_pop(xi,xi_new)
        self.population[xj_index] = self.change_pop(xj,xj_new)
        
    def commensalism(self, index_i):
        xi = self.population[index_i].copy()
        # 隨機亂數 index 一個非 i 的個體
        xj_index = np.random.permutation(np.delete(np.arange(self.pop_size),index_i))[0]
        xj = self.population[xj_index].copy()
        # 計算新的xi, 並確認是不是在domian range內, 如果比較好就取代 index 所代表的個體
        xi_new = np.around(xi + np.random.uniform(-1,1) *(self.best_solution-xj),self.precision)
        xi_new = self.tweak(xi_new)
        self.population[index_i] = self.change_pop(xi,xi_new)
        
    def parasitism(self, index_i):
        # 隨機亂數 index 一個非 i 的個體
        xj_index = np.random.permutation(np.delete(np.arange(self.pop_size),index_i))[0]
        xj = self.population[xj_index].copy()
        # 隨機取代寄生vector中的某個值
        parasite_vec = self.population[index_i].copy()
        parasite_vec[np.random.randint(0,self.dimension)] = np.around(np.random.uniform(self.low_bound,self.up_bound),self.precision)
        self.population[xj_index] = self.change_pop(xj, parasite_vec)
        
    def proceed(self, write_txt = True):
        self.generate_population() #生成一個最佳解
        self.history_best = []
        # 讀目前時間, 記錄用
        self.time = datetime.now().strftime('%Y_%m_%d %H%M')
        for i in range(self.epochs):
            
            for j in range(self.pop_size):
                # SOS 的流程
                self.mutualism(j)
                self.commensalism(j)
                self.parasitism(j)
                
            self.best_solution = np.around(sorted(self.population, key=lambda x: self.Quality(x))[0], self.precision)
            self.best = round(self.Quality(self.best_solution),self.precision)
            self.history_best.append(self.best)
            # 寫入txt檔
            if write_txt == True:
                with open(f'./record/SOS_{self.dimension}_{self.time}.txt','a') as f:
                    f.write(f'iter:{i+1} | best: {self.best}| best_solution: {self.best_solution} \n')
            
            
        #印出最佳解
        print(f'iter: {i+1} |final solution: {self.best} \n{self.best_solution}')
    
    def plotting(self):
        plt.figure(figsize =(15,3))
        plt.plot(self.history_best, label = 'best')
        plt.xlabel('epochs', fontsize= 15)
        plt.ylabel('Quality', fontsize = 15)
        plt.title(f'SOS_Convergence_D={self.dimension}', fontsize = 18)
        plt.legend()
        plt.savefig(f'./plots/SOS_{self.time}_D{self.dimension}.png')
        plt.show()