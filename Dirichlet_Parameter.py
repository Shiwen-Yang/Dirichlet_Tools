
import numpy as np
import torch
from torch.distributions import Dirichlet, Bernoulli, Uniform
import pandas as pd


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Given observations a Dirichlet random variable, estimate the parameter alpha using MOME and MLE
class Estimate:
    def __init__(self, observation, tol = 10e-5):
        
        #observation: (n, p) shape array
        #observation are the i.i.d. samples of a Dirichlet random variable
        #tolerance is the stopping criterion for the gradient descent method
        #MOME is the method of moment estimator for Dirichlet random variables
        #MLE is the maximum likelihood estimator

        self.observation = observation
        self.tolerance = tol
        self.MOME = self.method_of_moment()
        self.MLE = self.MLE()

    
    def method_of_moment(self):

        #Given observations, compute the MOME

        E = self.observation.mean(dim = 0)
        V = torch.var(self.observation, dim = 0)

        mome = E * (E*(1 - E)/V - 1)

        return(mome)
    
    def likelihood(self, alpha, log_likelihood = False):

        #Given observations, compute the likelihood or log-likelihood

        alpha.to(device)
        alphap = alpha - 1

        c = torch.exp(torch.lgamma(alpha.sum()) - torch.lgamma(alpha).sum())
        likelihood = c * (self.observation ** alphap).prod(axis=1)
        likelihood.to("cpu")
        del(alpha, alphap, c)
    
        if log_likelihood:
            return(torch.log(likelihood).sum())
        else:
            return(likelihood.sum())
        
    def mean_log(self, alpha):

        #A stand-alone function, it calculates the expected value of the log(Dir(alpha)) distribution

        mean_of_log = (torch.digamma(alpha) - torch.digamma(alpha.sum()))

        return(mean_of_log)
    
    def var_log(self, alpha, inverse = False):

        #A stand-alone function, it calculates the variance of the log(Dir(alpha)) distribution
        #When inverse is true, the Sherman-Morrison formula is used to compute the inverse of the variance matrix
    
        p = alpha.shape[0]
        one_p = torch.ones(p).unsqueeze(1)
        c = torch.polygamma(1, alpha.sum())
        Q = -torch.polygamma(1, alpha)
        Q_inv = torch.diag(1/Q)

        if inverse:

            numerator = (Q_inv @ (c*one_p @ one_p.T) @ Q_inv)

            denominator = (1 + c * one_p.T @ Q_inv @ one_p)

            var_inv = Q_inv - numerator/denominator

            return(var_inv)
        else:
            
            return(torch.diag(Q) + c)

    def MLE(self):

        #Given observations, compute the maximum likelihood estimator using newton gradient descent
        #The gradient descent is initialized on the computed MOME
        
        empirical_avg_log = torch.log(self.observation).mean(dim = 0)

        initialization = self.MOME

        tol = self.tolerance

        next_par = initialization

        go = True
        i = 1
        while go and i <= 1000:
            var_inv = self.var_log(next_par, inverse = True)
            
            log_mean = empirical_avg_log - self.mean_log(next_par)

            step = var_inv @ (log_mean)

            next_par = next_par - step

            go = (torch.norm(step) > tol)

            i += 1

        return(next_par)