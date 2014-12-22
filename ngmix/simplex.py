#!/usr/bin/env python
#
# -*- Mode: python -*-
#
# $Id: Simplex.py,v 1.2 2004/05/31 14:01:06 vivake Exp $
# 
# Copyright (c) 2002-2004 Vivake Gupta (vivakeATlab49.com).  All rights reserved.
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
# USA
#
# This software is maintained by Vivake (vivakeATlab49.com) and is available at:
#     http://shell.lab49.com/~vivake/python/Simplex.py
#
# 1.2  ( 5/2004) - Fixed a bug found by Noboru Yamamoto <noboru.yamamotoATkek.jp>
#                  which caused minimize() not to converge, and reach the maxiter
#                  limit under some conditions.
#      ( 1/2011) - Added **kwargs where appropriate to enable passing additional
#                  static parameters to the objective function (Filip Dominec)
#
# 2014-11-20 - Modernized some style, altered notation to make more sense
#              to me.  Erin Sheldon, Brookhaven National Laboratory


""" Simplex - a regression method for arbitrary nonlinear function minimization

Simplex minimizes an arbitrary nonlinear function of N variables by the
Nedler-Mead Simplex method as described in:

Nedler, J.A. and Mead, R. "A Simplex Method for Function Minimization." 
    Computer Journal 7 (1965): 308-313.

It makes no assumptions about the smoothness of the function being minimized.
It converges to a local minimum which may or may not be the global minimum
depending on the initial guess used as a starting point.
"""
from __future__ import print_function
import math
import copy

class Simplex(object):
    def __init__(self, testfunc, guess, increments, kR = -1, kE = 2, kC = 0.5):
        """
        Initializes the simplex.

        parameters
        ----------
        testfunc: function
            the function to minimize
        guess: list
            an list containing initial guesses
        increments: list
            a list containing initial increments, perturbation size
        kR: float, optional
            reflection constant, default -1
        kE: float, optional
            expansion constant, default 2
        kC: float, optional
            contraction constant, default 0.5
        """
        self.testfunc = testfunc

        self.guess = guess
        self.pars  = copy.deepcopy(guess)

        self.increments = increments
        self.kR = kR
        self.kE = kE
        self.kC = kC
        self.numvars = len(self.guess)


    def minimize(self, epsilon=1.0e-4, maxiter=250, monitor=False, **kwargs):
        """
        Walk the simplex down to a local minima.

        parameters
        ----------
        epsilon: float, optional
            Convergence requirement.  Default 1.0e-4
        maxiter: integer, optional
            Maximum number of iterations. Default 250
        monitor: bool, optional
            If True, progress info is output to stdout. Default False
        **kw: optional keywords
            Keywords passed on to the function

        returns
        -------
        (pars, minval, niter)

        pars: list
            an list containing the final values
        minval: float
            smallest value of the function found
        niter: integer
            number of iterations taken to get here
        """
        self.simplex = []

        self.lowest = -1
        self.highest = -1
        self.secondhighest = -1

        self.vals = []
        self.current_val = 0
        # Initialize vertices
        for vertex in xrange(0, self.numvars + 3): # Two extras to store centroid and reflected point
            self.simplex.append(copy.copy(self.pars))
        # Use initial increments
        for vertex in xrange(0, self.numvars + 1):
            for x in xrange(0, self.numvars):
                if x == (vertex - 1):
                    self.simplex[vertex][x] = self.pars[x] + self.increments[x]
            self.vals.append(0)
        self.calculate_vals_at_vertices(**kwargs)

        for iter in xrange(1, maxiter+1):
            # Identify highest, second highest, and lowest vertices
            self.highest = 0
            self.lowest = 0
            for vertex in xrange(0, self.numvars + 1):
                if self.vals[vertex] > self.vals[self.highest]:
                    self.highest = vertex
                if self.vals[vertex] < self.vals[self.lowest]:
                    self.lowest = vertex
            self.secondhighest = 0
            for vertex in xrange(0, self.numvars + 1):
                if vertex == self.highest:
                    continue
                if self.vals[vertex] > self.vals[self.secondhighest]:
                    self.secondhighest = vertex
            # Test for convergence
            S = 0.0
            S1 = 0.0
            for vertex in xrange(0, self.numvars + 1):
                S = S + self.vals[vertex]
            F2 = S / (self.numvars + 1)
            for vertex in xrange(0, self.numvars + 1):
                S1 = S1 + (self.vals[vertex] - F2)**2
            T = math.sqrt(S1 / self.numvars)
            
            # Optionally, print progress information
            if monitor:
                print('#%d: Best = %s   Worst = %s' % (iter,self.vals[self.lowest],self.vals[self.highest]),)
                print("    ",end='')
                for vertex in xrange(0, self.numvars + 1):
                    print("[",end='')
                    for x in xrange(0, self.numvars):
                        print(" %.2f" % self.simplex[vertex][x],end='')
                    print(" ]",end='')
                print()

                
            if T <= epsilon:   # We converged!  Break out of loop!
                break;
            else:                   # Didn't converge.  Keep crunching.
                # Calculate centroid of simplex, excluding highest vertex
                for x in xrange(0, self.numvars):
                    S = 0.0
                    for vertex in xrange(0, self.numvars + 1):
                        if vertex == self.highest:
                            continue
                        S = S + self.simplex[vertex][x]
                    self.simplex[self.numvars + 1][x] = S / self.numvars

                self.reflect_simplex()

                self.current_val = self.testfunc(self.pars, **kwargs)

                if self.current_val < self.vals[self.lowest]:
                    tmp = self.current_val
                    self.expand_simplex()
                    self.current_val = self.testfunc(self.pars, **kwargs)
                    if self.current_val < tmp:
                        self.accept_expanded_point()
                    else:
                        self.current_val = tmp
                        self.accept_reflected_point()

                elif self.current_val <= self.vals[self.secondhighest]:
                    self.accept_reflected_point()

                elif self.current_val <= self.vals[self.highest]:
                    self.accept_reflected_point()

                    self.contract_simplex()
                    self.current_val = self.testfunc(self.pars, **kwargs)
                    if self.current_val < self.vals[self.highest]:
                        self.accept_contracted_point()
                    else:
                        self.multiple_contract_simplex(**kwargs)

                elif self.current_val >= self.vals[self.highest]:
                    self.contract_simplex()
                    self.current_val = self.testfunc(self.pars, **kwargs)
                    if self.current_val < self.vals[self.highest]:
                        self.accept_contracted_point()
                    else:
                        self.multiple_contract_simplex(**kwargs)

        # Either converged or reached the maximum number of iterations.
        # Return the lowest vertex and the current_val.
        for x in xrange(0, self.numvars):
            self.pars[x] = self.simplex[self.lowest][x]
        self.current_val = self.vals[self.lowest]
        return self.pars, self.current_val, iter

    def contract_simplex(self):
        for x in xrange(0, self.numvars):
            self.pars[x] = self.kC * self.simplex[self.highest][x] + (1 - self.kC) * self.simplex[self.numvars + 1][x]
        return

    def expand_simplex(self):
        for x in xrange(0, self.numvars):
            self.pars[x] = self.kE * self.pars[x]                  + (1 - self.kE) * self.simplex[self.numvars + 1][x]
        return

    def reflect_simplex(self):
        for x in xrange(0, self.numvars):
            self.pars[x] = self.kR * self.simplex[self.highest][x] + (1 - self.kR) * self.simplex[self.numvars + 1][x]
            self.simplex[self.numvars + 2][x] = self.pars[x] # REMEMBER THE REFLECTED POINT
        return

    def multiple_contract_simplex(self, **kwargs):
        for vertex in xrange(0, self.numvars + 1):
            if vertex == self.lowest:
                continue
            for x in xrange(0, self.numvars):
                self.simplex[vertex][x] = 0.5 * (self.simplex[vertex][x] + self.simplex[self.lowest][x])
        self.calculate_vals_at_vertices(**kwargs)
        return

    def accept_contracted_point(self):
        self.vals[self.highest] = self.current_val
        for x in xrange(0, self.numvars):
            self.simplex[self.highest][x] = self.pars[x]
        return

    def accept_expanded_point(self):
        self.vals[self.highest] = self.current_val
        for x in xrange(0, self.numvars):
            self.simplex[self.highest][x] = self.pars[x]
        return

    def accept_reflected_point(self):
        self.vals[self.highest] = self.current_val
        for x in xrange(0, self.numvars):
            self.simplex[self.highest][x] = self.simplex[self.numvars + 2][x]
        return

    def calculate_vals_at_vertices(self,**kwargs):
        for vertex in xrange(0, self.numvars + 1):
            if vertex == self.lowest:
                continue
            for x in xrange(0, self.numvars):
                self.pars[x] = self.simplex[vertex][x]
            self.current_val = self.testfunc(self.pars, **kwargs)
            self.vals[vertex] = self.current_val
        return

	
def main():
    import random


    def my_objective_function(args, static_par):
        return (args[0]-static_par[0])**2 + (args[1]-static_par[1])**2

    guess=[0.5+random.random() for i in xrange(2)]
    s = Simplex(my_objective_function, guess, [.01, .01])

    epsilon=1.0e-7
    maxiter=250
    values, minval, iter = s.minimize(epsilon=epsilon,
                                      maxiter=maxiter,
                                      monitor=True,
                                      static_par=[2,4])

    print()
    if iter==maxiter:
        print("not converged")
    else:
        print("converged")

    print('args       =', values)
    print('min value  =', minval)
    print('iterations =', iter)
	
if __name__ == '__main__':
    main()

