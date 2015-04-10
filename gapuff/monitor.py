#coding=utf-8
import global_settings
import multi_puff; from multi_puff import model_puff_core

import input_info
import scipy.interpolate
import scipy.misc

import copy
import datetime

class monitor:

    def __init__(self, position, sigma=0.06):
        self.position = position
        self.sigma = sigma
        self.record = dict()
        self.simulate = dict()

    def targetfunc(self, mode=0):
        if mode == 0:
            return sum([(self.record[tick] - (0 if not self.simulate.has_key(tick) else self.simulate[tick]))**2 for tick in self.record]) 
        ##Here we can add new functions for example:
        ##elif mode==1:
        ##  return math.exp(-sum([(self.record[tick] - self.simulate[tick])**2 for tick in self.record]) / self.sigma)
        else:
            raise Exception("Unknown Probability or Target Function")

    def get_peeks_count(self):
        u"""插值求导获得极值点"""
        ticks = self.ticks
        chains = self.record_value
        x = numpy.array(ticks)
        y = numpy.array(chains)
        func = scipy.interpolate.interp1d(x, y, kind='cubic', copy=False)
        newx = numpy.arange(ticks[0], ticks[-1], 0.1)
        value_check = dict()
        for i in newx:
            dy = scipy.misc.derivative(func, i, dx=0.01, order=7, n=1)
            if numpy.abs(dy) < 0.3:
                if value_check.has_key(int(i-1)): value_check.pop(int(i-1))
                value_check.has_key[int(i)] = dy
        if len(value_check) <= 2:
            return 2
        else:
            return len(value_check)
