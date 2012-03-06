#coding=utf-8
import global_settings
import multi_puff; from multi_puff import model_puff_core

import source_main
import scipy.interpolate
import scipy.misc

import copy
import datetime

class monitor:

    def __init__(self, pos_x, pos_y, height, sigma=0.06):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos = (pos_x, pos_y)
        self.height = height
        self.ticks = []
        self.record_value = []
        self.sim_value = []
        self.value_list = None
        self.sigma = sigma
        self.src = None
        self.metfield = None
        self.metseq = None
        self.model_runned = False

    def set_source(self, src):
        self.src = src

    def set_met(self, metfield, metseq):
        self.metfield = metfield
        self.metseq = metseq

    def init_model(self):
        if self.src is None or self.metfield is None or self.metseq is None:
            raise Error("[Monitor] Parameter Error")
        self.model = model_puff_core(self.src, self.metfield, self.metseq)
        self.model.recepter_height = self.height
        self.model_runned = False

    def run_model(self, no_debug=True):
        self.model.run_point(point=[self.pos_x, self.pos_y, self.height], height_in=1, ticks=self.ticks, force_no_debug=no_debug)
        self.sim_value = self.model.point_list
        self.model_runned = True

    def add_record(self, tick, value):
        self.ticks.append(tick)
        self.record_value.append(value)
        self.sim_value.append(0)

    def list_record(self, list_item=False):
        self.value_list = zip(self.ticks, self.record_value, self.sim_value)
        if __debug__ and list_item:
            for a in self.value_list:
                print a
        return self.value_list

    def calc_probility(self):
        if not self.model_runned:
            self.run_model()
        self.list_record()
        sum = 0.0
        for tick, rec, sim in self.value_list:
            single_error = (rec-sim)**2
            sum += single_error
        return sum

    def sum_error(self, mode=0):
        return self.calc_probility()

    def find_peeks(self):
        u"""插值求导获得极值点"""
        ticks = self.ticks
        chains = self.record_value
        x = numpy.array(ticks)
        y = numpy.array(chains)
        func = scipy.interpolate.interp1d(x, y, kind='cubic', copy=False)
        newx = numpy.arange(ticks[0], ticks[-1], 0.1)
        value_check = dict()
        for i in newx:
            try:
                dy = scipy.misc.derivative(func, i, dx=0.01, order=7, n=1)
                if numpy.abs(dy) < 0.3:
                    try:
                        value_check.pop(int(i-1))
                    except:
                        pass
                    try:
                        value_check.pop(int(i))
                    except:
                        pass
                    value_check[int(i)] = dy
            except:
                pass
        if __debug__:
            print value_check
            #一般来说不可能一峰值都没有，这样的话我们就算一个平均值
        if len(value_check) <= 2:
            return 2
        else:
            return len(value_check)