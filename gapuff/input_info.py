#coding=utf-8
u"""This file is used to generate a met-condition-field."""
import global_settings
from global_settings import *

import numpy as np
import os
import math
import datetime

from met_def import met_def

class met_info:
    u"""met_info用来进行数据前处理并生成met_def的实例met_info有五种模式：
0 - 全局恒定气象场，返回type=0的met_def实例
1 - 气象站数据，返回type=1的met_def实例
2 - 存储的气象数据，type=1
3 - 存储的气象数据，type=2
4 - WRF输出的history文件
"""

    def __init__(self, sourceinfo, mode=0, dataset="wrfout.ncf", test=False):
        self.sourceinfo = sourceinfo
        self.mode = mode
        self.test = test
        self.dataset = dataset

    def get_met(self, center=None, accident_time=None):
        #In test mode, we generate a simple constant field
        if self.test:
            return self.__simple_test()
        elif self.mode == 4:
            return self.__read_wrf(self.dataset, center=self.sourceinfo.position, accident_time=self.sourceinfo.time)
        elif self.mode == 3 or self.mode == 2:
            return self.__load(self.dataset, mode)
        elif self.mode == 1:
            return self.__read_sam(accident_time=self.sourceinfo.time)
        elif self.mode == 0:
            return met_def(0, self.dataset, [7200])
        raise Exception("mode error!")

    def __load(self, datafile, mode):
        dataset = np.load(datafile)
        data = dataset['data']
        seq = dataset['seq']
        return met_def(mode-1,data,seq)

    def __read_sam(self, accident_time):
        raise NotImplementedError()

    def __read_wrf(self, center, accident_time):
        import wrf_processor
        dataset = wrf_processor.wrf_processor(self.dataset)
        data = dataset.read_met(center, accident_time)
        seq = dataset.read_ticks()
        return met_def(2, data, seq)

    def __simple_test(self):
        data = (3,0,0,4)
        seq = [7200]
        return met_def(0, data, seq)

    def __complex_test(self):
        MAXLENGTH = HALF_INTERVAL * math.sqrt(2) + 1
        basedata = np.empty((12,GRID_SIZE,GRID_SIZE,4))
        for t in range(12):
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    #start variable section
                    grid = basedata[t,i,j];
                    SPEED = int(math.sqrt((i-HALF_INTERVAL)**2 + (j-HALF_INTERVAL)**2) / MAXLENGTH * 5) + 3
                    STAB = 4 - int(math.sqrt((i-HALF_INTERVAL)**2 + (j-HALF_INTERVAL)**2) / MAXLENGTH * 4)
                    ZSPEED = int(math.sqrt((i-HALF_INTERVAL)**2 + (j-HALF_INTERVAL)**2) / MAXLENGTH * 3) + 1
                    ARC = math.pi / 2 / (GRID_SIZE - 1) * i
                    grid[0] = math.sin(ARC) * SPEED + tindex * 0.2 + tindex * 0.1 #U Speed
                    grid[1] = math.cos(ARC) * SPEED + tindex * 0.2 + tindex * 0.1 #V Speed
                    grid[2] = ZSPEED #Z Speed
                    grid[3] = STAB #Stabilities
                    #end variable section
        seq = range(600,7201,600)
        np.savez(self.dataset, data=result, seq=seq)
        return met_def(2, data, seq)

class source_info:

    def __init__(self, dataset=None, position=(348013.93273281929, 3471433.9195643286), time=None, test=False):
        self.test = test
        if test:
            self.reverse_source = False
            self.reverse_position = False
            self.time = datetime.datetime(2011, 1, 21, 12)
            self.dataset = 'testsrc.txt'
            self.position = (348013.93273281929, 3471433.9195643286)
            return
        self.reverse_source = not dataset
        self.reverse_position = not position
        self.dataset = dataset
        self.position = position
        self.time = datetime.datetime.now()

    def read_rate(self):
        if self.test:
            result = [50e6] * 12
            #result += [850e6 / 6.0] * 30
            #result += [840e6 / 6.0] * 30
            #result += [780e6 / 6.0] * 30
            #result += [50e6 / 6.0] * (45 * 6)
            return result
        elif isinstance(self.dataset, list):
            return self.dataset
        #Read from file
        else:
            fp = open(dataset)
            fp.close()
            raise NotImplementedError("Read from file!")



