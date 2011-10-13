#coding=utf-8
u"""This file is used to generate a met-condition-field."""
import global_settings
from global_settings import *

import numpy as np
import os
import math
import datetime

class MetPreProcessor:
    u"""前处理气象数据产生所需要的气象场"""
    def __init__(self, mode=1, simple=False, dataset="wrfout.ncf"):
        self.mode = mode
        self.simple = simple
        self.dataset = dataset
        pass

    # 模式II是假设，气象数据是从SAM上获取的历史记录，这个时候是每10分钟一次的均值
    def GenerateTestFieldMode2(self, start_time):
        now = datetime.datetime.now()
        while start_time < now:
            #TODO: get met data from SAM
            start_time += datetime.timedelta(seconds=600)
        pass

    def GenerateTestFieldMode1(self, center = None):
        import source_wrf
        met_source = source_wrf.met_wrf_source(self.dataset)
        accident_time = datetime.datetime(2011, 1, 21, 12)
        if center is None and __debug__:
            center = (348013.93273281929, 3471433.9195643286)
        else:
            raise Error("position cannot be blank in release mode")
        self.met_field = met_source.generate_met(center, accident_time)
        self.met_seq = met_source.generate_met_seq()
        return self.met_field

    def GenerateTestFieldSeq(self):
        return self.met_seq

    def GenerateConstantTestField(self):
        self.met_field = np.zeros((1, GRID_SIZE, GRID_SIZE, 4))
        self.met_field[0,:,:,0].fill(3)
        self.met_field[0,:,:,1].fill(0)
        self.met_field[0,:,:,2].fill(0)
        self.met_field[0,:,:,3].fill(3)
        self.met_seq = [7200]
        return self.met_field

    def GenerateTestField(self):
        #Get From WRF
        if self.mode == 1:
            return self.GenerateTestFieldMode1()
        #Get From SAM: Not implemented. We need communicate with COM1 interface to read data from SAM.
        if self.mode == 2:
            return self.GenerateTestFieldMode2()
        #If the field is simple, we generate a constant field
        if self.simple:
            return self.GenerateConstantTestField()
        #Otherwise, we generate a very complex met-field to test my model.
        #if the sample file exsits, load sample file.
        if os.path.exists('samplewind.npy'):
            print "We've found a generated sample file. Use this file"
            self.met_seq = range(600,7201,600)
            return np.load('samplewind.npy');
        #else, we generate a sample file
        print "We haven't found any sample files. We'll generate a new one"
        MAXLENGTH = HALF_SIZE * math.sqrt(2) + 1
        basedata = np.empty((1,GRID_SIZE,GRID_SIZE,4))
        for tindex in xrange(12):
            basedata_t = np.empty_like(basedata)
            for i in xrange(GRID_SIZE):
                for j in xrange(GRID_SIZE):
                    #start variable section
                    grid = basedata_t[0,i,j];
                    SPEED = int(math.sqrt((i-HALF_SIZE)**2 + (j-HALF_SIZE)**2) / MAXLENGTH * 5) + 3
                    STAB = 4 - int(math.sqrt((i-HALF_SIZE)**2 + (j-HALF_SIZE)**2) / MAXLENGTH * 4)
                    ZSPEED = int(math.sqrt((i-HALF_SIZE)**2 + (j-HALF_SIZE)**2) / MAXLENGTH * 3) + 1
                    ARC = math.pi / 2 / (GRID_SIZE - 1) * i
                    grid[0] = math.sin(ARC) * SPEED + tindex * 0.2 + tindex * 0.1 #U Speed
                    grid[1] = math.cos(ARC) * SPEED + tindex * 0.2 + tindex * 0.1 #V Speed
                    grid[2] = ZSPEED #Z Speed
                    grid[3] = STAB #Stabilities
                    #end variable section
            if tindex == 0:
                result = basedata_t
            else:
                result = np.concatenate((result, basedata_t))
                print tindex
        np.save('samplewind', result)
        self.met_seq = range(600,7201,600)
        return result

class SourcePreProcessor:

    def GenerateTestSource(self):
        #Generate a test-source: Constant release of 2t materials during 30min.
        result = []
        BASEFACTOR = 4.0e9 / 360
        for i in xrange(0,360):
            result.append(BASEFACTOR)
        return result

if __name__ == "__main__":
    a = MetPreProcessor(mode=0, simple=True)
    result = a.GenerateTestField();
    #Test a simple field
    seq = a.GenerateTestFieldSeq();
    print result.shape
    print seq
    result2 = a.GenerateTestSource();
    print len(result2)
    print "Press ENTER to end the program."; raw_input()
else:
    #sample_gen被调用则说明需要从WRF或者SAM上读取气象网格
    pass


