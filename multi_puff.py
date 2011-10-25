#coding=utf-8
u"""
一个多烟团模型，完全采用numpy和numexpr实现，目标是为了给GPU运算做基准测试
numpy和numexpr都采用了MKL加速
空间复杂度O(n)=GRID_SIZE^2*puff*O(exp)
时间复杂度o(n)=2*O(n)*timepoints
平均一个时刻的一个烟团在采用numexpr加速的情况下需要
GRID-SIZE       TIME
4096*4096       3.4s
2048*2048       1.8s  -> current choice
1024*1024       0.3s
"""

## Here is some note:
## Walking-length is a tuple (xy, xz). At beginning, xy equals xz until the stability changed so new walking-length
## are calculated.
## The position of smoke is a tuple (x, y, z). Remember x, y and z are all float numbers, not integers.
## Met-condition is a tuple (u,v,z,stab)
## Z-speed is ignored in this version of puff-model. Because of the diffusion coefficients didn't consider the z-wind
## Bundled WRF Processor are HARDCODED, please modify it if needed. "source_wrf.py" is the only script file licensed under MIT License.
## Some comments are written in Chinese. Please ignore them. They will be translated into English in next release

import wingdbstub

## FOR DEBUG USE:
if __debug__:
    import matplotlib.pyplot as plt

import os
os.environ["CUDA_PROFILE"] = "1"
os.environ["CUDA_DEVICE"] = "0"

import numpy
import numexpr as ne
import math
import datetime

from smoke_def import smoke_def

from global_settings import *
import source_main

try:
    import gpu_core
    from gpu_core import run_gpu_conc
except:
    cuda_disabled = True
    if __debug__:
        print "We found no cuda card. We'll use cpu only."
else:
    cuda_disabled = False
    if __debug__:
        print "We found a cuda card and pycuda initialized."

#Set up numexpr
ne.set_vml_accuracy_mode("high")

#建立坐标网格
HG = GRID_SIZE / 2
GridIdxY, GridIdxX = numpy.mgrid[HG-1:-HG-1:-1, -HG:HG]
CentralPositionMatrixX = ne.evaluate("GridIdxX*GRID_INTERVAL+0.5*GRID_INTERVAL")
CentralPositionMatrixY = ne.evaluate("GridIdxY*GRID_INTERVAL-0.5*GRID_INTERVAL")

class model_puff_core:

    def __init__(self, release, met_field, met_seq):
        self.release = release
        self.met_field = met_field
        self.met_seq = met_seq
        self.smoke_list = []
        self.result_list = []


#########################ENVELOP###############################
    def run_envelop(self, loc, height=1):
        if len(loc) == 0:
            raise Error("You must specify LOC values to calculate envelop")
        for l in loc:
            if l <= 0:
                raise Error("LOC value < 0?!")
        self.loc = loc
        return self.run_core_contour(loc=loc, height_in=height, envelop=True)
#########################END OF ENVELOP########################

#########################GIVEN_TIMES###########################
    def run_core_time(self, time_list, height_in=1):
        """time_list is the list of REAL time, NOT ticks"""
        tick_list = []
        for t in time_list:
            tick_list.append(t // TIMESTEP)
        return self.run_core_contour(self, height_in=height_in, ticks=ticks)
#########################END OF GIVEN_TIMES#####################

#########################CONTOUR################################
    def run_contour(self, height_in=1):
        self.smoke_list = []
        self.result_list = []
        return self.run_core_contour(height_in)

    def run_core_contour(self, height_in=1, envelop=False, ticks=None, loc=[]):
        tick = 0
        #Total Estimation Time is DURATION seconds
        while tick * TIMESTEP < DURATION:
            if __debug__:
                if ((not envelop) and (not ticks is None) and (tick in ticks)) or ((ticks is None or envelop) and (tick % OUTSTEP == 0)):
                    print "current time-tick is %d" % tick
            #判断烟团释放过程是否结束
            if tick < len(self.release):
                #表示此刻有无烟团释放
                if self.release[tick] > 0:
                    #加入烟团list
                    self.smoke_list.append(smoke_def(self.release[tick], self.met_field, self.met_seq, tick, pos=(0.0, 0.0, height_in)))
            #追踪当前每个烟团的位置，并且计算浓度场
            if cuda_disabled: #CPU Func
                empty_conc_matrix = numpy.zeros_like(CentralPositionMatrixX)
                for smoke in self.smoke_list:
                    if not smoke.invalid:
                        smoke.walk()
                    #只有当需要输出的时刻才计算浓度场
                    if ((not envelop) and (not ticks is None) and (tick in ticks)) or ((ticks is None or envelop) and (tick % OUTSTEP == 0)):
                        #Prepare some variables used in numexpr.evaluate
                        X, Y, Z = smoke.pos
                        stability = int(smoke.met[3])
                        mass = smoke.mass
                        #计算扩散系数
                        diffc = smoke.diffusion_coefficents(stability, smoke.walkinglength, smoke.windspeed)
                        x, z = diffc; y = x
                        PI = math.pi
                        height = 1.0
                        current_smoke_conc = ne.evaluate("mass/((2*PI)**1.5*x*y*z)*exp(-0.5*((CentralPositionMatrixX-X)/x)**2)*exp(-0.5*((CentralPositionMatrixY-Y)/y)**2)*(exp(-0.5*((Z-height)/z)**2) + exp(-0.5*((Z+height)/z)**2))")
                        empty_conc_matrix = ne.evaluate("empty_conc_matrix + current_smoke_conc")
                self.result_list.append(empty_conc_matrix)
            else: #GPU Func
                diffc = []; mass = []; stab = []; pos = [];
                for smoke in self.smoke_list:
                    if not smoke.invalid:
                        smoke.walk()
                    #只有当需要输出的时刻才计算浓度场
                    if ((not envelop) and (not ticks is None) and (tick in ticks)) or ((ticks is None or envelop) and (tick % OUTSTEP == 0)):
                        pos += list(smoke.pos)
                        #计算扩散系数
                        smoke.diffusion_coefficents(int(smoke.met[3]), smoke.walkinglength, smoke.windspeed)
                        diffc += list(smoke.diffc)
                        mass.append(smoke.mass)
                if len(mass) == 0:
                    tick += 1
                    continue
                dest = run_gpu_conc(pos=numpy.array(pos).astype(numpy.float32), diffc=numpy.array(diffc).astype(numpy.float32), mass=numpy.array(mass).astype(numpy.float32),
                                            height=1.0, GRID_WIDTH=GRID_SIZE, gridw=GRID_INTERVAL, smoke_count=len(mass))
                empty_conc_matrix = dest.reshape((GRID_SIZE,GRID_SIZE), order='C')
                self.result_list.append(empty_conc_matrix)
            #时钟+1
            tick += 1
            if envelop:
                self.envelop_list = []
                for l in loc:
                    self.envelop_list.append(self.make_envelop(self.result_list, l))
        return 0

    def make_envelop(self, result_list, loc):
        base = numpy.zeros_like(result_list[0])
        for r in result_list:
            base = ne.evaluate("base+where(r>loc, 1, 0)")
        return base


    def run_gpu_contour(self, height_in=1, envelop=False, ticks=None, loc=[]):
        NOUSECODE="""
        tick = 0
        #做一点赋值，方便调试
        gridw = GRID_INTERVAL
        GRID_WIDTH = GRID_SIZE
        height = height_in
        #Total Estimation Time is DURATION seconds
        while tick * TIMESTEP < DURATION:
            if __debug__:
                if (not ticks is None and tick in ticks) or (ticks is None and tick % OUTSTEP == 0):
                    print "current time-tick is %d" % tick
            #判断烟团释放过程是否结束
            if tick < len(self.release):
                #表示此刻有无烟团释放
                if self.release[tick] > 0:
                    #加入烟团list
                    self.smoke_list.append(smoke_def(self.release[tick], self.met_field, self.met_seq, tick, pos=(0.0, 0.0, height_in)))
            #追踪当前每个烟团的位置，并且计算浓度场
            diffc = []; mass = []; stab = []; pos = [];
            for smoke in self.smoke_list:
                if not smoke.invalid:
                    smoke.walk()
                #只有当需要输出的时刻才计算浓度场
                if (not ticks is None and tick in ticks) or (ticks is None and not smoke.invalid and tick % OUTSTEP == 0):
                    pos += list(smoke.pos)
                    smoke.diffusion_coefficents(int(smoke.met[3]), smoke.walkinglength, smoke.windspeed)
                    diffc += list(smoke.diffc)
                    mass.append(smoke.mass)
            if len(mass) == 0:
                tick += 1
                continue
            dest = gpu_core.run_gpu_conc(pos=numpy.array(pos).astype(numpy.float32), diffc=numpy.array(diffc).astype(numpy.float32), mass=numpy.array(mass).astype(numpy.float32),
                                        height=height, GRID_WIDTH=GRID_WIDTH, gridw=gridw, smoke_count=len(mass))
            #TODO: 用numexpr来计算包络线
            for loc in self.loc:
                pass
            empty_conc_matrix = dest.reshape((GRID_SIZE,GRID_SIZE), order='C')
            self.result_list.append(empty_conc_matrix)
            #时钟+1
            tick += 1
        return 0"""
        raise Error("Use run_core_contour instead!")
#########################END OF CONTOUR###############################


def plot_debug(result):
    from pylab import *
    contours = (70.9/22.4, 70.9/22.4*3, 70.9/22.4*20)
    contour(result, contours)
    grid(True)
    show()


if __name__ == '__main__':
    #读取输入数据
    testmet = source_main.MetPreProcessor(mode=0, simple=True, dataset='wrfout.ncf')
    testsrc = source_main.SourcePreProcessor()

    print "Reading source data"
    ReleaseQ = testsrc.GenerateTestSource()
    print "Reading met data"
    MetField = testmet.GenerateTestField()
    print "Reading met sequence"
    MetSeq = testmet.GenerateTestFieldSeq()
    model = model_puff_core(ReleaseQ, MetField, MetSeq)
    startcpu = datetime.datetime.now()
    if model.run_contour(height_in=1) == 0:
        result_list = model.result_list
    print len(result_list)
    print datetime.datetime.now() - startcpu
    import pylab
    pylab.matshow(model.make_envelop(result_list, 5000 * 32 / 22.4))
    pylab.show()
    #plot_debug(result_list[14])
    #temp = []
    #for r in result_list:
    #    temp.append(r[512,540])
    #tempout = numpy.array(temp).astype(numpy.float32)
    #numpy.save('tt1', tempout)








