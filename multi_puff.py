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

## Here are some notes:
## Walking-length is a tuple (xy, xz). At beginning, xy equals xz until the stability changed so new walking-length
## is calculated.
## The position of smoke is a tuple (x, y, z). Remember x, y and z are all float numbers, not integers.
## Met-condition is a tuple (u,v,z,stab)
## Z-speed is ignored in this version of puff-model. Because of the diffusion coefficients didn't consider the z-wind
## Bundled WRF Processor are HARDCODED, please modify it if needed. "source_wrf.py" is the only script file licensed under MIT License.
## Some comments are written in Chinese. Please ignore them. They will be translated into English in next release

import threading

import os
os.environ["CUDA_PROFILE"] = "1"
os.environ["CUDA_DEVICE"] = "0"

import math
import datetime

import numpy
import numexpr
#Set up numexpr
numexpr.set_vml_accuracy_mode("fast")

try:
    import gpu_core
    from gpu_core import run_gpu_conc
    import gpu_core2
    from gpu_core2 import run_gpu_conc_point
except:
    cuda_disabled = True
    #if __debug__: print "We found no cuda card. We'll use cpu only."
else:
    cuda_disabled = False
    #if __debug__: print "We found a cuda card and pycuda initialized."

from smoke_def import smoke_def
from global_settings import *

#建立坐标网格
HG = HALF_SIZE
GridIdxY, GridIdxX = numpy.mgrid[HG-1:-HG-1:-1, -HG:HG]
CentralPositionMatrixX = numexpr.evaluate("GridIdxX*GRID_INTERVAL+0.5*GRID_INTERVAL")
CentralPositionMatrixY = numexpr.evaluate("GridIdxY*GRID_INTERVAL-0.5*GRID_INTERVAL")
CoreMode=str.strip(CORE).lower()

class model_puff_core:

    def __init__(self, release, met_field, met_seq):
        self.release = release
        self.met_field = met_field
        self.met_seq = met_seq
        self.smoke_list = []
        self.result_list = []
        self.recepter_height = 1.0

    def run_envelop(self, loc, height=1, background=False):
        if len(loc) == 0:
            raise Error("You must specify LOC values to calculate envelop")
        for l in loc:
            if l <= 0:
                raise Error("LOC value < 0?!")
        self.loc = loc
        if background:
            th = threading.Thread(target=self.run_core_contour, args=(height, True, None, [], False))
            th.start()
            return 0
        else:
            return self.run_core_contour(loc=loc, height_in=height, envelop=True)

    def run_core_time(self, time_list, height_in=1, background=False):
        """time_list is the list of REAL time, NOT ticks"""
        tick_list = []
        for t in time_list:
            tick_list.append(t // TIMESTEP)
        if background:
            th = threading.Thread(target=self.run_core_contour, args=(height_in, False, tick_list, [], False))
            th.start()
            return 0
        else:
            return self.run_core_contour(self, height_in=height_in, ticks=ticks)

    def run_core_contour(self, height_in=1, envelop=False, ticks=None, loc=[], force_no_debug=False):
        """
Three modes are defined in core_contour funtion:
1 contour mode [default] -> compute contours every tick
2 given time mode [set ticks parameter] -> only compute contours at given ticks
3 envelop mode [set envelop=True and give LOCs] -> override Mode 1 and 2, compute contours in Mode 1 and generate envelops based on given LOCs
        """
        self.isRunning = True
        self.output_tick = []
        tick = 0
        #Total Estimation Time is DURATION seconds
        while tick * TIMESTEP < DURATION:
            if ((not envelop) and (not ticks is None) and (tick in ticks)) or ((ticks is None or envelop) and (tick % OUTSTEP == 0)):
                if __debug__ and (not force_no_debug):
                    print "current time-tick is %d" % tick
            #判断烟团释放过程是否结束
            if tick < len(self.release):
                #表示此刻有无烟团释放
                if self.release[tick] > 0:
                    #加入烟团list
                    self.smoke_list.append(smoke_def(self.release[tick], self.met_field, self.met_seq, tick, pos=(0.0, 0.0, height_in)))
            #追踪当前每个烟团的位置，并且计算浓度场
            #CPU Func
            if cuda_disabled or CoreMode != 'gpu':
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
                        height = self.recepter_height
                        #if not USEPSYCO:
                        current_smoke_conc = numexpr.evaluate("mass/((2*PI)**1.5*x*y*z)*exp(-0.5*((CentralPositionMatrixX-X)/x)**2)*exp(-0.5*((CentralPositionMatrixY-Y)/y)**2)*(exp(-0.5*((Z-height)/z)**2) + exp(-0.5*((Z+height)/z)**2))")
                        empty_conc_matrix = numexpr.evaluate("empty_conc_matrix + current_smoke_conc")
                        #else:
                        #    current_smoke_conc = mass/((2*PI)**1.5*x*y*z)*numpy.exp(-0.5*((CentralPositionMatrixX-X)/x)**2)*numpy.exp(-0.5*((CentralPositionMatrixY-Y)/y)**2)*(exp(-0.5*((Z-height)/z)**2) + exp(-0.5*((Z+height)/z)**2))
                        #    empty_conc_matrix = empty_conc_matrix + current_smoke_conc
                    else:
                        tick += 1
                        continue
                self.result_list.append(empty_conc_matrix)
                self.output_tick.append(tick)
            #GPU Func
            else:
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
                                            height=self.recepter_height, GRID_WIDTH=GRID_SIZE, gridw=GRID_INTERVAL, smoke_count=len(mass))
                empty_conc_matrix = dest.reshape((GRID_SIZE,GRID_SIZE), order='C')
                self.result_list.append(empty_conc_matrix)
                self.output_tick.append(tick)
            #时钟+1
            tick += 1
            if envelop:
                self.envelop_list = []
                for l in loc:
                    self.envelop_list.append(self.make_envelop(self.result_list, l))
        self.isRunning = False
        return 0

    #计算包络线覆盖区域
    def make_envelop(self, result_list, loc):
        base = numpy.zeros_like(result_list[0])
        for r in result_list:
            base = numexpr.evaluate("base+where(r>loc, 1, 0)")
        return base

    #废弃的函数
    def run_gpu_contour(self, height_in=1, envelop=False, ticks=None, loc=[]):
        raise Error("Use run_core_contour instead!")

    #计算指定点的浓度
    def run_point(self, point=[10.0,10.0,1.0], height_in=1.0, ticks=None, force_no_debug=False, envelop=False):
        """
This function calculate concenrations at ONE given point. Note that point is real position, NOT grid index.
"envelop" will override "ticks" so will force computing regular ticks.
        """
        tick = 0
        self.output_tick = []
        self.given_point = point
        smoke_countlist = []; diffc = []; mass = []; pos = [];
        smoke_count = 0
        dest = []
        while tick * TIMESTEP < DURATION:
            #判断烟团释放过程是否结束
            if tick < len(self.release):
                #表示此刻有无烟团释放
                if self.release[tick] > 0:
                    #加入烟团list
                    self.smoke_list.append(smoke_def(self.release[tick], self.met_field, self.met_seq, tick, pos=(0.0, 0.0, height_in)))
            if ((not envelop) and (not ticks is None) and (tick in ticks)) or ((ticks is None or envelop) and (tick % OUTSTEP == 0)):
                if __debug__ and (not force_no_debug):
                    print "current time-tick is %d" % tick
                self.output_tick.append(tick)
            diffc = []; mass = []; stab = []; pos = [];
            for smoke in self.smoke_list:
                if not smoke.invalid:
                    smoke.walk()
                if ((not envelop) and (not ticks is None) and (tick in ticks)) or ((ticks is None or envelop) and (tick % OUTSTEP == 0)):
                    smoke_count += 1
                    pos += list(smoke.pos)
                    #计算扩散系数
                    smoke.diffusion_coefficents(int(smoke.met[3]), smoke.walkinglength, smoke.windspeed)
                    diffc += list(smoke.diffc)
                    mass.append(smoke.mass)
            if ((not envelop) and (not ticks is None) and (tick in ticks)) or ((ticks is None or envelop) and (tick % OUTSTEP == 0)):
                X,Y,Z = point
                PI = math.pi
                POSX, POSY, POSZ = numpy.array(pos[0::3]).astype(numpy.float32), numpy.array(pos[1::3]).astype(numpy.float32), numpy.array(pos[2::3]).astype(numpy.float32)
                DIFFX, DIFFZ = numpy.array(diffc[0::2]).astype(numpy.float32), numpy.array(diffc[1::2]).astype(numpy.float32)
                MASS = numpy.array(mass).astype(numpy.float32)
                #if not USEPSYCO:
                temp_conc = numexpr.evaluate("MASS / ((2*PI)**1.5*DIFFX*DIFFX*DIFFZ) * exp(-0.5*((X-POSX)/DIFFX)**2) * exp(-0.5*((Y-POSY)/DIFFX)**2) * (exp(-0.5*((Z-POSZ)/DIFFZ)**2) + exp(-0.5*((Z+POSZ)/DIFFZ)**2))")
                #else:
                #temp_conc = MASS / ((2*PI)**1.5*DIFFX*DIFFX*DIFFZ) * numpy.exp(-0.5*((X-POSX)/DIFFX)**2) * numpy.exp(-0.5*((Y-POSY)/DIFFX)**2) * (numpy.exp(-0.5*((Z-POSZ)/DIFFZ)**2) + numpy.exp(-0.5*((Z+POSZ)/DIFFZ)**2))
                dest.append(numpy.sum(temp_conc))
            tick += 1
        self.point_list = dest
        return 0

    def writematrix(self, result, prefix):
        for i in range(len(result)):
            name = prefix + str(i) + ".gz"
            dataset = result[i]
            numpy.savetxt(name, dataset)

    def writefile(self, name):
        fp = open(name, 'w')
        fp.write('%d,%d\n' % (int(self.given_point[0]), int(self.given_point[1])))
        fp.write('%f\n' % self.given_point[2])
        for i in range(len(self.point_list)):
            r = self.point_list[i]
            t = self.output_tick[i]
            fp.write(str(t) + ',' + str(r) + '\n')
        fp.close()











