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
## Bundled WRF Processor are HARDCODED, please modify it if needed. "wrf_processor.py" is the only script file licensed under MIT License.
## Some comments are written in Chinese. Please ignore them. They will be translated into English in next release

import threading

import os

import math
import datetime

import numpy
import numexpr
numexpr.set_vml_accuracy_mode("fast")

import logging, sys

from smoke_def import smoke_def
from global_settings import *
CoreMode=str.strip(CORE).lower()
if CoreMode == 'gpu':
    try:
        import gpu_core
        from gpu_core import run_gpu_conc
    except Exception as ex:
        print ex
        cuda_disabled = True
    else:
        cuda_disabled = False
        if __debug__: print "We found a cuda card and pycuda initialized."
else:
    cuda_disabled = True


#建立坐标网格
HG = GRID_SIZE / 2
GridIdxY, GridIdxX = numpy.mgrid[HG-1:-HG-1:-1, -HG:HG]
CentralPositionMatrixX = numexpr.evaluate("GridIdxX*GRID_INTERVAL+0.5*GRID_INTERVAL")
CentralPositionMatrixY = numexpr.evaluate("GridIdxY*GRID_INTERVAL-0.5*GRID_INTERVAL")

class model_puff_core:

    def __init__(self, src, met):
        #Read arguments
        self.src = src
        self.met = met
        self.src_height = 2.0
        self.release = src.read_rate()
        #Basic settings
        self.smoke_list = []
        self.gpulog = self.initlog("GPUCore")
        self.cpulog = self.initlog("CPUCore")

    def initlog(self, name):
        logger = logging.getLogger(name)
        logger.setLevel(LOGLEVEL)
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(logging.Formatter(LOGFORMAT))
        logger.addHandler(ch)
        return logger

    def run_envelop(self, loc, height=1, background=False):
        if len(loc) == 0:
            raise Error("You must specify LOC values to calculate envelop")
        for l in loc:
            if l <= 0:
                raise Error("LOC value < 0?!")
        return self.run_core_contour(loc=loc, receipter_height=height, envelop=True)



    def run_core_contour(self, receipter_height=1, envelop=False, ticks=None, loc=[], force_no_debug=False):
        """
Three modes are defined in core_contour funtion:
1 contour mode [default] -> compute contours every tick
2 given time mode [set ticks parameter] -> only compute contours at given ticks
3 envelop mode [set envelop=True and give LOCs] -> override Mode 1 and 2, compute contours in Mode 1 and generate envelops based on given LOCs
        """

        #计算包络线覆盖区域
        make_envelop = lambda r, loc: numexpr("sum(where(r>loc,1,0),0)")
        #判断当前时刻是否需要计算浓度
        bOutput = lambda : ((not envelop) and (not ticks is None) and (tick in ticks)) or ((ticks is None or envelop) and (tick % OUTSTEP == 0))
        #是否使用GPU计算
        bUseGPU = not cuda_disabled
        self.isRunning = True
        result_list = []
        output_tick = []
        tick = 0
        #Total Estimation Time is DURATION seconds
        while tick * TIMESTEP < DURATION:
            #烟团释放过程未结束并且此刻烟团质量大于0则加入烟团列表
            if tick < len(self.release) and self.release[tick] > 0:
                self.smoke_list.append(smoke_def(self.release[tick], self.met, tick, pos=(0.0, 0.0, self.src_height)))
            #追踪当前每个烟团的位置，并且计算浓度场
            if filter(lambda x:not x.invalid, self.smoke_list):
                map(smoke_def.walk, self.smoke_list)
            if bOutput():
                empty_conc_matrix = numpy.zeros_like(CentralPositionMatrixX)
                gdiffc = []; gmass = []; gpos = [];
                for smoke in filter(lambda x:not x.invalid, self.smoke_list):
                    #if we use cpu, we compute each concentration field inside the loop
                    if not bUseGPU:
                        X, Y, Z = smoke.pos
                        mass = smoke.mass
                        #计算扩散系数
                        diffc = smoke.diffusion_coefficents(int(smoke.curr_met[3]), smoke.walkinglength, smoke.windspeed)
                        x, z = diffc; y = x
                        PI = math.pi
                        height = receipter_height
                        empty_conc_matrix = numexpr.evaluate("empty_conc_matrix + (mass/((2*PI)**1.5*x*y*z)*exp(-0.5*((CentralPositionMatrixX-X)/x)**2)*exp(-0.5*((CentralPositionMatrixY-Y)/y)**2)*(exp(-0.5*((Z-height)/z)**2) + exp(-0.5*((Z+height)/z)**2)))")
                    #otherwise, we just record each smoke and submit them to gpu once for all.
                    else:
                        gpos += list(smoke.pos)
                        #计算扩散系数
                        gdiffc += list(smoke.diffusion_coefficents(int(smoke.curr_met[3]), smoke.walkinglength, smoke.windspeed))
                        gmass.append(smoke.mass)
                #Submit job to kernel.
                if bUseGPU and len(gmass) > 0:
                    dest = run_gpu_conc(pos=numpy.array(gpos).astype(numpy.float32), diffc=numpy.array(gdiffc).astype(numpy.float32), mass=numpy.array(gmass).astype(numpy.float32),
                                        height=receipter_height, GRID_WIDTH=GRID_SIZE, gridw=GRID_INTERVAL, smoke_count=len(gmass))
                    empty_conc_matrix = dest.reshape((GRID_SIZE,GRID_SIZE), order='C')
                result_list.append(empty_conc_matrix)
                output_tick.append(tick)
                self.cpulog.debug("current time-tick is %d" % tick)
            tick += 1
        #Post process: return envelops or fields
        if envelop:
            envelop_list = []
            for l in loc:
                envelop_list.append(make_envelop(result_list, l))
            final_result = dict(zip(loc, envelop_list))
        else:
            final_result = dict(zip(output_tick, result_list))
        self.isRunning = False
        return final_result

    #计算指定点的浓度
    def run_point(self, points, ticks=None, force_no_debug=False, envelop=False):
        """This function calculate concenrations at ONE given point. Note that point is real position, NOT grid index."""
        bOutput = lambda : ((not envelop) and (not ticks is None) and (tick in ticks)) or ((ticks is None or envelop) and (tick % OUTSTEP == 0))
        tick = 0
        output_tick = []
        dest = dict(zip(points, tuple([dict() for i in range(len(points))])))
        while tick * TIMESTEP < DURATION:
            #判断烟团释放过程是否结束
            if tick < len(self.release):
                #表示此刻有无烟团释放
                if self.release[tick] > 0:
                    #加入烟团list
                    self.smoke_list.append(smoke_def(self.release[tick], self.met, tick, pos=(0.0, 0.0, self.src_height)))
            if bOutput():
                if __debug__ and (not force_no_debug):
                    print "current time-tick is %d" % tick
            map(smoke_def.walk, self.smoke_list)
            if bOutput():
                diffc = []; mass = []; stab = []; pos = [];
                #In point mode, we just record all smokes down here and calculated
                for smoke in filter(lambda x:not x.invalid, self.smoke_list):
                    pos += list(smoke.pos)
                    #计算扩散系数
                    diffc += list(smoke.diffusion_coefficents(int(smoke.curr_met[3]), smoke.walkinglength, smoke.windspeed))
                    mass.append(smoke.mass)
                for point in points:
                    if mass:
                        X,Y,Z = point
                        PI = math.pi
                        POSX, POSY, POSZ = numpy.array(pos[0::3]).astype(numpy.float32), numpy.array(pos[1::3]).astype(numpy.float32), numpy.array(pos[2::3]).astype(numpy.float32)
                        DIFFX, DIFFZ = numpy.array(diffc[0::2]).astype(numpy.float32), numpy.array(diffc[1::2]).astype(numpy.float32)
                        MASS = numpy.array(mass).astype(numpy.float32)
                        temp_conc = numexpr.evaluate("sum(MASS / ((2*PI)**1.5*DIFFX*DIFFX*DIFFZ) * exp(-0.5*((X-POSX)/DIFFX)**2) * exp(-0.5*((Y-POSY)/DIFFX)**2) * (exp(-0.5*((Z-POSZ)/DIFFZ)**2) + exp(-0.5*((Z+POSZ)/DIFFZ)**2)))")
                        dest[point][tick] = temp_conc
                    else:
                        dest[point][tick] = 0
            tick += 1
        return dest

    def writefield(self, result_dict, prefix):
        tempdict = dict()
        def convertkeys(k):
            tempdict[str(k)] = result_dict[k]
        map(convertkeys, list(result_dict))
        numpy.savez_compressed(prefix, **tempdict)

    def writepoint(self, result_dict, prefix="realvalue_"):
        i = 1
        for point in sorted(result_dict):
            fp = open(prefix + str(i) + '.txt', 'w')
            fp.write('%d,%d\n' % (int(point[0]), int(point[1])))
            fp.write('%f\n' % point[2])
            for tick in sorted(result_dict[point]):
                fp.write(str(tick) + ',' + str(result_dict[point][tick]) + '\n')
            fp.close()
            i += 1











