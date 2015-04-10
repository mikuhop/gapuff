#coding=utf8
import os, sys, logging
os.environ["CUDA_PROFILE"] = "1"
os.environ["CUDA_DEVICE"] = "0"

import global_settings
import multi_puff; from multi_puff import model_puff_core

import input_info

import copy
import datetime

from mpi4py import MPI

import numpy, math, random
#REMEMBER: one model instance for one monitor. Speed is not very important!

from monitor import monitor

class reverse_engine:    
    
    def __init__(self):
        self.source = []
        self.monitors = []
        #设定一些参数
        self.sigma = 60
        self.ignore_low_value = True
        self.default_step = (1, 1.0e6)
        self.end_step = (1, 1.0e3)
        self.mpilog = self.__initlog("MPI")
        self.samplerlog = self.__initlog("Sampler")
        
    def __initlog(self, name):
        logger = logging.getLogger(name)
        logger.setLevel(global_settings.LOGLEVEL)
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(logging.Formatter(global_settings.LOGFORMAT))
        logger.addHandler(ch)
        return logger

    def narrowdown(self, searchstep):
        if searchstep <= self.end_step:
            return (-1,-1)
        timestep = max(self.end_step[0], searchstep[0] - 1)
        valuestep = max(self.end_step[1], searchstep[1] / 10.0)
        return (timestep, valuestep)

    def read_file_input(self):
        #Read Inputs
        self.init_src = input_info.source_info(None,(0,0),datetime.datetime.now(),False)
        self.met = input_info.met_info(self.init_src, mode=global_settings.MET_FORMAT, dataset=global_settings.METFILE, test=bool(global_settings.METTEST)).get_met()
        #Read Reverse settings
        filepath = global_settings.REVERSE_FILES
        if len(filepath) == 0:
            raise Error("[Reverse Engine] ERROR: No input file")
        self.monitor_count = len(filepath)
        #Read all files, one file for one monitor
        maxvalue = -1; maxtick = None
        for f in filepath:
            fp = open(f)
            str_pos = fp.readline().strip().split(',')
            pos_x, pos_y = int(float(str_pos[0])), int(float(str_pos[1]))
            str_height = fp.readline().strip().split(',')
            height,sigma = float(str_height[0]), (0.0 if len(str_height) == 1 else float(str_height[1]))
            m = monitor((pos_x, pos_y, height))
            for line in fp.readlines():
                li = line.strip().split(',')
                ti, va = int(li[0]), float(li[1])
                if maxvalue < va:
                    maxvalue = va
                    maxtick = ti
                #If we open IGNORE MODE, all values less than 1.0e-5 will be filtered.
                if self.ignore_low_value and va <= 1.0e-5:
                    continue
                m.record[ti] = va
            fp.close()
            self.monitors.append(m)
            
    def prep_model(self, source, met):
        src = input_info.source_info(source, None, None, False)
        return multi_puff.model_puff_core(src, met)

    def run_model(self, source):
        model = self.prep_model(source, self.met)        
        points = map(lambda m: m.position, self.monitors)
        tick_sets = map(lambda m: set(list(m.record)), self.monitors)
        ticks = list(set.union(*tick_sets))
        result = model.run_point(points=points, ticks=ticks, force_no_debug=True)
        for m in self.monitors:
            m.simulate = result[m.position]
        return sum(map(monitor.targetfunc, self.monitors))


    def search_best(self, sample, position, step=(1, 1.0e6), searchrange=[], mpi=False):
        """给定sample，搜索position位置上的最佳值"""
        base_sample = copy.copy(sample); best = sample
        last_value = None; counter = 0; minvalue = None
        #Search all elements in given range (External MPI Routine)
        if not mpi:
            minvalue = None
            for searchitem in searchrange:
                base_sample[position] = searchitem
                source = global_settings.expand_src(base_sample)
                value = self.run_model(source)
                self.mpilog.info("[External MPI=%d]: sample=%s, count=%d, value=%g" % (mpi_rank, str(base_sample), counter, value))
                #TODO: 根据不同的分布函数的核，这里不一定是minvalue有可能是maxvalue
                if minvalue is None or minvalue > value:
                    minvalue = value
                    best = copy.copy(base_sample)
        #An unlimited search (Internal MPI Routine)
        else:
            #TODO: 4 is not fixed, should be the length of sample
            assert position != 0
            f = step[1] if position else step[0]
            base_sample[position] = mpi_rank * f if position else 4 + mpi_rank
            self.mpilog.debug("My Rank is %d" % mpi_rank)
            finished = False
            while not finished:
                if 1:
                    source1 = global_settings.expand_src(base_sample)
                    value = self.run_model(source1)
                    self.mpilog.info("[Internal MPI=%d]: sample=%s, count=%d, value=%g" % (mpi_rank, str(base_sample), counter, value))
                    if mpi_rank == 0:   #Master Node
                        valuelist = [(value, base_sample)]
                        valuelist += map(lambda r: mpi_comm.recv(source=r, tag=9), range(1, mpi_size))
                        #for rank in range(1, mpi_size):
                        #    valuelist += [mpi_comm.recv(source=rank, tag=9)]
                        self.mpilog.debug("[MPI Search], valuelist=%s" % str(valuelist))
                        for value_slave, base_sample_slave in valuelist:
                            if minvalue is None or minvalue > value_slave:
                                minvalue = value_slave
                                best = copy.copy(base_sample_slave)
                            counter = (counter+1) if minvalue < value_slave else 0
                            if counter > 40:
                               finished = True
                    else:   #Slave Node
                        mpi_comm.send((value, base_sample), dest=0, tag=9)
                    base_sample[position] += mpi_size * f
                    finished = mpi_comm.bcast(finished, root=0)
                    mpi_comm.barrier();
                    #if position == 0 and base_sample[position] >= 720:
                    #    break;
        return (best, minvalue)

    #search engine
    def gibbs_test(self, ref=[], initstep=(1,1.0e6), preset=360):
        u"""用Gibbs抽样直接来描述一下这个过程"""
        finished = False
        assert isinstance(ref, list)
        #先考虑一个比较简单的情况:把浓度分成登长的4段，产生一个随机序列
        sample = ref if ref else [preset, 1.0e6, 1.0e6, 1.0e6, 1.0e6]
        step = initstep
        #Search
        lastsample, lastvalue = ref, None
        count = 0
        while not finished and count < 10000:
            start = 0 if ref or count else 1
            for i in range(start,5):
                #Broadcast sample
                sample = mpi_comm.bcast(sample, root=0)
                #Because we use ALOHA first, so release won't last longer than 1 hour.
                #Always use external routing to search time [4,360]
                if i == 0:
                    srange = range(4,361,step[0])
                    sublist = srange[mpi_rank:len(srange):mpi_size]
                    #print "Node:", mpi_rank, "sublist=", sublist
                    minsample, minvalue = self.search_best(sample, i, step, sublist)
                    if mpi_rank == 0: #Master Node              
                        #收集所有值中的最小值               
                        self.samplerlog.debug("[Node %d]: Sample is %s" % (mpi_rank, str(sample)))
                        for rank in range(1, mpi_size):
                            rsample, rvalue = mpi_comm.recv(source=rank, tag=10)
                            if rvalue < minvalue:
                                minsample = rsample
                                minvalue = rvalue
                            sample, value = minsample, minvalue
                    else:    #Slave Node
                        mpi_comm.send((minsample, minvalue), dest=0, tag=10)
                else:
                    if not ref:
                        sample, value = self.search_best(sample, i, step, mpi=True)
                    else:   #USE MPI
                        srange = numpy.arange(max(0, sample[i]-40*step[1]), sample[i]+40*step[1], step[1]).tolist()
                        sublist = srange[mpi_rank:len(srange):mpi_size]
                        self.mpilog.debug("[Node %d] sublist=%s " % (mpi_rank, sublist))
                        minsample, minvalue = self.search_best(sample, i, step, sublist)
                        if mpi_rank == 0: #Master Node
                            #收集所有值中的最小值
                            self.samplerlog.debug("[Node %d]: Sample is %s" % (mpi_rank, str(sample)))
                            for rank in range(1, mpi_size):
                                rsample, rvalue = mpi_comm.recv(source=rank, tag=10)
                                if rvalue < minvalue:
                                    minsample = rsample
                                    minvalue = rvalue
                            sample, value = minsample, minvalue
                        else:    #Slave Node
                            mpi_comm.send((minsample, minvalue), dest=0, tag=10)
                if mpi_rank == 0:
                    self.samplerlog.warning("[Direct Test]: Step count %d, Position %d -> Sample=%s, value=%g" % (count, i, str(sample), value))
                mpi_comm.barrier()
                #在一般的情况下value是只要比较value就可以了，很少有sample不同但是value相同的情况，不过这里还是比较了sample
                if mpi_rank == 0 and i == 0:
                    if lastvalue == value:
                        if not ref:
                            finished = True
                        else:
                            step = self.narrowdown(step)
                            self.samplerlog.info("[Reverse Engine] Downscale search step =" + str(step))
                            if step == (-1, -1):
                                self.samplerlog.info("[Reverse Engine] Search Finished")
                                finished = True
                    lastsample, lastvalue = sample, value
                finished = mpi_comm.bcast(finished, root=0)
                if finished: break
            #the outter loop
            count += 1
        if ref:
            sample, value = mpi_comm.bcast((sample, value), root=0)
            return sample, value
        else:
            sample = mpi_comm.bcast(sample, root=0)
            return sample
        #if mpi_rank == 0:
        #    return (sample, value) if ref else sample
        #else:
        #    return "No returnings from slave nodes"

def init_mpi():
    global mpi_size, mpi_rank, mpi_comm
    mpi_comm = MPI.COMM_WORLD
    mpi_size = mpi_comm.size
    mpi_rank = mpi_comm.rank
    assert mpi_size > 1

