#coding=utf8
import os, sys
os.environ["CUDA_PROFILE"] = "1"
os.environ["CUDA_DEVICE"] = "0"

import global_settings
import multi_puff; from multi_puff import model_puff_core

import source_main

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

    def narrowdown(self, searchstep):
        if searchstep <= self.end_step:
            return (-1,-1)
        timestep = max(self.end_step[0], searchstep[0] - 1)
        valuestep = max(self.end_step[1], searchstep[1] / 10.0)
        return (timestep, valuestep)

    def read_file_input(self):
        #Read Met
        testmet = source_main.MetPreProcessor(mode=global_settings.MET_FORMAT, simple=bool(global_settings.METTEST), dataset=global_settings.METFILE)
        MetField = testmet.GenerateField()
        #Read MetSeq
        MetSeq = testmet.Generate_MetSeq()
        #Read Reverse settings
        filepath = global_settings.REVERSE_FILES
        if len(filepath) == 0:
            raise Error("[Reverse Engine] ERROR: No input file")
        self.monitor_count = len(filepath)
        #Read all files, one file for one monitor
        maxvalue = -1; maxtick = None; self.init_tick = -1;
        for f in filepath:
            t = open(f)
            str_pos = t.readline().strip().split(',')
            pos_x, pos_y = int(float(str_pos[0])), int(float(str_pos[1]))
            str_height = t.readline().strip().split(',')
            height = float(str_height[0])
            m = monitor(pos_x, pos_y, height)
            for line in t.readlines():
                line1 = line.strip().split(',')
                ti, va = int(line1[0]), float(line1[1])
                if maxvalue < va:
                    maxvalue = va; maxtick = ti
                #If we open IGNORE MODE, all values less than 1.0e-5 will be filtered.
                if self.ignore_low_value and va <= 1.0e-5:
                    continue
                m.add_record(ti, va)
            self.init_tick
            t.close()
            m.set_met(MetField, MetSeq)
            self.monitors.append(m)

    def sum_error(self, source = None):
        """probility function"""
        sum = 0.0
        for m in self.monitors:
            m.set_source(source)
            m.init_model()
            m.run_model()
            sum += m.sum_error()
        #return math.exp(-sum/(2*self.sigma*self.sigma))
        #TODO: 修正这个分布函数的核，尽管在抽样上来说可以使用这个来代替
        return sum

    def search_best(self, sample, position, step=(1, 1.0e6), searchrange=[], mpi=False):
        """给定sample，搜索position位置上的最佳值"""
        base_sample = copy.copy(sample); best = sample
        last_value = None; counter = 0; minvalue = None
        #Search all elements in given range (External MPI Routine)
        if not mpi:
            try:
                minvalue = None
                for searchitem in searchrange:
                    base_sample[position] = searchitem
                    source = global_settings.expand_src(base_sample)
                    value = self.sum_error(source)
                    if __debug__: print "[Search %d]:" % mpi_rank, datetime.datetime.now(), base_sample, "count=%d" % counter, value
                    #TODO: 根据不同的分布函数的核，这里不一定是minvalue有可能是maxvalue
                    if minvalue is None or minvalue > value:
                        minvalue = value
                        best = copy.copy(base_sample)
            except:
                print sys.exc_info()
        #An unlimited search (Internal MPI Routine)
        else:
            if position == 0:
                f = step[0]
                base_sample[position] = 4 + mpi_rank
            else:
                f = step[1]
                base_sample[position] = mpi_rank * f
            #print "My Rank is", mpi_rank
            finished = False
            while not finished:
                if mpi_size == 1:
                    source = global_settings.expand_src(base_sample)
                    value = self.sum_error(source)
                    #if __debug__: print "[Single Search %d]:" % mpi_rank, datetime.datetime.now(), base_sample, "count=%d" % counter, value
                    if minvalue is None or minvalue > value:
                        minvalue = value
                        best = copy.copy(base_sample)
                    if minvalue <= value:
                        counter += 1
                        if counter > 40:
                            break;
                    else:
                        counter = 0
                    base_sample[position] += f
                    last_value = value
                else:
                    source1 = global_settings.expand_src(base_sample)
                    value = self.sum_error(source1)
                    if __debug__: print "[MPI Search %d]:" % mpi_rank, datetime.datetime.now(), base_sample, "count=%d" % counter, value
                    if mpi_rank == 0:   #Master Node
                        valuelist = [(value, base_sample)]
                        for rank in range(1, mpi_size):
                            valuelist += [mpi_comm.recv(source=rank, tag=9)]
                        #if __debug__: print "[MPI Search], valuelist=", valuelist
                        for value1, base_sample1 in valuelist:
                            if minvalue is None or minvalue > value1:
                                minvalue = value1
                                best = copy.copy(base_sample1)
                            if minvalue < value1:
                                counter += 1
                                if counter > 40:
                                    finished = True
                            else:
                                counter = 0
                            last_value = value1
                    else:               #Slave Node
                        mpi_comm.send((value, base_sample), dest=0, tag=9)
                    base_sample[position] += mpi_size * f
                    finished = mpi_comm.bcast(finished, root=0)
                    mpi_comm.barrier();
                    if position == 0 and base_sample[position] >= 720:
                        break;
        return (best, minvalue)

    #search engine
    def gibbs_test(self, ref=[], initstep=(1,1.0e6)):
        u"""用Gibbs抽样直接来描述一下这个过程"""
        finished = False
        if not isinstance(ref, list):
            raise Error('[Search Engine]: Reference must be a list')
        if len(ref) == 0:
            #先考虑一个比较简单的情况:把浓度分成登长的4段，产生一个随机序列
            sample = [360, 1.0e6, 1.0e6, 1.0e6, 1.0e6]
        else:
            sample = ref
        step = initstep
        #Search
        lastsample, lastvalue = ref, None
        for count in range(10000):
            if len(ref) == 0 and count == 0:
                start = 1
            else:
                start = 0
            for i in range(start,5):
                sample = mpi_comm.bcast(sample, root=0)
                #step = mpi_comm.bcast(step, root=0)
                if len(ref) == 0 or i == 0:
                    sample, value = self.search_best(sample, i, step, mpi=True)
                else:
                    srange = []
                    istart = max(0, sample[i] - 40.0*step[1])
                    iend = sample[i] + 40.0*step[1]
                    assert iend >= istart
                    while istart < iend:
                        srange += [istart]
                        istart += step[1]
                    srange += [iend]
                    #if __debug__: print "[Node %d]: srange is %s" % (mpi_rank, str(srange))
                    if mpi_size == 1: #NO MPI
                        sample, value = self.search_best(sample, i, step, srange)
                    else:             #USE MPI
                        #Broadcast sample
                        if __debug__ and mpi_rank == 0: print "[Node %d]: Sample is %s" % (mpi_rank, str(sample))
                        if mpi_rank == 0: #Master Node
                            sublist = dict()
                            subsize = len(srange) // mpi_size
                            for rank in range(1, mpi_size):
                                #mpi_comm.send(srange[rank*subsize:(rank+1)*subsize],dest=rank,tag=11)
                                sublist[rank] = srange[rank*subsize:(rank+1)*subsize]
                            #把剩余的数据都放到最后一个节点上
                            #mpi_comm.send(srange[(mpi_size-1)*subsize:], dest=mpi_size-1, tag=11)
                            rank1 = 1
                            for data in srange[mpi_size*subsize:]:
                                sublist[rank1] += [data]
                                rank1 += 1
                            for rank in range(1,mpi_size):
                                mpi_comm.send(sublist[rank], dest=rank, tag=11)
                            #计算属于自己节点的工作
                            if __debug__: print "[Node %d] search range is %s" % (mpi_rank, srange[0:subsize])
                            minsample, minvalue = self.search_best(sample, i, step, srange[0:subsize])
                            #收集所有值中的最小值
                            for rank in range(1, mpi_size):
                                rsample, rvalue = mpi_comm.recv(source=rank, tag=10)
                                if rvalue < minvalue:
                                    minsample = rsample
                                    minvalue = rvalue
                            sample, value = minsample, minvalue
                        else:           #Slave Node
                            sub_srange = mpi_comm.recv(source=0, tag=11)
                            #if __debug__: print "[Node %d] search range is %s" % (mpi_rank, sub_srange)
                            #if __debug__: print "[Node %d] step = %s" % (mpi_rank, step)
                            mpi_comm.send(self.search_best(sample, i, step, sub_srange), dest=0, tag=10)
                if mpi_rank == 0: print "[Direct Test]: Step count %d, Position %d -> Sample %s" % (count, i, str(sample))
                mpi_comm.barrier()
                #在一般的情况下value是只要比较value就可以了，很少有sample不同但是value相同的情况，不过这里还是比较了sample
                if mpi_rank == 0:
                    if lastsample == sample and i == 0:
                        if len(ref) == 0: finished = True
                        else:
                            step = self.narrowdown(step)
                            print "[Reverse Engine] Downscale search step =", step
                            if step == (-1, -1):
                                print "[Reverse Engine] Search Finished"
                                finished = True
                    if i == 0: lastsample, lastvalue = sample, value
                finished = mpi_comm.bcast(finished, root=0)
                if finished: break
            #the outter loop
            if finished: break
        if len(ref) == 0:
            return sample
        else:
            return sample, value

def init_mpi():
    global mpi_size, mpi_rank, mpi_comm
    mpi_comm = MPI.COMM_WORLD
    mpi_size = mpi_comm.size
    mpi_rank = mpi_comm.rank

#=====================================================#
def run_reverse():
    """Main Program"""
    init_mpi()
    if global_settings.REVERSE > 0:
        r = reverse_engine()
        r.read_file_input()
        #TODO: Step 1: 先通过无限制搜索搜索出一个参考量和参考步长

        #Step 2: 通过参考量和参考步长搜进一步向内搜索
        r.gibbs_test(ref=[18.0, 110000000.0, 61000000.0, 362000000.0, 77000000.0], initstep=(1,1.0e6))

if __name__ == "__main__":
    run_reverse()

















