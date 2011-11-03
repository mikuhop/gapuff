#coding=utf8
import os
os.environ["CUDA_PROFILE"] = "1"
os.environ["CUDA_DEVICE"] = "0"

import multi_puff
from multi_puff import model_puff_core

import source_main

import datetime
import time
import math

def plot(result_list):
    import pylab
    pylab.matshow(model.make_envelop(result_list, 5000 * 32 / 22.4))
    pylab.show()
    plot_debug(result_list[14])

def writefile(name='tt1.txt', x=1140, y=20, h=1.0, result_list=[]):
    tt1 = open(name, 'w')
    tt1.write('%d,%d\n' % (int(x), int(y)))
    tt1.write('%s\n' % str(h))
    for i in range(len(result_list)):
        r = result_list[i]
        t = model.output_tick[i]
        tt1.write(str(t) + ',' + str(r) + '\n')
    tt1.close()

testmet = source_main.MetPreProcessor(mode=0, simple=True, dataset='wrfout.ncf')
testsrc = source_main.SourcePreProcessor()

print "Reading a random source data"
ReleaseQ = testsrc.GenerateVariedSource()
print "Reading met data"
MetField = testmet.GenerateTestField()
print "Reading met sequence"
MetSeq = testmet.GenerateTestFieldSeq()
#按照一个测点一个model的原则，如果有两个测点，我们就运行两个模型的实例
model = model_puff_core(ReleaseQ, MetField, MetSeq)
model.run_point(point=[1140.0,20.0,1.0], force_no_debug=True)
model1 = model_puff_core(ReleaseQ, MetField, MetSeq)
model1.run_point(point=[2400.0,200.0,1.0], force_no_debug=True)
#print ReleaseQ
writefile('tt1.txt', 1140, 20, 1.0, model.point_list)
writefile('tt2.txt', 2400, 200, 1.0, model1.point_list)
print "Program End Normally"




