#coding=utf8

#This file is used for test the script
import sys
if len(sys.argv) == 1:
    preset = 360
else:
    preset = int(sys.argv[1]) % 360 + 1

print "preset=%d" % preset

import reverse_engine

reverse_engine.init_mpi()
r = reverse_engine.reverse_engine()
r.read_file_input()
r.end_step=(1, 1.0e2)
ref = r.gibbs_test(preset=preset)
#ref = [360, 100e6, 50e6, 20e6, 20e6]
#ref = [484, 203000000.0, 18000000.0, 15000000.0, 6000000.0]
print ref
#r.gibbs_test(ref=ref, initstep=(1,1.0e5))


















