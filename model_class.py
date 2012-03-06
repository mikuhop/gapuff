#coding=utf8
import os
os.environ["CUDA_PROFILE"] = "1"
os.environ["CUDA_DEVICE"] = "0"

import multi_puff
from multi_puff import model_puff_core

import source_main
import global_settings

def plot(result_list):
    import pylab
    #pylab.matshow(model.make_envelop(result_list, 5000 * 32 / 22.4))
    pylab.matshow(result_list[14])
    pylab.colorbar()
    pylab.show()
    #plot_debug(result_list[14])


def plot_debug(result):
    import pylab
    from pylab import contour, grid, show
    contours = (70.9/22.4, 70.9/22.4*3, 70.9/22.4*20)
    contour(result, contours)
    grid(True)
    show()

if __debug__:
    print __name__

def runselftest():
    testmet = source_main.MetPreProcessor(mode=global_settings.MET_FORMAT, simple=bool(global_settings.METTEST), dataset=global_settings.METFILE)
    testsrc = source_main.SourcePreProcessor()

    print "Reading a random source data"
    ReleaseQ = testsrc.Generate_TestSrc()
    print "Reading met data"
    MetField = testmet.GenerateField()
    print "Reading met sequence"
    MetSeq = testmet.Generate_MetSeq()
    ## 这里是设置案例编号的地方
    print datetime.now()
    case = 2
    #按照一个测点一个model的原则，如果有两个测点，我们就运行两个模型的实例
    if case == 0:
        model = model_puff_core(ReleaseQ, MetField, MetSeq)
        model.run_point(point=[1140.0,20.0,1.0], force_no_debug=True)
        result = model.point_list
    elif case == 1:
        model = model_puff_core(ReleaseQ, MetField, MetSeq)
        model.run_core_contour(2, False, None, [], False)
        result = model.result_list
        plot(result)
    elif case == 2:
        model = model_puff_core(ReleaseQ, MetField, MetSeq)
        model.run_point(point=[2320,0.0,1.0], force_no_debug=True)
        model.writefile("tt1.txt")
        model1 = model_puff_core(ReleaseQ, MetField, MetSeq)
        model1.run_point(point=[1000,0.0,1.0], force_no_debug=True)
        model1.writefile("tt2.txt")
        if __debug__: print ReleaseQ
    print datetime.now()
    print "Program End Normally"

if __name__ == "__main__":
    from datetime import datetime
    runselftest()



