#coding=utf8
import os
os.environ["CUDA_PROFILE"] = "1"
os.environ["CUDA_DEVICE"] = "0"

import hotshot
import hotshot.stats

import sys

import multi_puff
from multi_puff import model_puff_core

import input_info
import global_settings

def plot_debug(result):
    import pylab
    from pylab import contour, grid, show
    contours = (70.9/22.4, 70.9/22.4*3, 70.9/22.4*20)
    contour(result, contours)
    grid(True)
    show()

def runselftest(case=0):
    testsrc = input_info.source_info(test=True)
    testmet = input_info.met_info(testsrc, mode=global_settings.MET_FORMAT, dataset=global_settings.METFILE, test=bool(global_settings.METTEST))

    print "Reading source data"
    ReleaseQ = testsrc.read_rate()
    ReleaseQ1 = global_settings.expand_src([90, 141718800.0, 140853300.0, 136447200.0, 130601400.0])
    print "Reading met data"
    met = testmet.get_met()
    ## 这里是设置案例编号的地方
    if __debug__: print "Case Num: %d" % case
    starttime = datetime.now()
    logger.info("starttime=%s" % str(datetime.now()))
    if case == 0:
        model = model_puff_core(testsrc, met)
        model.run_point(points=([1140.0,20.0,1.0]), force_no_debug=True)
        result = model.point_list
    elif case == 1:
        model = model_puff_core(testsrc, met)
        result = model.run_core_contour(1, False, None, [], False)
        model.writefield(result, prefix="conc_field")
    elif case == 2:
        model = model_puff_core(testsrc, met)
        points = [(500,20.0,1.0),(1000,50.0,1.0),(2000,150.0,1.)]
        result = model.run_point(points=points, force_no_debug=True)
        model.writepoint(result)
    elif case == 3:
        model = model_puff_corea(testsrc, met)
        result = model.run_point(points=[(500,20.0,1.0)], force_no_debug=True)
        model1 = model_puff_core(ReleaseQ1, MetField, MetSeq)
        result1 = model1.run_point(points=[(500,20.0,1.0)], force_no_debug=True)
        from pylab import plot, xlabel, ylabel, grid, show, Rectangle, legend
        x = result.values()[0].keys()
        y1 = result.values()[0].values()
        y2 = result1.values()[0].values()
        plot(x,y1,'go-')
        plot(x,y2,'bo--')
        xlabel(u"时间刻度(10s)")
        ylabel(u"浓度(mg/m3)")
        p1 = Rectangle((0, 0), 1, 1, fc="g")
        p2 = Rectangle((0, 0), 1, 1, fc="b")
        legend([p1, p2], [u"真实源", u"反算源"])
        show()
    logger.info("endtime=%s" % str(datetime.now()))
    print "duration=%s" % str(datetime.now() - starttime)
    print "Program End Normally"

import logging
logger = logging.getLogger('Main')
logger.setLevel(global_settings.LOGLEVEL)
ch = logging.StreamHandler(sys.stderr)
ch.setFormatter(logging.Formatter(global_settings.LOGFORMAT))
logger.addHandler(ch)


if __name__ == "__main__":
    from datetime import datetime
    runselftest(2)
    #prof = hotshot.Profile("hs_prof.txt", 1)
    #prof.runcall(runselftest, 2)
    #prof.close()
    #p = hotshot.stats.load("hs_prof.txt")
    #p.print_stats()
    #if len(sys.argv) == 1:
    #    runselftest(2)
    #else:
    #    runselftest(int(sys.argv[1]))




