#coding=utf-8

import ConfigParser
import logging

def parse_engine():
    global GRID_INTERVAL, GRID_SIZE, TIMESTEP, OUTSTEP, DURATION, HALF_INTERVAL, HALF_SIZE
    GRID_INTERVAL=configparser.getint("engine", "grid_interval")
    GRID_SIZE=configparser.getint("engine", "grid_size")
    TIMESTEP=configparser.getint("engine", "timestep")
    OUTSTEP=configparser.getint("engine", "outstep")
    DURATION=configparser.getint("engine", "duration")
    HALF_INTERVAL=GRID_INTERVAL/2
    HALF_SIZE=GRID_SIZE/2

def parse_met():
    global METTEST, METFILE, MET_FORMAT
    METTEST=configparser.getint('met', 'test')
    METFILE=configparser.get('met', 'met_file')
    MET_FORMAT=configparser.getint('met', 'met_format')
    pass


def parse_source():
    global SRCTEST, SRCFILE
    SRCTEST=configparser.getint('src', 'test')
    SRCFILE=configparser.get('src', 'src_file')
    pass

def parse_reverse():
    global REVERSE,REVERSE_FILES,START_TIME,START_POS
    REVERSE=configparser.getint('reverse', 'reverse')
    REVERSE_FILES=str.split(configparser.get('reverse', 'reverse_files'), ',')
    #TODO: 这里很无赖的假定的开始时间是整数0，需要处理输入数据的哦~
    START_TIME=configparser.get('reverse', 'start_time')
    #TODO: 这里我们直接无视这里
    START_POS=configparser.get('reverse', 'start_pos')
    pass

def parse_ui():
    global MODE, CORE, LOGLEVEL, LOGFORMAT
    MODE=configparser.get("ui", "mode")
    CORE=configparser.get("ui", "core")
    LOGLEVEL=eval(configparser.get("ui", "log"))
    LOGFORMAT="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"

def expand_src(sourcelist):
    """根据ratelist和timelist生成一个source列表"""
    limit = 360
    result = list()
    if sourcelist[0] < 4: sourcelist[0] = 4
    ratelist = sourcelist[1:]
    timelist = [int(sourcelist[0] / len(ratelist))] * len(ratelist)
    mod = sourcelist[0] % len(ratelist)
    for i in range(mod):
        if not i % 2: #Even
            timelist[len(timelist) - i / 2 - 1] += 1
        else: #Odd
            timelist[(i - 1) / 2] += 1
    #if __debug__:
    #    print "[Time Expander]: timelist is", timelist
    for r, t in zip(ratelist, timelist):
        result += [r] * int(t)
    return result if len(result) <= limit else result[:limit]

class normalwind:
    a11=(0.901074,0.914370,0.919325,0.924279,0.926849,0.929418,0.925118,0.920818,0.929418)
    a12=(0.850934,0.865014,0.875086,0.885157,0.886940,0.888723,0.892794,0.896864,0.888723)
    r11=(0.425809,0.281846,0.2295,0.177154,0.143940,0.110726,0.0985631,0.0864001,0.0553634)
    r12=(0.602052,0.396353,0.314238,0.232123,0.189396,0.146669,0.124308,0.101947,0.0733348)
    a21=(1.12154,0.964435,0.941015,0.917595,0.838628,0.826212,0.776864,0.788370,0.78440)
    a22=(1.5136,0.964435,0.941015,0.917595,0.7569410,0.632023,0.572347,0.565188,0.525969)
    a23=(2.10881,1.09356,1.0077,0.917595,0.815575,0.55536,0.499149,0.414743,0.322659)
    r21=(0.0799904, 0.127190, 0.114682 ,0.106803,0.126152, 0.104634, 0.111771, 0.0927529, 0.0620765)
    r22=(0.00854771,0.127190,0.114682,0.106803,0.235667,0.400167,0.528992,0.433384,0.370015)
    r23=(0.000211545,0.0570251,0.0757182,0.106803,0.136659,0.810763,1.03810,1.73241,2.40691)

class smallwind:
    r011=(0.93, 0.76, 0., 0.55, 0., 0.47 , 0., 0.44, 0.44)
    r012=(0.76 ,0.56 ,0., 0.35, 0., 0.27, 0. ,0.24, 0.24)
    r021=(1.57, 0.47,0., 0.21, 0., 0.12, 0., 0.07, 0.05)
    r022=(1.57,0.47,0.,0.21 ,0., 0.12 ,0., 0.07 ,0.05)

class windprofile:
    rural = (0.07, 0.07, 0.10, 0.15, 0.35, 0.55)
    urban = (0.10, 0.15, 0.20, 0.25, 0.30, 0.30)    
    

configfile = "gapuff.conf"
configparser = ConfigParser.ConfigParser()
configparser.read(configfile)
parse_engine()
parse_source()
parse_met()
parse_ui()
parse_reverse()




