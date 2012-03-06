#coding=utf-8

#try:
#    import wingdbstub
#except:
#    pass

#try:
    #import psyco
    #psyco.full()
    #USEPSYCO = True
#except:
USEPSYCO = False

import ConfigParser

def parse_engine():
    global GRID_INTERVAL, GRID_SIZE, TIMESTEP, OUTSTEP, DURATION, HALF_SIZE
    GRID_INTERVAL=configparser.getint("engine", "grid_interval")
    GRID_SIZE=configparser.getint("engine", "grid_size")
    TIMESTEP=configparser.getint("engine", "timestep")
    OUTSTEP=configparser.getint("engine", "outstep")
    DURATION=configparser.getint("engine", "duration")
    HALF_SIZE=GRID_INTERVAL/2

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
    global MODE, CORE
    MODE=configparser.get("ui", "mode")
    CORE=configparser.get("ui", "core")


def expand_src(sourcelist):
    """根据ratelist和timelist生成一个source列表"""
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
    if len(result) > 720:
        return result[:720]
    else:
        return result


configfile = "gapuff.conf"
configparser = ConfigParser.ConfigParser()
configparser.read(configfile)
parse_engine()
parse_source()
parse_met()
parse_ui()
parse_reverse()




