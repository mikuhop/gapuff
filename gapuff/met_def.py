#coding=utf-8
import numpy
import global_settings

class met_def:
    """We define meterological data here. And we can define meterological sequence and field here.
Met的数据格式有三种
0 恒定气象场，必须是一个tuple并且长度是(u,v,z,stab)
1 SAM站点数据，必须是一个list，并且list中的每个tuple都是(u,v,z,stab), then convert it to a numpy array (t_index, value_index)
2 气象场，必须是numpy的ndarray(t_index,x_index,y_index,value_index)
"""

    def __init__(self, mode, data, seq):
        self.mode = mode
        self.seq = seq
        if self.mode == 0 and isinstance(data, tuple) and len(data) == 4:
            self.data = data
        elif self.mode == 1 and isinstance(data, list) and len(data) == len(seq):
            self.data = numpy.array(data, numpy.float32)
        elif self.mode == 2 and isinstance(data, numpy.ndarray) and data.shape[0] == len(seq):
            self.data = data
        else:
            raise Exception("Invalid input")

    def extract(self, tick, position):
        x, y, z = position
        if self.mode == 0:
            if abs(x) > global_settings.HALF_SIZE * global_settings.GRID_INTERVAL or abs(y) > global_settings.HALF_SIZE * global_settings.GRID_INTERVAL:
                return None
            else:
                return self.data
        if self.mode == 1:
            raise NotImplementedError("Not Implemented")
        if self.mode == 2:
            try:
                GridIdxX = int(x // global_settings.GRID_INTERVAL) + global_settings.HALF_SIZE
                GridIdxY = int(y // global_settings.GRID_INTERVAL) + global_settings.HALF_SIZE
                timeindex = sum(t < TIMESTEP * tick for t in self.seq)
                virtualmet = self.data[timeindex, GridIdxX, GridIdxY]
                stab = virtualmet[3]
                uspeed = virtualmet[0] * (z / 10.0) ** global_settings.windprofile.urban[int(stab) - 1]
                vspeed = virtualmet[1] * (z / 10.0) ** global_settings.windprofile.urban[int(stab) - 1]
                #self.windspeed = math.sqrt(virtualmet[0] ** 2 + virtualmet[1] ** 2)
                return [uspeed, vspeed, 0, stab]
            except:
                return None