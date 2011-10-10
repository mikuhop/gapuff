#coding=utf8

#Read wrfout the project into my input
#TODO: Considering writing a cuda function to search grid

import numpy
import netCDF4
import datetime

from pyproj import Proj
#project to utm=51N
p = Proj(proj='utm', zone=51, ellps='WGS84')

from global_settings import *

class met_wrf_source:

    def __init__(self, dataset_path = 'wrfout.ncf'):
        self.dataset = netCDF4.Dataset(dataset_path)
        #extract XLONG and XLAT
        self.xlong_orig = self.dataset.variables['XLONG'][0]
        self.xlat_orig = self.dataset.variables['XLAT'][0]
        #prepare projection target
        self.xlat = numpy.empty_like(self.xlat_orig)
        self.xlong = numpy.empty_like(self.xlong_orig)
        self.height, self.width = self.xlong_orig.shape
        self.called = False

    def generate_met_seq_in(self, minute):
        if not minute == 0:
            return [60 * (60 - minute), 60 * (120 - minute), 7200]
        else:
            return [3600, 7200]
        
    def generate_met_seq(self):
        if not self.called:
            raise Error("You must call generate_met first before you are calling generate_met_seq")
        return self.met_seq

    def generate_met_simple(self, center, time):
        u"""简单模式假设我们的网格是方的，WRF的网页也是方的，无视第三层网格倾斜带来的误差"""
        nc_u10 = self.dataset.variables['U10']
        nc_v10 = self.dataset.variables['V10']
        nc_t2 = self.dataset.variables['T2']
        nc_ust = self.dataset.variables['UST']
        nc_alt = self.dataset.variables['ALT']
        nc_hfx = self.dataset.variables['HFX']
        nc_lh = self.dataset.variables['LH']
        dataset = self.dataset
        #convert to UTC timestamp
        utc_time = time - datetime.timedelta(seconds=3600*8)
        #NOTE: WRF的时刻是从UTC的昨天0点到明天0点，所以啊当前时刻需要+24
        tick = utc_time.hour + 24
        minute = utc_time.minute

        for y in xrange(self.height):
            for x in xrange(self.width):
                self.xlong[y,x], self.xlat[y,x] = p(self.xlong_orig[y,x], self.xlat_orig[y,x])
        if __debug__ and center is None:
            start_pos = [self.xlong[0,0] + GRID_INTERVAL/2, self.xlat[65,0] - GRID_INTERVAL/2] #left-top corner
        else:
            start_pos = [center[0] - GRID_SIZE / 2 * GRID_INTERVAL + GRID_INTERVAL / 2, center[1] + GRID_SIZE / 2 * GRID_INTERVAL - GRID_INTERVAL / 2]
        #find left-top corner in WRF domain 3
        m, n = self.find_grid_index(start_pos, None, True)
        #我们需要特别的处理角上的四个网格，其它的网格直接填充
        #计算左上角第一个WRF网格内气象场子网格的数目
        x0, y0 = self.xlong[m,n], self.xlat[m,n]
        subcount_x, subcount_y = int((start_pos[0]-x0)/GRID_INTERVAL+13), int((start_pos[1]-y0)/GRID_INTERVAL+13)
        if subcount_x < 0:
            print "WARNING: The domain 3 are not large enough, offset_x is %d meters" % subcount_x * GRID_INTERVAL
            subcount_x = 0
        if subcount_y < 0:
            subcount_y = 0
            print "WARNING: The domain 3 are not large enough, offset_y is %d meters" % subcount_y * GRID_INTERVAL
        #做一个稍微大一点的网格，然后取出从subcount_x:subcount_x+GRID_SIZE,subcount_y:subcount_y+GRID_SIZE的子集
        base_count = 1000 / GRID_INTERVAL
        if minute == 0:
            base_met_matrix = numpy.empty((2, base_count * 42, base_count * 42, 4))
        else:
            base_met_matrix = numpy.empty((3, base_count * 42, base_count * 42, 4))
        lm = base_met_matrix.shape[0]
        for y in range(m-42,m):
            for x in range(n,n+42):
                for k in xrange(lm):
                    u10 = nc_u10[tick + k, y, x]
                    v10 = nc_v10[tick + k, y, x]
                    t2 = nc_t2[tick + k, y, x]
                    ust = nc_ust[tick + k, y, x]
                    alt = nc_alt[tick + k, 0, y, x]
                    hfx = nc_hfx[tick + k, y, x]
                    lh = nc_lh[tick + k, y, x]
                    stab = self.calc_stab(t2, ust, alt, hfx, lh)
                    assert stab == 6
                    y1 = y - (m - 42)
                    x1 = x - n
                    base_met_matrix[k,y1*base_count:(y1+1)*base_count,x1*base_count:(x1+1)*base_count,0].fill(u10)
                    base_met_matrix[k,y1*base_count:(y1+1)*base_count,x1*base_count:(x1+1)*base_count,1].fill(v10)
                    base_met_matrix[k,y1*base_count:(y1+1)*base_count,x1*base_count:(x1+1)*base_count,2].fill(0)
                    base_met_matrix[k,y1*base_count:(y1+1)*base_count,x1*base_count:(x1+1)*base_count,3].fill(stab)
        self.met_matrix = base_met_matrix[:,subcount_y:subcount_y+GRID_SIZE,subcount_x:subcount_x+GRID_SIZE,:]
        self.met_seq = self.generate_met_seq_in(minute)
        self.called = True
        return self.met_matrix

    def generate_met(self, center, time):
        
        NO_USE_CODE="""
        nc_u10 = self.dataset.variables['U10']
        nc_v10 = self.dataset.variables['V10']
        nc_t2 = self.dataset.variables['T2']
        nc_ust = self.dataset.variables['UST']
        nc_alt = self.dataset.variables['ALT']
        nc_hfx = self.dataset.variables['HFX']
        nc_lh = self.dataset.variables['LH']
        dataset = self.dataset
        #convert to UTC timestamp
        utc_time = time - datetime.timedelta(seconds=3600*8)
        #NOTE: WRF的时刻是从UTC的昨天0点到明天0点，所以啊当前时刻需要+24
        tick = utc_time.hour + 24
        minute = utc_time.minute
        if minute == 0:
            self.met_matrix = numpy.empty((2, GRID_SIZE, GRID_SIZE, 4))
        else:
            self.met_matrix = numpy.empty((3, GRID_SIZE, GRID_SIZE, 4))
        for y in xrange(self.height):
            for x in xrange(self.width):
                self.xlong[y,x], self.xlat[y,x] = p(self.xlong_orig[y,x], self.xlat_orig[y,x])
        if __debug__ and center is None:
            start_pos = [self.xlong[0,0] + GRID_INTERVAL/2, self.xlat[65,0] - GRID_INTERVAL/2] #left-top corner
        else:
            start_pos = [center[0] - GRID_SIZE / 2 * GRID_INTERVAL + GRID_INTERVAL / 2, center[1] + GRID_SIZE / 2 * GRID_INTERVAL - GRID_INTERVAL / 2]
        lm = self.met_matrix.shape[0]
        #scan the met-matrix to extract value.
        for i in xrange(GRID_SIZE):                 #row index -> y
            for j in xrange(GRID_SIZE):             #column index -> x
                m, n = self.find_grid_index([start_pos[0] + GRID_INTERVAL * j, start_pos[1] - GRID_INTERVAL * i], None, False)
                for k in xrange(lm):
                    u10 = nc_u10[tick + k, m, n]
                    v10 = nc_v10[tick + k, m, n]
                    t2 = nc_t2[tick + k, m, n]
                    ust = nc_ust[tick + k, m, n]
                    alt = nc_alt[tick + k, 0, m, n]
                    hfx = nc_hfx[tick + k, m, n]
                    lh = nc_lh[tick + k, m, n]
                    stab = self.calc_stab(t2, ust, alt, hfx, lh)
                    #assert stab == 6
                    grid = self.met_matrix[k,j,i]
                    grid[0] = u10; grid[1] = v10; grid[2] = 0; grid[3] = stab
            if __debug__:
                print "row count =", i
        self.met_seq = self.generate_met_seq(minute)
        return self.met_matrix, self.met_seq"""
        return self.generate_met_simple(center, time)

    def calc_stab(self, t2, ust, alt, hfx, lh):
        t = t2; u = ust; h = hfx - lh; p = 1.0 / alt; cp = 1.004; k = 0.4; g = 9.8
        #TODO: Check this formular, SEEMS wrong!
        L = -(t * u**3 * cp * p) / (k * g * h)
        if L < 0 and L >= -10.64:
            return 1
        elif L <= -10.64 and L > -23.07:
            return 2
        elif L <= -23.07 and L > -99.93:
            return 3
        elif L <= -99.93 or L > 152.52:
            return 4
        elif L <= 152.52 and L > 29.24:
            return 5
        elif L > 0 and L <= 29.24:
            return 6
        else:
            raise Error("Incorrect stability")
            
    def find_grid_index(self, pos, ref=None, full=False):
        """function that return nearest point"""
        dis = numpy.inf; m = -1; n = -1
        NO_USE_CODE="""
        if False:
            for y in xrange(self.height):
                for x in xrange(self.width):
                    dis1 = (self.xlong[y,x]-pos[0])**2 + (self.xlat[y,x]-pos[1])**2
                    if dis1 < dis:
                        dis = dis1
                        m = x
                        n = y
            return [n, m]"""
        #Quick Search
        for y in xrange(self.height):
            row_min = numpy.inf
            for x in xrange(self.width):
                dis1 = (self.xlong[y,x]-pos[0])**2 + (self.xlat[y,x]-pos[1])**2
                if dis1 < dis:
                    m = x; n = y;
                if dis1 < row_min:
                    row_min = dis1
                else:
                    break;
            if row_min < dis:
                dis = row_min;
            else:
                break
        return [n, m]

if __name__ == "__main__":
    width = GRID_SIZE; height = GRID_SIZE
    if __debug__:
        a = met_wrf_source()
        met = a.generate_met(None, datetime.datetime.now())
        seq = a.generate_met_seq()
        print met.shape
        print seq


