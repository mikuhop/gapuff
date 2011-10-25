#coding=utf8

import numpy, math
from global_settings import *

class smoke_def:
    u"""一个烟团的定义"""

    def __init__(self, mass, metconditions, metsequence, inittime, pos=(0.0, 0.0, 10.0)):
        #Copy parameters
        self.mass = mass
        self.met_matrix = metconditions
        self.met_seq = self.expand_seq(metsequence)
        #Current position and Last-tick position. A smoke always starts from self.pos
        self.pos = list(pos)
        if self.pos[2] <= 0:
            self.pos[2] = 0.01
        self.last_pos = None
        #Walking length is critical parameter for diffusion factors
        self.walkinglength = [0.0, 0.0]
        #Current met conditions and Last-tick conditions
        self.met = self.extract_met_conditions(inittime, self.pos)
        self.last_met = None
        #tick count of smoke-self
        self.tick = inittime
        #mark smoke-self to be invalid
        self.invalid = False

    def expand_seq(self, metsequence):
        met_seq = []
        for tick in xrange(DURATION // TIMESTEP):
            for i in range(len(metsequence)):
                if tick * TIMESTEP < metsequence[i]:
                    break
            met_seq.append(i)
        return met_seq

    def extract_met_conditions(self, tick, pos):
        """Extract the met-conditions at given postion and time-tick"""
        try:
            x, y, z = pos
            GridIdxX = int(x // GRID_INTERVAL) + GRID_SIZE // 2
            GridIdxY = int(y // GRID_INTERVAL) + GRID_SIZE // 2
            virtualmet = self.met_matrix[self.met_seq[tick], GridIdxX, GridIdxY]
            stab = virtualmet[3]
            uspeed = virtualmet[0] * (z / 10.0) ** windprofile.urban[int(stab) - 1]
            vspeed = virtualmet[1] * (z / 10.0) ** windprofile.urban[int(stab) - 1]
            self.windspeed = math.sqrt(virtualmet[0] ** 2 + virtualmet[1] ** 2)
            return [uspeed, vspeed, 0, stab]
        except:
            return None

    def walk(self):
        u"""walk the smoke for timetick period, change pos, met and walking_length"""
        #TODO: track smoke position every-second instead of every TIMESTEP
        #FIXME: By tracking smoke every TIMESTEP seconds, it will import errors when across the boudary of grid with different-met-condition
        #NOTE: We ignored Z-direct wind to shift-up or press-down the smoke
        x,y,z = self.pos
        new_x = x + self.met[0] * TIMESTEP
        new_y = y + self.met[1] * TIMESTEP
        new_z = z
        self.last_pos = self.pos
        self.pos = (new_x, new_y, new_z)
        self.tick += 1
        self.last_met = self.met
        self.met = self.extract_met_conditions(self.tick, self.pos)
        if self.met == None:
            self.invalid = True
        else:
            #Increase walking length
            self.walkinglength[0] += math.sqrt((new_x - x) ** 2 + (new_y - y) ** 2)
            self.walkinglength[1] += math.sqrt((new_x - x) ** 2 + (new_y - y) ** 2)
            #Calculate a virtual source to correct the walking-length if the STAB changed
            if not self.met[3] == self.last_met[3]:
                print "STAB changed from %d to %d" % (self.last_met[3], self.met[3])
                current_diff_C = self.diffusion_coefficents(self.last_met[3], self.walkinglength, self.windspeed)
                self.walkinglength = self.reverse_walkinglength(self.met[3], current_diff_C, self.windspeed)

    def diffusion_coefficents(self, stab, walkinglength, speed):
        istab = (1,2,4,6,8,9)
        assert speed > 0
        dist_x, dist_z = walkinglength
        try:
            is1 = istab[stab - 1]
            is1 = is1 - 1
            if speed > 1.5:
                if dist_x > 1000:
                    sigy = normalwind.r12[is1]*dist_x**normalwind.a12[is1]
                else:
                    sigy = normalwind.r11[is1]*dist_x**normalwind.a11[is1]
                if is1 <= 4:
                    if dist_z > 500:
                        sigz = normalwind.r23[is1]*dist_z**normalwind.a23[is1]
                    elif dist_z <= 500 and dist_z > 300:
                        sigz = normalwind.r22[is1]*dist_z**normalwind.a22[is1]
                    else:
                        sigz = normalwind.r21[is1]*dist_z**normalwind.a21[is1]
                elif is1 == 5 or is1 == 7:
                    if dist_z > 10000:
                        sigz = normalwind.r23[is1]*dist_z**normalwind.a23[is1]
                    elif dist_z <= 10000 and dist_z > 2000:
                        sigz = normalwind.r22[is1]*dist_z**normalwind.a22[is1]
                    else:
                        sigz = normalwind.r21[is1]*dist_z**normalwind.a21[is1]
                else:
                    if dist_z > 10000:
                        sigz = normalwind.r23[is1]*dist_z**normalwind.a23[is1]
                    elif dist_z <= 10000 and dist_z > 1000:
                        sigz = normalwind.r22[is1]*dist_z**normalwind.a22[is1]
                    else:
                        sigz = normalwind.r21[is1]*dist_z**normalwind.a21[is1]
            elif speed < 0.5:
                sigy = smallwind.r011[is1] * dist_x / speed
                sigz = smallwind.r021[is1] * dist_z / speed
            else:
                sigy = smallwind.r012[is1] * dist_x / speed
                sigz = smallwind.r022[is1] * dist_z / speed
            self.diffc = [sigy, sigz]
        except:
            self.diffc = None
        return self.diffc

    def reverse_walkinglength(self, stab, DiffC, speed):
        sigy = DiffC[0]
        sigz = DiffC[2]
        istab = (1,2,4,6,8,9)
        try:
            is1 = istab[stab - 1]
            is1 = is1 - 1
            if speed > 1.5:
                dist_x = (sigy / normalwindr12[is1]) ** (1 / normalwind.a12[is1])
                if dist_x < 1000:
                    dist_x = (sigy / normalwindr11[is1]) ** (1 / normalwind.a11[is1])
                if is1 <= 4:
                    dist_z = (sigz / normalwind.r23[is1]) ** (1 / normalwind.a23[is1])
                    if dist_z < 500:
                        dist_z = (sigz / normalwind.r22[is1]) ** (1 / normalwind.a22[is1])
                    if dist_z < 300:
                        dist_z = (sigz / normalwind.r21[is1]) ** (1 / normalwind.a21[is1])
                elif is1 == 5 or is1 == 7:
                    dist_z = (sigz / normalwind.r23[is1]) ** (1 / normalwind.a23[is1])
                    if dist_z < 10000:
                        dist_z = (sigz / normalwind.r22[is1]) ** (1 / normalwind.a22[is1])
                    if dist_z < 2000:
                        dist_z = (sigz / normalwind.r21[is1]) ** (1 / normalwind.a21[is1])
                else:
                    dist_z = (sigz / normalwind.r23[is1]) ** (1 / normalwind.a23[is1])
                    if dist_z < 10000:
                        dist_z = (sigz / normalwind.r22[is1]) ** (1 / normalwind.a22[is1])
                    if dist_z < 1000:
                        dist_z = (sigz / normalwind.r21[is1]) ** (1 / normalwind.a21[is1])
            elif speed < 0.5:
                dist_x = sigy * speed / smallwind.r011[is1]
                dist_z = sigz * speed / smallwind.r021[is1]
            else:
                dist_x = sigy * speed / smallwind.r012[is1]
                dist_z = sigz * speed / smallwind.r022[is1]
            return [dist_x, dist_z]
        except:
            return None

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

#Global functions
def GetDiffusionFactors(stab, walk, speed):
    pass