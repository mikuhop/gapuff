#coding=utf8

import numpy, math
from global_settings import *

class smoke_def:
    u"""一个烟团的定义"""

    def __init__(self, mass, met, inittime, pos=(0.0, 0.0, 10.0)):
        #Copy parameters
        self.mass = mass
        self.met = met
        #Current position and Last-tick position. A smoke always starts from self.pos
        self.pos = list(pos)
        if self.pos[2] <= 0:
            self.pos[2] = 0.01
        self.last_pos = None
        #Walking length is critical parameter for diffusion factors
        self.walkinglength = [0.0, 0.0]
        #Current met conditions and Last-tick conditions
        self.curr_met = self.met.extract(inittime, self.pos)
        self.windspeed = math.sqrt(self.curr_met[0] ** 2 + self.curr_met[1] ** 2)
        self.last_met = None
        #tick count of smoke-self
        self.tick = inittime
        #mark smoke-self as valid
        self.invalid = False

    def walk(self):
        u"""walk the smoke for timetick period, change pos, met and walking_length"""
        #TODO: track smoke position every-second instead of every TIMESTEP
        #FIXME: By tracking smoke every TIMESTEP seconds, it will import errors when across the boudary of grid with different-met-condition
        #NOTE: We ignored Z-direct wind to shift-up or press-down the smoke
        if self.invalid:
            return
        x,y,z = self.pos
        new_x = x + self.curr_met[0] * TIMESTEP
        new_y = y + self.curr_met[1] * TIMESTEP
        new_z = z
        self.last_pos = self.pos
        self.pos = (new_x, new_y, new_z)
        self.tick += 1
        self.last_met = self.curr_met
        self.curr_met = self.met.extract(self.tick, self.pos)
        if self.curr_met == None:
            self.invalid = True
        else:
            #Get current windspeed
            self.windspeed = math.sqrt(self.curr_met[0] ** 2 + self.curr_met[1] ** 2)
            #Increase walking length
            self.walkinglength[0] += math.sqrt((new_x - x) ** 2 + (new_y - y) ** 2)
            self.walkinglength[1] += math.sqrt((new_x - x) ** 2 + (new_y - y) ** 2)
            #Calculate a virtual source to correct the walking-length if the STAB changed
            if not self.curr_met[3] == self.last_met[3]:
                assert self.met.type > 0                
                #TODO: Log this -> print "STAB changed from %d to %d" % (self.last_met[3], self.met[3])
                current_diff_C = self.diffusion_coefficents(int(self.last_met[3]), self.walkinglength, self.windspeed)
                self.walkinglength = self.reverse_walkinglength(int(self.curr_met[3]), current_diff_C, self.windspeed)

    def diffusion_coefficents(self, stab, walkinglength, speed):
        istab = (1,2,4,6,8,9)
        assert speed > 0
        dist_x, dist_z = walkinglength
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
        return self.diffc

    def reverse_walkinglength(self, stab, DiffC, speed):
        sigy = DiffC[0]
        sigz = DiffC[1]
        istab = (1,2,4,6,8,9)
        is1 = istab[stab - 1]
        is1 = is1 - 1
        if speed > 1.5:
            dist_x = (sigy / normalwind.r12[is1]) ** (1 / normalwind.a12[is1])
            if dist_x < 1000:
                dist_x = (sigy / normalwind.r11[is1]) ** (1 / normalwind.a11[is1])
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


