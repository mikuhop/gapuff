#coding=utf-8

# 假设网格的宽度是总宽度大概是左右各20KM的范围
# Profile HIGH
#GRID_INTERVAL = 10
#GRID_SIZE = 2048*2
#TIMESTEP = 10

# Profile MIDDLE
#GRID_INTERVAL = 20
#GRID_SIZE = 1024*2
#TIMESTEP = 10


# Profile LOW
GRID_INTERVAL = 40
GRID_SIZE = 512*2
TIMESTEP = 10

#Set output ticks interval. Here is 2min, becuase it equals TIMESTEP * OUTSTEP = 120s = 2min
OUTSTEP = 12
#Total Simulation Duration
DURATION = 7200
HALF_SIZE = GRID_SIZE/2



