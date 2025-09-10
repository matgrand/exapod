# hexapod geometry and measurements
# x,y,z,R,rx,ry,rz
HT, HB = 327.916, 65.0 # height of the top and bottom plates [mm] 
A1T = (-50.506,-375.988,HT)
A1B = (-160.506,-375.956,HB)
A2T = (49.494,-376.017,HT)
A2B = (159.494,-376.048,HB)
B1T = (-300.559,231.388,HT)
B1B = (-245.53,326.635,HB)
B2T = (-350.584,144.8,HT)
B2B = (-405.611,49.553,HB)
C1T = (350.669,144.595,HT)
C1B = (405.641,49.317,HB)
C2T = (300.694,231.212,HT)
C2B = (245.721,326.491,HB)

D0 = 285.0 # nominal distance of the 6 arms at rest [mm] 

# disks -> [x,y,z,R,rx,ry,rz]
D1 = (0.426,-24.219,369.539,134.5,0.1658062789,0,0) # disk 1 -> top of the pipe
D2 = (0.69,46.717,-221,245,0,0,0) # disk 2 -> lower constraint disk
D3 = (0.69,46.717,-115.531,200,0,0,0) # disk 3 -> upper constraint disk