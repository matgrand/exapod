from numpy import array as arr, pi as π, cos, sin, ndarray as V, linspace
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np

R = 0.05 # [m] radius of the 2 discs
D = 0.01 # [m] distance between the pairs of ball bearings
H0 = 0.08 # [m] height of the column at "rest"

THREAD_STEP = 0.001 # [m]/[turn] thread pitch of the rods (1mm/turn)

# rest positions and orientations of the bottom and top discs
X0, Y0, Z0, RX0, RY0, RZ0, = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 
X1, Y1, Z1, RX1, RY1, RZ1 = 0.0, 0.0, H0, 0.0, 0.0, π/3

def rototranslation(v:V, dx=0.0, dy=0.0, dz=0.0, rx=0.0, ry=0.0, rz=0.0) -> V:
    """Rotate vector(s) v by angles rx, ry, rz and translate by dx, dy, dz"""
    assert v.ndim==2 and v.shape[1] == 3, f"v must be a Nx3 array, not {v.shape}"
    Rx = arr([[1, 0, 0], [0, cos(rx), -sin(rx)], [0, sin(rx), cos(rx)]])
    Ry = arr([[cos(ry), 0, sin(ry)], [0, 1, 0], [-sin(ry), 0, cos(ry)]])
    Rz = arr([[cos(rz), -sin(rz), 0], [sin(rz), cos(rz), 0], [0, 0, 1]])
    return (Rx @ Ry @ Rz @ v.T + arr([[dx, dy, dz]]).T).T

def exapod(dx, dy, dz, drx, dry, drz, x1=X1, y1=Y1, z1=Z1, rx1=RX1, ry1=RY1, rz1=RZ1):
    αp = D/R #angle between the 2 ball bearings of a pair
    N = 60 # number of points to draw the disc
    NC = 10 # number of colors
    ## rest positions
    α_balls = [0-αp, 0+αp, 2*π/3-αp, 2*π/3+αp, 4*π/3-αp, 4*π/3+αp] # balls angles
    balls = arr([[R*cos(α), R*sin(α), 0.0] for α in α_balls]) # reference position of the balls
    pnames = arr([[1.3*R*cos(α), 1.3*R*sin(α), 0.0] for α in α_balls]) # reference position of the names
    disc = arr([[R*cos(α), R*sin(α), 0.0] for α in linspace(0, 2*π, N)]) # reference position of the disc
    names = ['A','B','C','D','E','F'] # names of the balls/rods
    
    disc0, balls0, pnames0 = [np.roll(rototranslation(v,X0,Y0,Z0,RX0,RY0,RZ0),-1,axis=0) for v in [disc, balls, pnames]] # bottom disc at rest
    disc1, balls1, pnames1 = [rototranslation(v,x1,y1,z1,RX1,RY1,RZ1) for v in [disc, balls, pnames]] # top disc at rest
    disc2, balls2, pnames2 = [rototranslation(v, x1+dx, y1+dy, z1+dz, rx1+drx, ry1+dry, rz1+drz) for v in [disc, balls, pnames]] # top disc moved


    title = f"Exapod: \n[ x1:  {x1:+.4f}, y1:  {y1:+.4f}, z1:  {z1:+.4f} ]\n" + \
            f"[ rx1: {rx1:+.4f}, ry1: {ry1:+.4f}, rz1: {rz1:+.4f} ] \n" + \
            f"[ dx:  {dx:+.4f}, dy:  {dy:+.4f}, dz:  {dz:+.4f} ]\n" +\
            f"[ drx: {drx:+.4f}, dry: {dry:+.4f}, drz: {drz:+.4f} ]" 
    print(title)
    print()

    # rods lengths
    lengths1 = np.linalg.norm(balls1-balls0, axis=1) # [mm]
    lengths2 = np.linalg.norm(balls2-balls0, axis=1) # [mm]
    for i in range(6): # print stats for each rod
        δ = lengths2[i]-lengths1[i]
        print(f"rod {names[i]}: [{lengths1[i]:.4f} -> {lengths2[i]:.4f}] [Δ{names[i]} = {δ:+.4f}] [turns_{names[i]} = {δ/THREAD_STEP:+.1f}]")

    ## plot
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    #define colormap 
    cmap = plt.get_cmap('plasma')
    c0, c1 = 0.0, 0.7
    cols = cmap(np.linspace(c1, c0, NC)) # colors
    α = 0.3

    ax.scatter(balls0[:,0], balls0[:,1], balls0[:,2], color=cmap(c0), s=100) # bottom disc
    ax.scatter(balls1[:,0], balls1[:,1], balls1[:,2], color=cmap(c1), s=20) # top disc rest
    ax.scatter(balls2[:,0], balls2[:,1], balls2[:,2], color=cmap(c0), s=100) # top disc moved 
    ax.plot(disc0[:,0], disc0[:,1], disc0[:,2], color=cmap(c0), linestyle='-') # bottom disc
    ax.plot(disc1[:,0], disc1[:,1], disc1[:,2], color=cmap(c1), linestyle='--') # top disc rest
    ax.plot(disc2[:,0], disc2[:,1], disc2[:,2], color=cmap(c0), linestyle='-') # top disc moved
    for i in range(0, 6): # rods and names
        ax.plot([balls0[i,0], balls1[i,0]], [balls0[i,1], balls1[i,1]], [balls0[i,2], balls1[i,2]], color=cmap(c1), linestyle='--')
        ax.plot([balls0[i,0], balls2[i,0]], [balls0[i,1], balls2[i,1]], [balls0[i,2], balls2[i,2]], color=cmap(c0), linestyle='-', linewidth=2)
        ax.text(pnames0[i,0], pnames0[i,1], pnames0[i,2], names[i], color=cmap(c0))
        ax.text(pnames1[i,0], pnames1[i,1], pnames1[i,2], names[i], color=cmap(c1))
        ax.text(pnames2[i,0], pnames2[i,1], pnames2[i,2], names[i], color=cmap(c0))
        for j in range(NC):
            (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = balls0[i], balls1[i], balls2[i]
            vx, vy, vz = [x0, x1+(x2-x1)*(j+1)/NC], [y0, y1+(y2-y1)*(j+1)/NC], [z0, z1+(z2-z1)*(j+1)/NC]
            ax.plot(vx, vy, vz, color=cols[j], linestyle=':', linewidth=2, alpha=α)
    for i in range(N):
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = disc0[i], disc1[i], disc2[i]
        for j in range(NC-1):
            dxj, dyj, dzj = [x1+(x2-x1)*j/NC, x1+(x2-x1)*(j+1)/NC], [y1+(y2-y1)*j/NC, y1+(y2-y1)*(j+1)/NC], [z1+(z2-z1)*j/NC, z1+(z2-z1)*(j+1)/NC]
            ax.plot(dxj, dyj, dzj, color=cols[j], linestyle=':', linewidth=2, alpha=α)
    plt.title(title)

    #draw 3 arrows for the x, y, z axis
    L = 0.02
    ax.quiver(0, 0, 0, L, 0, 0, color='black')
    ax.quiver(0, 0, 0, 0, L, 0, color='black')
    ax.quiver(0, 0, 0, 0, 0, L, color='black')
    ax.text(L, 0, 0, 'x')
    ax.text(0, L, 0, 'y')
    ax.text(0, 0, L, 'z')

    ax.set_xlim([-2*R, 2*R])
    ax.set_ylim([-2*R, 2*R])
    ax.set_zlim([0, 0.12])

    ax.view_init(25, -49) # set camera orientation to 25, -49 0, to match the image in the picture
    plt.show()

# set up argparser to get the values of dx, dy, dz, drx, dry, drz, x1, y1, z1, rx1, ry1, rz1
import argparse
if __name__ == "__main__":

    # examples of how to call the script: 
    # python exapod.py -dx 0.03 -dy 0.01 -dz 0.008 -drx 0.0 -dry -0.157 -drz 0.0 -x1 0.0 -y1 0.0 -z1 0.08 -rx1 0.0 -ry1 0.0 -rz1 1.047
    # python exapod.py -dx 0.03 -dy 0.01 -dz 0.008 -dry -0.157 -z1 0.08 -rz1 1.047
    # python exapod.py -dx=0.03 -dy=0.01 -dz=0.008 -drx=0.0 -dry=-0.157 -drz=0.0 -x1=0.0 -y1=0.0 -z1=0.08 -rx1=0.0 -ry1=0.0 -rz1=1.047
    # python exapod.py -dx=0.03 -dy=0.01 -dz=0.008 -dry=-0.157 -z1=0.08 -rz1=1.047
    
    parser = argparse.ArgumentParser(description='Exapod')
    parser.add_argument('-dx', type=float, help='displacement along x', default=0.0, required=False)
    parser.add_argument('-dy', type=float, help='displacement along y', default=0.0, required=False)
    parser.add_argument('-dz', type=float, help='displacement along z', default=0.0, required=False)
    parser.add_argument('-drx', type=float, help='delta rotation around x', default=0.0, required=False)
    parser.add_argument('-dry', type=float, help='delta rotation around y', default=0.0, required=False)
    parser.add_argument('-drz', type=float, help='delta rotation around z', default=0.0, required=False)
    parser.add_argument('-x1', type=float, help='x initial position', default=X0, required=False)
    parser.add_argument('-y1', type=float, help='y initial position', default=Y0, required=False)
    parser.add_argument('-z1', type=float, help='z initial position', default=Z0, required=False)
    parser.add_argument('-rx1', type=float, help='initial rotation around x', default=RX0, required=False)
    parser.add_argument('-ry1', type=float, help='initial rotation around y', default=RY0, required=False)
    parser.add_argument('-rz1', type=float, help='initial rotation around z', default=RZ0, required=False)
    args = parser.parse_args()

    exapod(args.dx, args.dy, args.dz, args.drx, args.dry, args.drz, args.x1, args.y1, args.z1, args.rx1, args.ry1, args.rz1)