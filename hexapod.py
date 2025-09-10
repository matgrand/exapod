## HEXAPOD

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

# disks
D1 = (0.426,-24.219,369.539,134.5,0.1658062789,0,0) # disk 1 -> top of the pipe
D2 = (0.69,46.717,-221,245,0,0,0) # disk 2 -> lower constraint disk
D3 = (0.69,46.717,-115.531,200,0,0,0) # disk 3 -> upper constraint disk

import numpy as np
from numpy import pi as π, cos, sin, exp, abs, sum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.linalg import norm
import argparse

np.set_printoptions(precision=4, suppress=True)

# geometric functions
#rototranslation
def rt(p:np.ndarray, r=np.array([0.0, 0.0, 0.0]), t=np.array([0.0,0.0,0.0])): # rotation and translation
    assert p.shape[-1] == 3, f'p shape: {p.shape}'
    assert r.shape == (3,), f'r shape: {r.shape}'
    assert t.shape == (3,), f't shape: {t.shape}'
    p_shape = p.shape
    p = p.reshape(-1, 3)
    rx, ry, rz = r
    Rx = np.array([[1, 0, 0], [0, cos(rx), -sin(rx)], [0, sin(rx), cos(rx)]])
    Ry = np.array([[cos(ry), 0, sin(ry)], [0, 1, 0], [-sin(ry), 0, cos(ry)]])
    Rz = np.array([[cos(rz), -sin(rz), 0], [sin(rz), cos(rz), 0], [0, 0, 1]])    
    # r_xyz = Rz @ Ry @ Rx # rotation matrix
    # p = (r_xyz @ p.T).T + t
    p = (Rx @ Ry @ Rz @ p.T).T + t

    return p.reshape(p_shape)

# projection onto horizontal plane along the normal of the oriented circle
def project_circle_onto_plane(center, radius, rotation_angles, plane_z=0):
    num_points = 100
    angles = np.linspace(0, 2 * np.pi, num_points)

    # 1. Create a circle in the XY plane centered at the origin
    circle_2d = np.array([radius * np.cos(angles), radius * np.sin(angles), np.zeros(num_points)]).T

    # 2. Rotate the circle according to the given rotation angles
    rotated_circle = rt(circle_2d, r=rotation_angles)

    # 3. Translate the rotated circle to the specified center
    circle_points_3d = rotated_circle + center

    # 4. Determine the normal vector of the oriented circle
    # The normal of a circle in the XY plane is [0, 0, 1].
    # We rotate this normal by the same rotation angles.
    initial_normal = np.array([0, 0, 1])
    circle_normal = rt(initial_normal.reshape(1, 3), r=rotation_angles).flatten()

    # Normalize the normal vector
    circle_normal = circle_normal / np.linalg.norm(circle_normal)

    # 5. Project each point of the 3D circle onto the horizontal plane (plane_z)
    projected_points = np.zeros_like(circle_points_3d)

    for i, p_3d in enumerate(circle_points_3d):
        # The line passing through p_3d in the direction of the normal vector
        # is p_3d + t * circle_normal.
        # We want to find t such that the Z-component of this line is plane_z.
        # p_3d.z + t * circle_normal.z = plane_z
        # t * circle_normal.z = plane_z - p_3d.z
        
        # Handle the case where the normal is almost perfectly horizontal (normal_z is close to 0)
        # If normal_z is 0, the circle is vertical and the projection is a line segment,
        # or it's parallel to the plane, in which case it just takes the plane_z.
        if abs(circle_normal[2]) < 1e-6: # If normal is parallel to the plane (or very close)
            projected_points[i] = [p_3d[0], p_3d[1], plane_z]
            # If the circle is exactly vertical, it projects to a line segment.
            # However, the problem states projection *along* the normal, which implies a
            # finite intersection unless normal_z is zero.
            # In this case, we simply project to the z-plane.
        else:
            t = (plane_z - p_3d[2]) / circle_normal[2]
            projected_points[i] = p_3d + t * circle_normal

    return projected_points, circle_points_3d, circle_normal

## Dynamics
HEXA_TOP_REST = np.array([A1T, A2T, C1T, C2T, B1T, B2T])
HEXA_BOT_REST = np.array([A1B, A2B, C1B, C2B, B1B, B2B])

# rototranslation to distances
def rt2d(r=np.array([0.0, 0.0, 0.0]), t=np.array([0.0,0.0,0.0])):
    hexa_top = rt(HEXA_TOP_REST, r, t)
    d = norm(hexa_top - HEXA_BOT_REST, axis=1)
    return d

# distances to rototranslation
def d2rt(d=np.array([D0, D0, D0, D0, D0, D0])):
    target_loss = 8e-2 # target loss 1e-1
    r, t = np.array([0.0, 0.0, 0.0]), np.array([0.0,0.0,0.0])
    lr = 1.0e-1 # learning rate
    sr = 1.0e-4 # step for numerical gradient (rotations)
    st = 1.0e-3 # step for numerical gradient (translations)

    def loss(d, d1):
        # return sum((d - d1)**2)
        return sum(abs((d - d1)))

    def grad(r, t):
        l = loss(d, rt2d(r, t))
        x,y,z = t.copy()
        rx,ry,rz = r.copy() 

        gx = (loss(d, rt2d(r, np.array([x+st, y, z]))) - l) 
        gy = (loss(d, rt2d(r, np.array([x, y+st, z]))) - l) 
        gz = (loss(d, rt2d(r, np.array([x, y, z+st]))) - l) 
        grx = (loss(d, rt2d(np.array([rx+sr, ry, rz]), t)) - l)
        gry = (loss(d, rt2d(np.array([rx, ry+sr, rz]), t)) - l)
        grz = (loss(d, rt2d(np.array([rx, ry, rz+sr]), t)) - l)
        return np.array([grx, gry, grz]), np.array([gx, gy, gz])

    for i in range(500):
        l = loss(d, rt2d(r, t))
        if l < target_loss: break
        r_grad, t_grad = grad(r, t)
        r -= lr * r_grad
        t -= lr * t_grad * 1e4
        lr *= 0.99 # decay learning rate
        sr *= 0.997 # .997
        st *= 0.999 # .999

    if loss(d, rt2d(r, t)) >= target_loss:
        print(f'not converged: {loss(d, rt2d(r, t))} ')

    # assert norm(rt2d(r, t) - d) < 1e-1, f'error too large: {norm(rt2d(r, t) - d)}'

    return r, t

def get_min_dist(p1, p2):
    assert p1.shape[-1] == 3, f'p1 shape: {p1.shape}'
    assert p2.shape[-1] == 3, f'p2 shape: {p2.shape}'
    p1 = p1.reshape(-1, 3)
    p2 = p2.reshape(-1, 3)
    dmin = np.inf
    argmin1, argmin2 = -1, -1
    for i in range(p1.shape[0]):
        for j in range(p2.shape[0]):
            d = norm(p1[i] - p2[j])
            if d < dmin:
                dmin = d
                argmin1, argmin2 = i, j
    return dmin, argmin1, argmin2

def in2Dpoly(pts, poly):
    pts_shape = pts.shape
    assert pts_shape[-1] == 2, f"pts.shape = {pts_shape}, should be (whatever, 2)"
    pts = pts.reshape(-1, 2)  # ensure pts is a 2D array of shape (n, 2)
    # Polygon edges: (xi, yi) to (xj, yj)
    xi, yi = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xi, -1), np.roll(yi, -1)
    px = pts[:, 0][:, np.newaxis]
    py = pts[:, 1][:, np.newaxis]
    # Check if point is between yi and yj, and if it is to the left of the edge
    intersect = ((yi <= py) & (py < yj)) | ((yj <= py) & (py < yi))
    slope = (xj - xi) / (yj - yi + 1e-12)  # avoid division by zero
    xints = xi + (py - yi) * slope
    crosses = px < xints
    inside = np.sum(intersect & crosses, axis=1) % 2 == 1
    return inside.reshape(pts_shape[:-1])  # reshape back to original shape

# plotting functions
def plot_3d_poly(ax, vertices, color='cyan', alpha=0.5, **kwargs):
    poly = Poly3DCollection([vertices], color=color, alpha=alpha, **kwargs)
    poly.set_facecolor(color)
    poly.set_edgecolor(color)
    poly.set_alpha(alpha)
    ax.add_collection3d(poly)
    return ax

def plot_origin(ax):
    ax.plot([0], [0], [0], 'ro')
    # ax.text(0, 0, 0, 'O', color='r')
    m = 500
    # arrows for axes
    L = m/5
    alr = 0.3
    lw = 3.5
    ax.quiver(0, 0, 0, L, 0, 0, color='r', arrow_length_ratio=alr, linewidth=lw)
    ax.quiver(0, 0, 0, 0, L, 0, color='g', arrow_length_ratio=alr, linewidth=lw)
    ax.quiver(0, 0, 0, 0, 0, L, color='b', arrow_length_ratio=alr, linewidth=lw)
    ax.text(L, 0, 0, 'X', color='r')
    ax.text(0, L, 0, 'Y', color='g')
    ax.text(0, 0, L, 'Z', color='b')

    #set axis limits
    ax.set_xlim([-m, m])
    ax.set_ylim([-m, m])
    ax.set_zlim([-m, m])
    ax.view_init(elev=30, azim=210)

    return ax

def plot_pipe(ax, r=np.array([0.0, 0.0, 0.0]), t=np.array([0.0, 0.0, 0.0]), d=D1, α_ratio=1.0):
    na = 50
    nz = 10
    alpha_intersect = 0.2
    alpha_edges = 0.6
    θs = np.linspace(0, 2*π, na)
    zs = np.linspace(200, -800, nz)
    x,y,z,R,rx,ry,rz = d
    p = np.zeros((na,nz,3))
    p[:,:,0] = R * np.cos(θs)[:,None]
    p[:,:,1] = R * np.sin(θs)[:,None]
    p[:,:,2] = zs[None,:]

    # rototranslation fixed wrt top of hexapod
    p = rt(p, r=np.array([rx, ry, rz]), t=np.array([x, y, z]))
    center_rest = np.mean(p[:,0,:], axis=0)

    # hexapod rototranslation
    p = rt(p, r=r, t=t)

    # cylinder surface
    ax.plot_surface(p[:,:,0], p[:,:,1], p[:,:,2], color='gray', alpha=0.2*α_ratio, edgecolor='none')

    if α_ratio >= 1.0:
        # x,y,z,R,rx,ry,rz = D1
        center = rt(np.array([x, y, z]), r=r, t=t)
        line_style = ':'
        #project the first disk onnto the D2 and D3 planes
        z2, z3, z4 = D2[2], D3[2], HT
        p2,_,_ = project_circle_onto_plane(center=center, radius=R, rotation_angles=np.array([rx, ry, rz])+r, plane_z=z2)
        p3,_,_ = project_circle_onto_plane(center=center, radius=R, rotation_angles=np.array([rx, ry, rz])+r, plane_z=z3)
        p4,_,_ = project_circle_onto_plane(center=center_rest, radius=R, rotation_angles=np.array([rx, ry, rz]), plane_z=z4)
        p4 = rt(p4, r=r, t=t) # apply hexapod rototranslation to p4
        ax.plot(p2[:,0], p2[:,1], p2[:,2], f'r{line_style}', alpha=alpha_edges*α_ratio)
        ax.plot(p3[:,0], p3[:,1], p3[:,2], f'r{line_style}', alpha=alpha_edges*α_ratio)
        ax.plot(p4[:,0], p4[:,1], p4[:,2], f'{line_style}', color='orange', alpha=alpha_edges*α_ratio)
        plot_3d_poly(ax, p2, color='r', alpha=alpha_intersect*α_ratio)
        plot_3d_poly(ax, p3, color='r', alpha=alpha_intersect*α_ratio)
        plot_3d_poly(ax, p4, color='orange', alpha=alpha_intersect*α_ratio)
        #project onto the base plane
        p0,_,_ = project_circle_onto_plane(center=center, radius=R, rotation_angles=np.array([rx, ry, rz])+r, plane_z=0)
        ax.plot(p0[:,0], p0[:,1], p0[:,2], f'{line_style}', color='gray', alpha=alpha_edges*α_ratio)
        plot_3d_poly(ax, p0, color='gray', alpha=alpha_intersect*α_ratio)

    return ax

def plot_hexa(ax, r=np.array([0.0, 0.0, 0.0]), t=np.array([0.0, 0.0, 0.0]), α_ratio=1.0):
    hexa = rt(HEXA_TOP_REST, r, t)
    a1t, a2t, c1t, c2t, b1t, b2t = hexa
    col = 'orange'
    marker_style = dict(color=col, linewidth=4, marker='o', markersize=10, markerfacecolor=col, markeredgecolor='k', alpha=α_ratio)
    text_style = dict(color='k', fontsize=12, weight='bold', bbox=dict(facecolor=col, edgecolor='k', boxstyle='round,pad=0.2'), alpha=α_ratio)
    # A
    ax.plot([a1t[0], A1B[0]], [a1t[1], A1B[1]], [a1t[2], A1B[2]], **marker_style)
    ax.plot([a2t[0], A2B[0]], [a2t[1], A2B[1]], [a2t[2], A2B[2]], **marker_style)
    ax.text(A1B[0], A1B[1], -10, 'A1', **text_style)
    ax.text(A2B[0], A2B[1], -10, 'A2', **text_style)
    # B
    ax.plot([b1t[0], B1B[0]], [b1t[1], B1B[1]], [b1t[2], B1B[2]], **marker_style)
    ax.plot([b2t[0], B2B[0]], [b2t[1], B2B[1]], [b2t[2], B2B[2]], **marker_style)
    ax.text(B1B[0], B1B[1], -10, 'B1', **text_style)
    ax.text(B2B[0], B2B[1], -10, 'B2', **text_style)
    # C
    ax.plot([c1t[0], C1B[0]], [c1t[1], C1B[1]], [c1t[2], C1B[2]], **marker_style)
    ax.plot([c2t[0], C2B[0]], [c2t[1], C2B[1]], [c2t[2], C2B[2]], **marker_style)
    ax.text(C1B[0], C1B[1], -10, 'C1', **text_style)
    ax.text(C2B[0], C2B[1], -10, 'C2', **text_style)
    #base
    hexa_base = np.array([A1B, A2B, C1B, C2B, B1B, B2B])
    hexa_base[:,2] = 0.0
    plot_3d_poly(ax, hexa_base, color='gray', alpha=0.1*α_ratio)
    hexa_base = np.append(hexa_base, hexa_base[0]).reshape(-1,3) # close the loop
    ax.plot(hexa_base[:,0], hexa_base[:,1], hexa_base[:,2], 'gray', alpha=0.5*α_ratio)
    # top (in dark orange)
    plot_3d_poly(ax, hexa, color='orange', alpha=0.2*α_ratio)
    hexa = np.append(hexa, hexa[0]).reshape(-1,3) # close the loop
    ax.plot(hexa[:,0], hexa[:,1], hexa[:,2], color='orange', alpha=0.5*α_ratio)


    return ax

def plot_constraint_disks(ax):
    text_style = dict(color='k', fontsize=12, weight='bold', bbox=dict(facecolor='r', edgecolor='k', boxstyle='round,pad=0.2'), alpha=1.0)
    # D2
    n = 50
    α1 = 0.1
    θs = np.linspace(0, 2*π, n)
    x,y,z,R,rx,ry,rz = D2
    p = np.array([R * np.cos(θs), R * np.sin(θs), np.zeros(n)]).T
    p = rt(p, r=np.array([rx, ry, rz]), t=np.array([x, y, z]))
    ax.plot(p[:,0], p[:,1], p[:,2], 'r-')
    plot_3d_poly(ax, p, color='r', alpha=α1)
    ax.text(x, y-1.1*R, z, 'D2', **text_style)
    # D3
    x,y,z,R,rx,ry,rz = D3
    p = np.array([R * np.cos(θs), R * np.sin(θs), np.zeros(n)]).T
    p = rt(p, r=np.array([rx, ry, rz]), t=np.array([x, y, z]))
    ax.plot(p[:,0], p[:,1], p[:,2], 'r-')
    plot_3d_poly(ax, p, color='r', alpha=α1)
    ax.text(x, y-1.1*R, z, 'D3', **text_style)
    return ax


if __name__ == '__main__':
    # examples of how to call the script: 
    # python hexapod.py -rx 20 -ry 0
    # python hexapod.py -d 330 285 200 200 285 300
    # python hexapod.py -rx 20 -y -40

    instructions = '''
    Hexapod:
     set [x, y, z, rx, ry, rz] (or a subset of them) to get the 6 arm extensions,
     or set the 6 arm extensions [-d] (all of them) to get [x, y, z, rx, ry, rz]
    Examples:
     python hexapod.py -rx 20 -ry 0
     python hexapod.py -d 330 285 200 200 285 300
     python hexapod.py -rx 20 -y -40
    '''

    parser = argparse.ArgumentParser(description=instructions, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-x', type=float, help='displacement along x [mm] -> -x 20', default=0.0, required=False)
    parser.add_argument('-y', type=float, help='displacement along y [mm] -> -y -40', default=0.0, required=False)
    parser.add_argument('-z', type=float, help='displacement along z [mm] -> -z 42', default=0.0, required=False)
    parser.add_argument('-rx', type=float, help='delta rotation around x [deg] -> -rx 20', default=0.0, required=False)
    parser.add_argument('-ry', type=float, help='delta rotation around y [deg] -> -ry 0', default=0.0, required=False)
    parser.add_argument('-rz', type=float, help='delta rotation around z [deg] -> -rz 0', default=0.0, required=False)
    parser.add_argument('-d', type=float, help='list of 6 arm extensions [mm] -> "-d A1 A2, C1, C2, B1, B2"', default=None, nargs=6, required=False)
    args = parser.parse_args()

    print(f'd: {args.d}')

    if args.d is None: # from rotation to ARM EXTENSIONS
        r = np.deg2rad(np.array([args.rx, args.ry, args.rz]))
        t = np.array([args.x, args.y, args.z])
        d = rt2d(r, t)
    else: # from ARM EXTENSIONS to rotation
        d = np.array([float(x) for x in args.d])
        r, t = d2rt(d)
        # check the conversion is consistent
        d2 = rt2d(r, t)
        error = norm(d2 - d)
        print(f'd error: {error:.3f} [mm]')
        assert error < 1e-1, f'UNFEASIBLE EXTENSIONS: error too large: {norm(d2 - d)}'
        d = d2 # use the consistent value


    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    α_rest = 0.2

    plot_origin(ax)
    plot_pipe(ax, α_ratio=α_rest)
    plot_pipe(ax, r=r, t=t)
    plot_hexa(ax, α_ratio=α_rest)
    plot_hexa(ax, r=r, t=t)
    plot_constraint_disks(ax)

    x,y,z,R,rx,ry,rz = D1
    center = rt(np.array([x, y, z]), r=r, t=t)
    z2, z3 = D2[2], D3[2]
    p2,_,_ = project_circle_onto_plane(center=center, radius=R, rotation_angles=np.array([rx, ry, rz])+r, plane_z=z2)
    p3,_,_ = project_circle_onto_plane(center=center, radius=R, rotation_angles=np.array([rx, ry, rz])+r, plane_z=z3)
    #plot the center of D2 and D3
    c2, c3, r2, r3 = D2[0:3], D3[0:3], D2[3], D3[3]
    θs = np.linspace(0, 2*π, 100)
    p_c2 = np.array([c2[0] + r2*cos(θs), c2[1] + r2*sin(θs), np.full(θs.shape, c2[2])]).T
    p_c3 = np.array([c3[0] + r3*cos(θs), c3[1] + r3*sin(θs), np.full(θs.shape, c3[2])]).T
    d_min2, amin1_2, amin2_2 = get_min_dist(p2, p_c2)
    d_min3, amin1_3, amin2_3 = get_min_dist(p3, p_c3)

    # check that p2 is inside the D2 disk
    inside2 = np.all(in2Dpoly(p2[:,0:2], p_c2[:,0:2]))
    inside3 = np.all(in2Dpoly(p3[:,0:2], p_c3[:,0:2]))
    if not inside2: d_min2 = -d_min2
    if not inside3: d_min3 = -d_min3
    
    # line connecting the closest points
    pm2a, pm2b = p2[amin1_2], p_c2[amin2_2]
    pm3a, pm3b = p3[amin1_3], p_c3[amin2_3]
    ax.plot([pm2a[0], pm2b[0]], [pm2a[1], pm2b[1]], [pm2a[2], pm2b[2]], 'r', linewidth=3, marker='o', markersize=4)
    ax.plot([pm3a[0], pm3b[0]], [pm3a[1], pm3b[1]], [pm3a[2], pm3b[2]], 'r', linewidth=3, marker='o', markersize=4)

    # print(f'min dist -> D2: {d_min2:.1f} [mm], D3: {d_min3:.1f} [mm]')
    title = ''
    title += f'HEXAPOD\nX: {t[0]:.1f} [mm], Y: {t[1]:.1f} [mm], Z: {t[2]:.1f} [mm] \n'
    title += f'RX: {np.rad2deg(r[0]):.1f} [deg], RY: {np.rad2deg(r[1]):.1f} [deg], RZ: {np.rad2deg(r[2]):.1f} [deg] \n'
    title += f'A1: {d[0]:.1f} [mm], A2: {d[1]:.1f} [mm] \nC1: {d[2]:.1f} [mm], C2: {d[3]:.1f} [mm] \nB1: {d[4]:.1f} [mm], B2: {d[5]:.1f} [mm]\n'
    title += f'Disk D2 {"TOUCHING" if not inside2 else ""}: {d_min2:.1f} [mm], Disk D3 {"TOUCHING" if not inside3 else ""}: {d_min3:.1f} [mm]' 
    ax.set_title(title, fontsize=16, weight='bold')
    plt.tight_layout()
    print(title)
    # Show the plot and close on any keypress
    def on_key(event):
        plt.close(event.canvas.figure)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


    # # show some random rotations
    # for _ in range(5):
    #     random_r = np.random.uniform(-0.25, 0.25, size=3)
    #     random_t = np.random.uniform(-20, 20, size=3)
    #     d = rt2d(random_r, random_t)
    #     r2, t2 = d2rt(d)

    #     fig = plt.figure(figsize=(12, 12))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.view_init(elev=30, azim=210)
    #     α_rest = 0.2


    #     plot_origin(ax)
    #     # plot_pipe(ax, α_ratio=α_rest)
    #     plot_pipe(ax, r=r2, t=t2)
    #     # plot_hexa(ax, α_ratio=α_rest)
    #     plot_hexa(ax, r=r2, t=t2)
    #     plot_constraint_disks(ax)

    #     x,y,z,R,rx,ry,rz = D1
    #     center = rt(np.array([x, y, z]), r=r2, t=t2)
    #     z2, z3 = D2[2], D3[2]
    #     p2,_,_ = project_circle_onto_plane(center=center, radius=R, rotation_angles=np.array([rx, ry, rz])+r2, plane_z=z2)
    #     p3,_,_ = project_circle_onto_plane(center=center, radius=R, rotation_angles=np.array([rx, ry, rz])+r2, plane_z=z3)
    #     #plot the center of D2 and D3
    #     c2, c3, r2, r3 = D2[0:3], D3[0:3], D2[3], D3[3]
    #     θs = np.linspace(0, 2*π, 100)
    #     p_c2 = np.array([c2[0] + r2*cos(θs), c2[1] + r2*sin(θs), np.full(θs.shape, c2[2])]).T
    #     p_c3 = np.array([c3[0] + r3*cos(θs), c3[1] + r3*sin(θs), np.full(θs.shape, c3[2])]).T
    #     d_min2, amin1_2, amin2_2 = get_min_dist(p2, p_c2)
    #     d_min3, amin1_3, amin2_3 = get_min_dist(p3, p_c3)
        
    #     pm2a, pm2b = p2[amin1_2], p_c2[amin2_2]
    #     pm3a, pm3b = p3[amin1_3], p_c3[amin2_3]
    #     ax.plot([pm2a[0], pm2b[0]], [pm2a[1], pm2b[1]], [pm2a[2], pm2b[2]], 'r', linewidth=3, marker='o', markersize=4)
    #     ax.plot([pm3a[0], pm3b[0]], [pm3a[1], pm3b[1]], [pm3a[2], pm3b[2]], 'r', linewidth=3, marker='o', markersize=4)

    #     # print(f'min dist -> D2: {d_min2:.1f} [mm], D3: {d_min3:.1f} [mm]')
    #     title = f'A1: {d[0]:.1f}[mm], A2: {d[1]:.1f}[mm] \nC1: {d[2]:.1f}[mm], C2: {d[3]:.1f}[mm] \nB1: {d[4]:.1f}[mm], B2: {d[5]:.1f}[mm]\nDist D2: {d_min2:.1f} [mm], D3: {d_min3:.1f} [mm]'
    #     ax.set_title(title, fontsize=16, weight='bold')
    #     plt.tight_layout()
    #     plt.show()

    #     print(title)
        