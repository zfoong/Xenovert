# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 07:19:39 2022

@author: zfoong
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from model.xenovert_2D import xenovert_2D
from sklearn.preprocessing import normalize
import copy
import operator
import matplotlib.patches as patches
from matplotlib import colors
from numpy import ma

np.random.seed(99)

input_size = 100000
testing_size = 2000
movie = True
plot = False
dynamic = True
learning_rate = 0.000001
s_learning_rate = 0.001
level = 3
output_num = 8
trials = 1
shift_type = "instant" #["instant", "gradual", "seasonal"]

def gen_multivariate_normal(m, c, input_size = 100000):
    size = int(input_size / len(m))
    x = np.array([])
    y = np.array([])
    for i in range(len(m)):
        mean = m[i]
        cov = c[i]
        x1, y1 = np.random.multivariate_normal(mean, cov, size).T
        x = np.concatenate((x, x1))
        y = np.concatenate((y, y1))
    z = np.vstack((x, y))
    z = z.T
    np.random.shuffle(z)
    return x, y, z

def rot(array, theta, origin):
    t = theta # 1.5708  # 90 degree
    ox, oy = origin[0], origin[1]  # point to rotate about
    A = np.matrix([[np.cos(t), -np.sin(t)],
                   [np.sin(t), np.cos(t)]])
    
    w = np.zeros(testing_array.shape)
    shifted = testing_array-np.array([ox,oy])
    for i,v in enumerate(shifted):
      w[i] = A @ v
    return w

def getcov(scale, theta):
    cov = np.array([
        [1*(scale + 1), 0],
        [0, 1/(scale + 1)]
    ])

    r = rot(theta)
    return r @ cov @ r.T

def histogram_intersection(h1, h2):
    minima = np.minimum(h1, h2)
    intersection = np.true_divide(np.sum(minima), np.sum(h2))
    return intersection

nearest_id = []
anim_history = []
l1_anim_history = []
four_points_history = []

results_list = []

# Rotation
# mean_vector_1 = [[-50, -50], [50, 50]]
# cov_matrix_1 = [[[200, 4], [4, 200]], [[200, 5], [5, 200]]]

# mean_vector_2 = [[-50, 50], [50, -50]]
# cov_matrix_2 = [[[200, 4], [4, 200]],  [[200, 5], [5, 200]]]

# Sysmetrical
# mean_vector_1 = [[-100, 50], [0, -50], [100, 50]]
# cov_matrix_1 = [[[200, 4], [4, 200]], [[200, 5], [5, 200]], [[200, 4], [4, 200]]]

# mean_vector_2 = [[-100, -50], [0, 50], [100, -50]]
# cov_matrix_2 = [[[200, 4], [4, 200]], [[200, 5], [5, 200]], [[200, 4], [4, 200]]]

# Asyncmetrical 
mean_vector_1 = [[-50, 0]]
cov_matrix_1 = [[[200, 5], [5, 200]],  [[200, 5], [5, 200]]]

mean_vector_2 = [[150, 0]]
cov_matrix_2 = [[[200, 200], [10, 50]]]


for trial in range(trials):
    if dynamic is True:
        x1, y1, input_array1 = gen_multivariate_normal(mean_vector_1, 
                                                       cov_matrix_1, 
                                                       input_size)
    
    x2, y2, input_array2 = gen_multivariate_normal(mean_vector_2, 
                                                   cov_matrix_2, 
                                                   input_size)
    
    
    x, y, testing_array = gen_multivariate_normal(mean_vector_2, 
                                                  cov_matrix_2, 
                                                  testing_size)
    
    if dynamic is True:
        if shift_type == "instant":
            input_array = np.concatenate((input_array1, input_array2))
        elif shift_type == "gradual":
            
            p = round(input_size/2)
            ls = np.linspace(0, 10, 10, dtype=int)**2
            
            sub_input_array1 = input_array1[input_size-p:]
            sub_input_array2 = input_array2[input_size-p:]
            
            input_array = input_array1[:input_size-p]
            
            split_1 = [sub_input_array1[ls[i]*500:ls[i+1]*500] for i in range(9)]
            split_2 = [sub_input_array2[ls[i]*500:ls[i+1]*500] for i in range(9)]
            split_1 = np.flip(split_1)
            
            for i in range(9):
                mixed_array = np.concatenate((split_1[i], split_2[i]))
                mixed_array = np.concatenate((split_1[i]+(np.mean(mixed_array) - np.mean(split_1[i])), 
                                              split_2[i]+(np.mean(mixed_array) - np.mean(split_2[i]))))
                
                np.random.shuffle(mixed_array)
                input_array = np.concatenate((input_array, mixed_array))
            
            input_array = np.concatenate((input_array, input_array2[:input_size-p]))
        
        elif shift_type == "seasonal":
            part = round(input_size/2)
            input_array = np.concatenate((input_array1[:part], 
                                          input_array2[:part], 
                                          input_array1[part:], 
                                          input_array2[part:]))
    else:
        input_array = input_array2      
    
    if plot is True:
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        hist, xedges, yedges = np.histogram2d(x, y, bins=(50,50))
        xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
        
        xpos = xpos.flatten()/2.
        ypos = ypos.flatten()/2.
        zpos = np.zeros_like (xpos)
        
        dx = xedges [1] - xedges [0]
        dy = yedges [1] - yedges [0]
        dz = hist.flatten()
        
        cmap = cm.get_cmap('jet') 
        max_height = np.max(dz)   
        min_height = np.min(dz)
        rgba = [cmap((k-min_height)/max_height) for k in dz] 
        
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    
    xeno = xenovert_2D(learning_rate, init_value=[0,0], s_learning_rate=s_learning_rate)
    for i in range(level):
        xeno.grow()
        
    print(xeno.return_val())
        
    for v in input_array:
        xeno.input(v)
        value_list, l1_list = xeno.return_val()
        anim_history.append(copy.deepcopy(value_list))
        l1_anim_history.append(copy.deepcopy(l1_list))
        
        xeno.area_subdivision()
        
        four_points = xeno.return_area_points()
        four_points_history.append(copy.deepcopy(four_points))
    
    for x in testing_array:
        max_id = np.argmax(xeno.convert(x))
        nearest_id.append(max_id)
    
std_list = []
env_std_list = []
env_mean_list = []

ni = []
_, _, testing_array = gen_multivariate_normal(mean_vector_2, 
                                              cov_matrix_2, 
                                              testing_size)
for t in testing_array:
    max_id = np.argmax(xeno.convert(t))
    ni.append(max_id)

uni_hist = [testing_size/output_num for _ in range(output_num)] 
hist, _ = np.histogram(ni, bins=output_num)
hi = histogram_intersection(hist, uni_hist)

results_list.append(hi)
print(results_list)

def animation_update(n, img1, lines):  

    test = np.array(anim_history[n])

    for i in range(output_num-1):
        c1 = anim_history[n][i][1] - l1_anim_history[n][i] * anim_history[n][i][0]
        
        loc1_x = anim_history[n][i][0] + 10
        loc2_x = anim_history[n][i][0] - 10
            
        loc1_1_y = l1_anim_history[n][i] * loc1_x + c1
        loc1_2_y = l1_anim_history[n][i] * loc2_x + c1
        
        l1_xy = ([loc1_x, loc2_x], [loc1_1_y, loc1_2_y])
        
        lines[i].set_xdata(l1_xy[0])
        lines[i].set_ydata(l1_xy[1])
    
    img1.set_offsets(test)
    return img1, lines

if movie:    
    ss = 150        
    anim_history = anim_history[::100]
    l1_anim_history = l1_anim_history[::100]
    four_points_history = four_points_history[::100]
    
    fig, ax = plt.subplots(dpi=300)
    plt.gca().set_aspect('equal')
    
    hist, xedges, yedges = np.histogram2d(x2, y2, bins=(50,50))
    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
    
    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    
    x = np.linspace(np.min(xpos), np.max(xpos), 50)
    y = np.linspace(np.min(ypos), np.max(ypos), 50)
    
    X, Y = np.meshgrid(x, y)
    hist_log = ma.log2(hist)
    fracs = hist_log / hist_log.max()
    norm = colors.Normalize(fracs.min(), 1)
    plt.contourf(X, Y, fracs.T, 5, cmap='Blues', norm=norm)

    min_xpos = np.min(xpos)
    max_xpos = np.max(xpos)
    min_ypos = np.min(ypos)
    max_ypos = np.max(ypos)
    
    if dynamic:
        hist, xedges, yedges = np.histogram2d(x1, y1, bins=(50,50))
        xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
        
        xpos = xpos.flatten()/2.
        ypos = ypos.flatten()/2.
        
        x = np.linspace(np.min(xpos), np.max(xpos), 50)
        y = np.linspace(np.min(ypos), np.max(ypos), 50)
        
        X, Y = np.meshgrid(x, y)

        hist_log = ma.log2(hist)
        fracs = hist_log / hist_log.max()
        norm = colors.Normalize(fracs.min(), 1)
        plt.contourf(X, Y, fracs.T, 5, cmap='Reds', norm=norm)
        
        min_xpos = min(np.min(xpos), min_xpos)
        max_xpos = max(np.max(xpos), max_xpos)
        min_ypos = min(np.min(ypos), min_ypos)
        max_ypos = max(np.max(ypos), max_ypos)
    
    ax.set_xlim(min_xpos, max_xpos)
    ax.set_ylim(min_ypos, max_ypos)
    
    img1 = ax.scatter([],[], cmap="rgb", color='r', alpha=1)

    lines = []
    for _ in range(output_num-1):
        l1 = plt.axline((450, -50), (451, -51), linewidth=1, color='r')
        lines.append(l1)
    
    ani = animation.FuncAnimation(fig, animation_update, frames=len(anim_history), fargs=([img1, lines]), interval=1, blit=False)
    FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
    ani.save('movie/xenovert_multivariate_anim.mp4', writer=FFwriter)

# ------------------------------------------------------
