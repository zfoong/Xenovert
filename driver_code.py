
# ------------------ system import --------------------
# import sys
# sys.path.insert(1, '/model')
# sys.path.insert(1, '/data_loader')
# ------------------ system import --------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import matplotlib.animation as animation
from model.xenovert import xenovert
from scipy.stats import chisquare
from scipy import stats
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
from colour import Color
from utils import *
from model.xenovert_new import xenovert_new
plt.rcParams.update(plt.rcParamsDefault)

def ridgeline(data, overlap=0, fill=True, labels=None, n_points=150):
    if overlap > 1 or overlap < 0:
        raise ValueError('overlap must be in [0 1]')
    xx = np.linspace(np.min(np.concatenate(data)),
                     np.max(np.concatenate(data)), n_points)
    curves = []
    ys = []
    fig = plt.figure(figsize=(2, 1.5), dpi=300)
    c_start = Color("tomato")
    colors = list(c_start.range_to(Color("red"),len(data)))
    for i, d in enumerate(data):
        pdf = gaussian_kde(d)
        y = i*(1.0-overlap)
        ys.append(y)
        curve = pdf(xx)
        if fill:
            plt.fill_between(xx, np.ones(n_points)*y, 
                              curve+y, zorder=len(data)-i+1, color=str(colors[i]))
        plt.plot(xx, curve+y, c='white', zorder=len(data)-i+1, linewidth=.5)
    if labels:
        plt.yticks(ys, labels)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xlabel('input')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()

def is_shifted(x, y, alpha=0.05):
    _, pvalue = stats.kstest(x, y)
    if pvalue > alpha:
       print("No shift detected")
       return False
    else:
       print("Shift detected")
       return True

def histogram_intersection(h1, h2):
    minima = np.minimum(h1, h2)
    intersection = np.true_divide(np.sum(minima), np.sum(h2))
    return intersection

def multimodal_gaussian(ags, input_size):
    m, std = ags
    n = len(m)
    dis_list = []
    for i in range(n):    
        dis = np.random.normal(m[i], std[i], int(input_size/n))
        dis_list.append(dis)
    multi_dis = np.concatenate(dis_list)
    np.random.shuffle(multi_dis)
    return multi_dis

def env(problem, x, prior, input_size=100000, testing_size=100000, dynamic=True, testing=False):
    input_array1 = None
    input_array2 = None
    testing_array = None
    func = None
    
    if problem == "uniform":
        func = lambda s, x : np.random.rand(s) + x
    elif problem == "poisson":
        func = lambda s, x : np.random.poisson(x, s)+1
    elif problem == "normal":
        func = lambda s, x : np.random.normal(x[0], x[1], s)
    elif problem == "exp":
        func = lambda s, x : np.exp(-np.random.rand(s)*x)
    elif problem == "beta":
        func = lambda s, x : np.tanh(np.random.rand(s)*x)
    elif problem == "chi":
        func = lambda s, x : np.random.chisquare(x, s)
    elif problem == "multimodal_gaussian":
        func = lambda s, x : multimodal_gaussian(x, s)
    else:
        print("Invalid problem id")
    
    if not testing:
        if dynamic:
            input_array1 = func(input_size, prior)
        input_array2 = func(input_size, x)
    testing_array = func(testing_size, x)
    return input_array1, input_array2, testing_array

def evaluate():
    return

# ---------------------------- parameters ----------------------------
input_size = 100000
testing_size = 100000

trials = 1
problems = ["uniform", "normal", "multimodal_gaussian", "chi"]
shift_types = ["instant", "gradual", "seasonal"]
dynamic = True
movie = True

learning_rate = 0.00001
level = 2
output_num = 2**(level+1) if level > 0 else 2
# ---------------------------- parameters ----------------------------

for problem in problems:
    for shift_type in shift_types:
        results_list = []
        for trial in range(trials):
        # ---------------------------- problems ----------------------------
            input_array1 = None
            input_array2 = None
            testing_array = None
            env_func = None
            
            # Generate Uniform distribution
            if problem == "uniform":
                env_func = lambda t: env(problem, 5, 3, input_size, testing_size, dynamic, t)
                input_array1, input_array2, testing_array = env_func(False)
            
            # Generate Poisson distribution
            elif problem == "poisson":
                env_func = lambda t: env(problem, 6, 3, input_size, testing_size, dynamic, t)
                input_array1, input_array2, testing_array = env_func(False)
            
            # Generate Gaussian distribution
            elif problem == "normal":
                env_func = lambda t: env(problem, (2, 4), (10, 2), input_size, testing_size, dynamic, t)
                # env_func = lambda t: env(problem, (0, 5), (250, 5), input_size, testing_size, dynamic, t)
                input_array1, input_array2, testing_array = env_func(False)
            
            # Generate Multimodal Gaussian distribution
            elif problem == "multimodal_gaussian":
                env_func = lambda t: env(problem, ([20, 40], [5, 3]),([-5, 8, 25], [2, 5, 5]), input_size, testing_size, dynamic, t)
                input_array1, input_array2, testing_array = env_func(False)
            
            # EXP distribution
            elif problem == "exp":
                env_func = lambda t: env(problem, 4, 2, input_size, testing_size, dynamic, t)
                input_array1, input_array2, testing_array = env_func(False)
            
            # tanh distribution
            elif problem == "beta":
                env_func = lambda t: env(problem, 4, 2, input_size, testing_size, dynamic, t)
                input_array1, input_array2, testing_array = env_func(False)
            
            # chi distribution
            elif problem == "chi":
                env_func = lambda t: env(problem, 1, 4, input_size, testing_size, dynamic, t)
                input_array1, input_array2, testing_array = env_func(False)
                
            if dynamic:
                total_input_size = input_size*2
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

            # ---------------------------- problems ----------------------------
            
            
            # ---------------------------- learning ----------------------------
            nearest_id = []
            anim_history = []
            result = []
            
            xeno= xenovert(learning_rate, round_to=0, init_value=input_array[0]+0.001)
            for i in range(level):
                xeno.grow()
                
            # print(xeno.return_val())
                
            for i, x in enumerate(input_array):
                xeno.input(x)
                value_list = xeno.return_val()
                xeno_values = np.sort(value_list)
                anim_history.append(xeno_values)
                max_id = np.argmax(xeno.convert(x))
                nearest_id.append(max_id)
                
                if i != 0 and i % 100 == 0:
                    uni_hist = [100/output_num for _ in range(output_num)] 
                    hist, _ = np.histogram(nearest_id, bins=output_num)
                    hi = histogram_intersection(hist, uni_hist)
                    result.append(hi)
                    nearest_id = []
            
            results_list.append(result)
        # ---------------------------- learning ----------------------------
        
        
        # ---------------------------- testing ----------------------------
        nearest_id = []
        for x in testing_array:
            max_id = np.argmax(xeno.convert(x))
            nearest_id.append(max_id)
        
        std_list = []
        env_std_list = []
        env_mean_list = []
        for i in range(trials): 
            ni = []
            _, _, testing_array = env_func(True)
            for x in testing_array:
                max_id = np.argmax(xeno.convert(x))
                ni.append(max_id)
            
            uni_hist = [testing_size/output_num for _ in range(output_num)] 
            hist, _ = np.histogram(ni, bins=output_num)
            hi = histogram_intersection(hist, uni_hist)
            
            std_list.append(hi)
        
            env_std_list.append(np.std(testing_array))
            env_mean_list.append(np.mean(testing_array))
            
        
        score = np.mean(std_list)
        score_std = np.std(std_list)
        env_deviation = np.std(env_std_list) + np.std(env_mean_list)
        
        print("--------------------")
        print(problem)
        print(shift_type)
        print("score = " + str(score))
        print("score std = " + str(score_std))
        print("distribution std = " + str(env_deviation))
        # ---------------------------- testing ----------------------------
        
        
        # ---------------------------- plot ----------------------------
        fig = plt.figure(figsize=(6,3), dpi=300)
        _, _, patches = plt.hist(nearest_id, bins=np.arange((output_num)+0.5), edgecolor='black', linewidth=0.3)
        
        ticks = [(patch._x0 + patch._x1)/2 for patch in patches]
        ticklabels = [i for i in range(output_num)]
        plt.xticks(ticks, ticklabels)
        
        plt.xlabel('Interval')
        plt.ylabel('Activation frequency')
        plt.title('Xenovert output')
        
        fig = plt.figure(figsize=(4,1), dpi=300)
        plt.xlabel('Input')
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        if dynamic:
            plt.hist(input_array1, 100, density=True, color="#47a3ff", alpha=0.7)
        plt.hist(input_array2, 100, density=True, color='tomato', alpha=0.7)
        for i in xeno_values:
            plt.axvline(x=i, color='black', linestyle='--', linewidth=0.7)
        # ---------------------------- plot ----------------------------
        
        
        # ---------------------------- anim ----------------------------
        def animation_update(n, vl_list):  
            for i, vl in enumerate(vl_list):
                vl.set_xdata([anim_history[n][i], anim_history[n][i]])
            return vl_list,
        
        if movie:
            anim_history = anim_history[::100]
            plt.figure(figsize=(5,5), dpi=300)
            fig, ax = plt.subplots()
            vl_list = []
            for _ in range(output_num-1):
                vl = ax.axvline(0, color='black', linestyle='--', linewidth=0.7)
                vl_list.append(vl)
            
            if dynamic:
                plt.hist(input_array1, 100, density=True, color='#840924', alpha=0.7)
            plt.hist(input_array2, 100, density=True, color='#276eb0', alpha=0.7)
            plt.tight_layout()
            
            ani = animation.FuncAnimation(fig, animation_update, frames=len(anim_history), fargs=([vl_list]), interval=1, blit=False)
            FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
            ani.save(f"movie/xenovert_anim_{problem}_{shift_type}.mp4", writer=FFwriter)
        # ---------------------------- anim ----------------------------