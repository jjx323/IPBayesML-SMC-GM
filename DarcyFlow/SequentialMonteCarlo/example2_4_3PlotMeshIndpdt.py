import matplotlib.pyplot as plt
import scienceplots
import matplotlib
plt.style.use('science')
plt.style.use(['science','ieee'])
plt.rcParams['font.family'] = 'Times New Roman'
import numpy as np

fig_size=(16,10)
width_frame_line=1
font_size=50
pad_tem=10


folder_list=['./results/example2MeshDpdt/','./results/example2MeshIndpdt/']
for results_folder in folder_list:
    tem20 = np.append(np.loadtxt(results_folder+'h20.txt'),1)
    tem40 = np.append(np.loadtxt(results_folder+'h40.txt'),1)
    tem60 = np.append(np.loadtxt(results_folder+'h60.txt'),1)
    tem80 = np.append(np.loadtxt(results_folder+'h80.txt'),1)
    tem100= np.append(np.loadtxt(results_folder+'h100.txt'),1)
    
    fig=plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(width_frame_line)
    ax.spines['bottom'].set_linewidth(width_frame_line)
    ax.spines['left'].set_linewidth(width_frame_line)
    ax.spines['right'].set_linewidth(width_frame_line)
    
    ax.tick_params(axis='both', which='major',labelsize=font_size,pad=pad_tem, width=1,   length=5  ) 
    ax.tick_params(axis='both', which='minor',labelsize=font_size,pad=pad_tem, width=0.5, length=2.5) 
    
    width_line=2
    plt.plot(tem20,label='400',linewidth=width_line)
    plt.plot(tem40,label='1600',linewidth=width_line)
    plt.plot(tem60,label='3600',linewidth=width_line)
    plt.plot(tem80,label='6400',linewidth=width_line)
    plt.plot(tem100,label='10000',linewidth=width_line*2)
    
    plt.xlabel('Layer Number',labelpad=pad_tem,fontsize=font_size)  # Label for x-axis
    plt.ylabel('Sum of h',labelpad=pad_tem,fontsize=font_size)   # Label for y-axis
    plt.legend(fontsize=font_size)
    
    if results_folder=='./results/example2MeshIndpdt/':
        fig_name='./results/example2Analysis/meshIndpdt.png'
    else:
        fig_name='./results/example2Analysis/meshDpdt.png'
    
    plt.savefig(fig_name)
    plt.close()




