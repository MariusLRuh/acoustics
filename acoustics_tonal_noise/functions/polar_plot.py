import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
# sns.set()
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Times"],
# })

def polar_plot(theta, tonal_noise,verification,BM_tonal_noise,verification_2):
    shape = tonal_noise.shape
    min_dim = min(shape[0],shape[1])
    max_dim = max(shape[0],shape[1])
    # print(shape)

    frequency_modes = shape[0]
    num_evaluations = shape[1]


    fig, axs = plt.subplots(frequency_modes,num_evaluations,subplot_kw={'projection': 'polar'},figsize=(10, 10))
    angle = np.deg2rad(0)

    if min_dim > 1:        
        for i in range(frequency_modes):
            for j in range(num_evaluations):
                axs[i,j].plot(theta, tonal_noise[i,j,:],label = 'SPL (dB)')
                # axs[i,j].plot(theta, verification,label ='Hyunjune GD (dB)')
                axs[i,j].set_theta_zero_location("N")
                axs[i,j].set_theta_direction(-1)
                axs[i,j].set_title('Frequency mode: {}'.format(i+1) + '\n'+ 'Evaluation number: {}'.format(j+1))
                axs[i,j].legend(loc="lower left",bbox_to_anchor=(.5 + np.cos(angle)/1.5, .5 + np.sin(angle)/1.5))
    
    elif max_dim == 1:
        axs.plot(theta, tonal_noise[0,0,:],label ='CSDL GD (dB)')
        axs.plot(theta, BM_tonal_noise[0,0,:], label = 'CSDL BM (dB)')
        axs.plot(theta, verification,label ='Hyunjune GD (dB)')
        axs.plot(theta, verification_2,label ='Hyunjune BM (dB)')
        axs.set_theta_zero_location("N")
        axs.set_theta_direction(-1)
        axs.set_title('Frequency mode: {}'.format(1) + '\n'+ 'Evaluation number: {}'.format(1))
        axs.legend(loc="lower left",bbox_to_anchor=(.5 + np.cos(angle)/1.5, .5 + np.sin(angle)/1.5))

    elif frequency_modes == 1:
        for i in range(frequency_modes):
            for j in range(num_evaluations):
                axs[j].plot(theta, tonal_noise[i,j,:],label ='SPL (dB)')
                axs[j].set_theta_zero_location("N")
                axs[j].set_theta_direction(-1)
                axs[j].set_title('Frequency mode: {}'.format(i+1) + '\n'+ 'Evaluation number: {}'.format(j+1))
                axs[j].legend(loc="lower left",bbox_to_anchor=(.5 + np.cos(angle)/1.5, .5 + np.sin(angle)/1.5))
                
    elif num_evaluations == 1:
        for i in range(frequency_modes):
            for j in range(num_evaluations):
                axs[i].plot(theta, tonal_noise[i,j,:], label ='SPL (dB)')
                axs[i].set_theta_zero_location("N")
                axs[i].set_theta_direction(-1)
                axs[i].set_title('Frequency mode: {}'.format(i+1) + '\n'+ 'Evaluation number: {}'.format(j+1))
                axs[i].legend(loc="lower left",bbox_to_anchor=(.5 + np.cos(angle)/1, .5 + np.sin(angle)/1))

    plt.tight_layout()
    plt.show()