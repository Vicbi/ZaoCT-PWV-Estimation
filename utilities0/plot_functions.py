from matplotlib import pyplot as plt
from matplotlib import cm as cm
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np
    
    
def plot_data(data1,data2,experiment_type,prediction_variable,regressor,save_figure_boolean):
    plt.rcParams.update({'font.size': 16})
    left = 0.125  # the left side of the subplots of the figure
    right = 0.9   # the right side of the subplots of the figure
    bottom = 0.1  # the bottom of the subplots of the figure
    top = 0.9     # the top of the subplots of the figure
    wspace = 0.2  # the amount of width reserved for space between subplots, expressed as a fraction of the average axis width
    hspace = 0.4  # the amount of height reserved for space between subplots, expressed as a fraction of the average axis height
            
    data1 = np.ravel(data1)
    data2 = np.ravel(data2)
    
    if prediction_variable == 'CO':
        xmin = 2; xmax = 12; ymin = xmin; ymax = xmax; err = 1.5;
        temp_ticks = list(range(xmin, xmax+1, 2))
        ylabel_name1 = 'Predicted CO [L/min]'; xlabel_name1 = 'Reference CO [L/min]'
        ylabel_name2 = 'CO Error [L/min]'; xlabel_name2 = 'Average CO [L/min]'
        file_suffix = 'CO'; unit = 'L/min'
        
    if prediction_variable == 'aSBP': 
        xmin = 60; xmax = 200; ymin = xmin; ymax = xmax; err = 20; 
        temp_ticks = list(range(xmin, xmax+1, 20))
        ylabel_name1 = 'Predicted aSBP [mmHg]'; xlabel_name1 = 'Reference aSBP [mmHg]'
        ylabel_name2 = 'aSBP Error [mmHg]'; xlabel_name2 ='Average [mmHg]'
        file_suffix = 'aSBP'; unit = 'mmHg'

    if prediction_variable == 'Ees': 
        xmin = 0.5; xmax = 4; ymin = xmin; ymax = xmax; err = 1; 
        temp_ticks = [x/2 for x in range(1, 9)]
        ylabel_name1 = 'Predicted Ees [mmHg/mL]'; xlabel_name1 ='Reference Ees [mmHg/mL]'; 
        ylabel_name2 ='Ees Error [mmHg/mL]'; xlabel_name2 ='Average Ees [mmHg/mL]'
        file_suffix = 'Ees'; unit = 'mmHg/mL'

    if prediction_variable == 'CT_sys':
        xmin = 0; xmax = 4; ymin = xmin; ymax =xmax ; err = 1.2; 
        temp_ticks = np.arange(0, 4.5, 0.5)
        ylabel_name1 = 'Predicted CT [mL/mmHg]'; xlabel_name1 = 'Reference CT [mL/mmHg]'
        ylabel_name2 = 'Estimated - Reference CT [mL/mmHg]'; xlabel_name2 ='Mean of Reference and Estimated CT [mL/mmHg]'
        file_suffix = 'CT_sys'; unit = 'mmHg/mL'  
    
    if prediction_variable == 'Z_compl':
        xmin = 0; xmax = 0.12; ymin = xmin; ymax = xmax; err = 0.02; 
        temp_ticks = np.arange(0, 0.13, 0.03)
        ylabel_name1 = 'Predicted Zao [mmHg.s/mL]'; xlabel_name1 = 'Reference Zao [mmHg.s/mL]'
        ylabel_name2 = 'Estimated - Reference Zao [mmHg.s/mL]'; xlabel_name2 ='Mean of Reference and Estimated Zao [mmHg.s/mL]'
        file_suffix = 'Zao'; unit = 'mmHg.s/mL'
        

    f, axs = plt.subplots(2,1,figsize=(5,10))
    plt.subplots_adjust(left,bottom,right,top,wspace,hspace)
    plt.suptitle(regressor)
    axs[0].set_title('Scatterplot')
    axs[0].plot([xmin,xmax],[xmin,xmax],'k')
    axs[0].axis([xmin,xmax,ymin,ymax])
    axs[0].set_ylabel(ylabel_name1, fontsize=18, labelpad=20)
    axs[0].set_xlabel(xlabel_name1, fontsize=18, labelpad=10)
    axs[0].scatter(data1, data2, marker="o",color="k",alpha=0.2,s=15)
    plt.setp(axs[0], xticks=temp_ticks, xticklabels=temp_ticks,
        yticks=temp_ticks)
    

    axs[1].set_title('Bland-Altman')
    bias,upper_LoA,lower_LoA = bland_altman_plot(data2, data1,axs[1])
    axs[1].axis([xmin,xmax,-err,err])
    axs[1].set_ylabel(ylabel_name2, fontsize=18, labelpad=10)
    axs[1].set_xlabel(xlabel_name2, fontsize=18, labelpad=10)
    plt.setp(axs[1], xticks=temp_ticks, xticklabels=temp_ticks)
    plt.show()
    
    print('Bias:',bias,'{}'.format(unit)); print('Upper LoA:', upper_LoA,'{}'.format(unit)); print('Lower LoA:', lower_LoA,'{}'.format(unit));
    
    # Save figure
    if save_figure_boolean:
        filename = '{}_scatter_bland-altman_{}_{}.tiff'.format(regressor,prediction_variable,experiment_type)
        f.savefig(filename, dpi = 300, bbox_inches='tight')
        
                        
def bland_altman_plot(data1, data2, ax_i,*args, **kwargs):    
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    ax_i.scatter(mean, diff, marker="o", color="k",alpha=0.2,s=15)
    ax_i.axhline(md,           color='black', linestyle='-')
    ax_i.axhline(md + 1.96*sd, color='gray', linestyle='--')  
    ax_i.axhline(md - 1.96*sd, color='gray', linestyle='--')
    
    bias = np.round(md,3)
    upper_LoA = np.round(md + 1.96*sd,4)
    lower_LoA = np.round(md - 1.96*sd,4)
    
    return bias,upper_LoA,lower_LoA


def plot_data_distributions(data,input_feature):
    import numpy as np
    import matplotlib.pyplot as plt
    
    if input_feature == 'brSBP':
        select_color = "paleturquoise"
        ylabel_name = 'Normalized SBP'
    
    if input_feature == 'brDBP':
        select_color = "mediumaquamarine"
        ylabel_name = 'Normalized DBP'
    
    if input_feature == 'HR':
        select_color = "khaki"
        ylabel_name = 'Normalized HR'
        
    if input_feature == 'brMAP':
        select_color = "mediumaquamarine"
        ylabel_name = 'Normalized MAP'
        
    if input_feature == 'cfPWV':
        select_color = "orchid"
        ylabel_name = 'Normalized cfPWV'
    
    if input_feature == 'crPWV':
        select_color = "lightcoral"
        ylabel_name = 'Normalized crPWV'
    
    if input_feature == 'Zao':
        select_color = "darkturquoise"
        ylabel_name = 'Normalized Zao'
    
    if input_feature == 'local_ascending_aortic_PWV':
        select_color = "darkturquoise"
        ylabel_name = 'Normalized aortic PWV'
    
    if input_feature == 'CT_sys':
        select_color = "darkturquoise"
        ylabel_name = 'Normalized CT'
        
        
    plt.rcParams.update({'font.size': 21})
    fig = plt.figure(figsize=(7,6))
    plt.plot(data,'o',color=select_color,alpha=0.5,markersize=5) # mediumaquamarine, lightseagreen
    plt.ylabel(ylabel_name, fontsize=32)
    plt.xlabel('Cases', fontsize=32)
    fig.savefig('{}_data.tiff'.format(input_feature),dpi=500, bbox_inches='tight')
    plt.show()
  