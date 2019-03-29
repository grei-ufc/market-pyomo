import matplotlib.pyplot as plt
import numpy as np
import glb as glb

def config():
    # use ggplot style for more sophisticated visuals
    plt.style.use('ggplot')

    #plt.clf()
    glb.fig = plt.figure(figsize=(7,3))

def live_plotter(x_vec,y1_data,line1,identifier='',pause_time=0.01, plot_position = 121, ylabel = 'Preço'):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        #fig = plt.figure(figsize=(7,3))
        print("adding subplot")
        ax = glb.fig.add_subplot(plot_position)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)        
        #update plot label/title
        plt.ylabel(ylabel)
        plt.title('Title: {}'.format(identifier))
        plt.show()
    
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
        line1.axes.set_ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return line1