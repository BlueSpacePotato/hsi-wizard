import wizard
import numpy as np

# create a dc
dc = wizard.DataCube(np.random.rand(20, 640, 460))

# start recoding the methods
dc.start_recording()

# process the dc
dc.remove_spikes(threshold = 6500, window = 3)
dc.resize(x_new=dc.shape[1]-100, y_new=dc.shape[2]-100)

# save template
dc.save_template('02_example')

