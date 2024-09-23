
import wizard

print('- - - - - - - - S T A R T - - - - - - - -')

"""
dc = wizard.read('/Users/flx/Documents/data/Tom_MA/2024_engels/Image.fsm')

print(dc)

print('- - - - - S T A R T   R E C O R D - - - - -')
dc.start_recording()

dc.remove_spikes(threshold = 6500, window = 3)
dc.resize(y_new=dc.shape[1]-100, x_new=dc.shape[2]-100)

print('- - - - - S T O P     R E C O R D - - - - -')

dc.save_template('ahhhhh')
print(dc)
"""
dc2 = wizard.read('/Users/flx/Documents/data/Tom_MA/2024_engels/Image.fsm')

# wizard.plotter(dc2)
dc2.execute_template('ahhhhh.yml')
# wizard.plotter(dc2)
print( '- - - - - - - - - D O N E - - - - - - - -')


