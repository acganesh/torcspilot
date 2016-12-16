import matplotlib.pyplot as plt
import numpy as np
# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)

trackName = 'alpine1'

testData = np.genfromtxt('alpine1_log.csv', delimiter=',', skip_header=1,
                     skip_footer=0, names=["time","speedX","speedY","speedZ","angle","damage","rpm","trackPos","steering","accel","brake","reward","loss"])

fig, ax1 = plt.subplots()
fig.suptitle(trackName+' Track Reward')
iterations = np.arange(0, testData.shape[0], 1) 
speedXsin = -abs(np.multiply(testData["speedX"], np.sin(testData["angle"])))
speedXcos = abs(np.multiply(testData["speedX"], np.cos(testData["angle"])))
speedXtrackPos = abs(np.multiply(testData["speedX"], testData["trackPos"]))

# Label graph axes 
ax1.set_xlabel('Iteration #')
ax1.set_ylabel('Reward Value')

# Plot multiple curves (one per training model) 
#transversal = ax1.scatter(iterations, abs(speedXsin), c=tableau20[0],s=5,edgecolor='none',label = '|V * sin(${\\theta}$)|')#'b-', , color=tableau20[0])
parallel = ax1.scatter(iterations, abs(speedXcos), c=tableau20[0],s=5,edgecolor='none', label = '|V * cos(${\\theta}$)|')#'r-', color=tableau20[1])
#position = ax1.scatter(iterations, abs(speedXtrackPos), c=tableau20[2],s=5,edgecolor='none', label = '|V * trackPos|')#'g-', color=tableau20[2])
total = ax1.scatter(iterations, testData['reward'], c=tableau20[2],s=5,edgecolor='none', label = "Total Reward")#'b-', color=tableau20[3])
plt.gca().set_xlim(left=0, right=testData.shape[0])

plt.legend(loc='lower right', scatterpoints=1, fontsize=8,
           ncol=2, borderaxespad=0.)#bbox_to_anchor=(0., 1.02, 1., .102), mode="expand",
#plt.show() 
plt.savefig(trackName+'_reward_plot.pdf', bbox_inches='tight')