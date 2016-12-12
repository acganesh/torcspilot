import matplotlib.pyplot as plt
import numpy as np

filePath = 'a_speedway_log.csv'
trackName = 'a_speedway'

testData = np.genfromtxt(filePath, delimiter=',', skip_header=1,
                     skip_footer=0, names=["time","speedX","speedY","speedZ","angle","damage","rpm","trackPos","steering","accel","brake","reward","loss"])

fig, ax1 = plt.subplots()
fig.suptitle(trackName)
iterations = np.arange(0, testData.shape[0], 1) 
speedXsin = -abs(np.multiply(testData["speedX"], np.sin(testData["angle"])))
speedXcos = abs(np.multiply(testData["speedX"], np.cos(testData["angle"])))
speedXtrackPos = abs(np.multiply(testData["speedX"], testData["trackPos"]))

# Label graph axes 
ax1.set_xlabel('Iteration #')
ax1.set_ylabel('Reward Value')

# Plot multiple curves (one per training model) 
transversal = ax1.plot(iterations, speedXsin, 'b-', label = 'speedX * sin(angle) (Transversal vel.)')
parallel = ax1.plot(iterations, speedXcos, 'r-', label = '-speedX * cos(angle) (-Parallel vel.)')
position = ax1.plot(iterations, speedXtrackPos, 'g-', label = 'speedX * trackPos (Weighted Track Position)')
total = ax1.plot(iterations, testData['reward'], 'b-', label = "Total Reward")

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show() 
