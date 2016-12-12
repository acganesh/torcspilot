import matplotlib.pyplot as plt
import numpy as np

testData = np.genfromtxt('test_experiment_log.csv', delimiter=',', skip_header=1,
                     skip_footer=0, names=["time","focus","speedX","speedY","speedZ","angle","damage","opponents","rpm","track","trackPos","wheelSpinVel","steering","accel","brake","reward","loss"])

fig, ax1 = plt.subplots()
iterations = np.arange(0, testData.shape[0], 1) 
loss = testData["loss"]

# Label graph axes 
ax1.set_xlabel('iteration #')
ax1.set_ylabel('loss')

# Plot multiple curves (one per training model) 
ax1.plot(iterations, loss, 'b-')
ax1.plot([1,2,3,4], [7,3,4,6], 'r-')

plt.show() 
