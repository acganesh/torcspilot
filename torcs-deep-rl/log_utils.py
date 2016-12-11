import logging
import datetime
import os
import collections
import numpy as np

EXPERIMENTS_PATH = "./experiments/"

class TORCS_ExperimentLogger:
    """
    Logs TORCS experiments, given observation, action, reward, and loss.
    """
    def __init__(self, experiment_name):
        directory = "%s%s/" % (EXPERIMENTS_PATH, experiment_name)

        if not os.path.exists(directory):
            os.makedirs(directory)
        """
        else:
            print 'Experiment already exists!'
            assert(False)
        """

        self.logname = '%s%s_log.csv' % (directory, experiment_name)
        logging.basicConfig(filename=self.logname,
                            filemode='a',
                            format='%(asctime)s,%(message)s',
                            datefmt='%m-%d-%Y;%H:%M:%S',
                            level=logging.INFO)

        self.logger = logging.getLogger("TORCS Experiment Logger")
        self.logger.setLevel(logging.INFO)
        self.header_needed = True

    def log(self, observation, action, reward, loss):
        if self.header_needed:
            with open(self.logname, 'w') as f:
                f.write("time,speedX,speedY,speedZ,angle,damage,rpm,trackPos,steering,accel,brake,reward,loss,\n")
            self.header_needed = False


        # Keys in gym_torcs Observation object
        # Removed track and oppponents for now
        keys = ('speedX', 'speedY', 'speedZ', 'angle', 'damage', 'rpm', 'trackPos')

        # Log should contain keys, actions, reward, and loss
        length = len(keys) + len(action) + 2

        vals = np.concatenate(([getattr(observation, k) for k in keys], action, [reward, loss]))
        msg = ('%s,'*length) % tuple(vals)
        with open(self.logname, 'a') as f:
            time=datetime.datetime.now().strftime("%H:%M;%m-%d-%y")
            f.write(time+',')
            f.write(msg)
            f.write('\n')
        #self.logger.log(logging.INFO, msg)


def logger_example():
    """
    Basic example that demonstrates how to use TORCS_ExperimentLogger.
    """
    exp_logger = TORCS_ExperimentLogger("test_experiment2") 
    names = ['focus',
             'speedX', 'speedY', 'speedZ', 'angle', 'damage',
             'opponents',
             'rpm',
             'track', 
             'trackPos',
             'wheelSpinVel']
    Observation = collections.namedtuple('Observation', names)
    obs = Observation(focus=np.array(1.0, dtype=np.float32),
                       speedX=np.array(2.0, dtype=np.float32),
                       speedY=np.array(3.0, dtype=np.float32),
                       speedZ=np.array(4.0, dtype=np.float32),
                       angle=np.array(5.0, dtype=np.float32),
                       damage=np.array(6.0, dtype=np.float32),
                       opponents=np.array(7.0, dtype=np.float32),
                       rpm=np.array(8.0, dtype=np.float32),
                       track=np.array(9.0, dtype=np.float32),
                       trackPos=np.array(10.0, dtype=np.float32),
                       wheelSpinVel=np.array(11.0, dtype=np.float32))
    action = np.array([1.0, 2.0, 3.0])
    reward = 1000.0
    loss = 0.10
    for _ in xrange(10):
        logger.log(obs, action, reward, loss)

if __name__ == '__main__':
    logger_example()
