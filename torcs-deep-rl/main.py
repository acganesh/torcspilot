# Train the network, using the deep deterministic policy gradients algorithm
# Implementation adapted from Ben Lau https://github.com/yanpanlau/DDPG-Keras-Torcs

import tensorflow as tf
import numpy as np
from keras import backend as K
import json
import traceback
import sys
import random

from gym_torcs import TorcsEnv

from exploration import OrnsteinUhlenbeck
from models import ActorFCNet, CriticFCNet
from rewards import lng_trans
from replay_buffer import ReplayBuffer
from log_utils import TORCS_ExperimentLogger

EXPERIMENTS_PATH = "./experiments/"
OU = OrnsteinUhlenbeck()

def main(is_training):
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001
    LRA = 0.0001
    LRC = 0.001

    action_dim = 3  # Steering / Acceleration / Blake
    state_dim = 29  # Dimension of sensor inputs

    np.random.seed(42)

    vision = False
    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 100000
    done = False
    step = 0
    epsilon = 1

    exp_logger = TORCS_ExperimentLogger("train_convergence_12-10-16")

    # TensorFlow GPU
    config = tf.ConfigProto()
    # Not sure if this is really necessary, since we only have a single GPU
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    K.set_session(sess)

    actor = ActorFCNet(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticFCNet(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)

    env = TorcsEnv(vision=vision, throttle=True, gear_change=False)

    # Weight loading
    if not is_training:
        try:
            actor.model.load_weights("%s/starter_weights/actormodel.h5" % EXPERIMENTS_PATH)
            critic.model.load_weights("%s/starter_weights/criticmodel.h5" % EXPERIMENTS_PATH)
            actor.target_model.load_weights("%s/starter_weights/actormodel.h5" % EXPERIMENTS_PATH)
            critic.target_model.load_weights("%s/starter_weights/criticmodel.h5" % EXPERIMENTS_PATH)
            print "Weights loaded successfully"
        except:
            print "Error in loading weights"
            print '-'*60
            traceback.print_exc(file=sys.stdout)
            print '-'*60
            assert(False)

    for i in xrange(episode_count):
        print "Episode: %i; Replay Buffer: %i" % (i, buff.count())

        if np.mod(i, 3) == 0:
            # Relaunch TORCS every 3 episodes; memory leak error
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()
    
        state_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        total_reward = 0.
        # Compute rewards
        for j in xrange(max_steps):
            loss = 0
            epsilon -= 2.0 / EXPLORE # exploration factor
            action_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            action_t_raw = actor.model.predict(state_t.reshape(1, state_t.shape[0])) # this call to reshape seems suboptimal

            noise_t[0][0] = is_training * max(epsilon, 0) * OU.run(action_t_raw[0][0], 0.0, 0.60, 0.30)
            noise_t[0][1] = is_training * max(epsilon, 0) * OU.run(action_t_raw[0][1], 0.5, 1.00, 0.10)
            noise_t[0][2] = is_training * max(epsilon, 0) * OU.run(action_t_raw[0][2], -0.1, 1.00, 0.05)

            # stochastic brake
            if random.random() <= 0.1:
                noise_t[0][2] = is_training * max(epsilon, 0) * OU.run(action_t_raw[0][2], 0.2, 1.00, 0.10)
 

            # May be able to do this a bit more concisely with NumPy vectorization
            action_t[0][0] = action_t_raw[0][0] + noise_t[0][0]
            action_t[0][1] = action_t_raw[0][1] + noise_t[0][1]
            action_t[0][2] = action_t_raw[0][2] + noise_t[0][2]

            # Raw_reward_t is the raw reward computed by the gym_torcs script.
            # We will compute our own reward metric from the ob object 
            ob, raw_reward_t, done, info = env.step(action_t[0])

            state_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
            reward_t = lng_trans(ob)

            buff.add(state_t, action_t[0], reward_t, state_t1, done)  # Add replay buffer

            # Batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            done_indicators = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            # Can't we just use BATCH_SIZE here
            for k in xrange(len(batch)):
                if done_indicators[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]

            if (is_training):
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.train_target_net()
                critic.train_target_net()


            exp_logger.log(ob, action_t[0], reward_t, loss) 

            total_reward += reward_t
            state_t = state_t1

            print("Episode", i, "Step", step, "Action", action_t, "Reward", reward_t, "Loss", loss)
        
            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if (is_training):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    main(is_training=1) 
