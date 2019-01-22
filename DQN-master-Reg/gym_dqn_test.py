import gym
import time
import logging
import shutil
import os
import sys
import tensorflow as tf
import gc
gc.enable()
from modules.dqn import *
from modules.env_utils import *

# -----------------------------------------------------------------
# collection model: collect samples produced by teacher
# regression model: using regression methods to training an agent
# only can choose one of above two model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
collection = False
regression = True
testing = False
sozo = 25000  # define the replay memory size, it's also the num of teacher produced
ENV_NAME = 'Assault-v0'  # testing game's name
Pkl = 'Assault-v0'
cost = []  # save the cost data, 10000 training steps record once
reward_s = []  # save the rewards, 25000 training steps record once
# -----------------------------------------------------------------

def main():
	print "Usage:", '\n', "  ", sys.argv[0], " [optional: path_to_ckpt_file] [optional: True/False test mode]", '\n\n'
	outdir = "gym_results"
	TOTAL_FRAMES = 100000  ## TRAIN ori is 20 million
	MAX_TRAINING_STEPS = 20 * 60 * 60 / 3  ## MAX STEPS BEFORE RESETTING THE ENVIRONMENT
	TESTING_GAMES = 30  # no. of games to average on during testing
	MAX_TESTING_STEPS = 5 * 60 * 60 / 3  # 5 minutes  '/3' because gym repeating the last action 3-4 times already!
	TRAIN_AFTER_FRAMES = 10000    # 50000
	epoch_size = 10000  # every how many frames to test  # 50000
	MAX_NOOP_START = 30
	LOG_DIR = outdir + '/' + ENV_NAME + '/logs/'
	if os.path.isdir(LOG_DIR):
		shutil.rmtree(LOG_DIR)
	journalist = tf.summary.FileWriter(LOG_DIR)

	# Build environment
	env = AtariEnvWrapper(ENV_NAME)

	# Initialize Tensorflow session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.InteractiveSession(config=config)

	# Create DQN agent
	UseDoubleDQN = False
	agent = DQN(state_size=env.observation_space.shape,
							action_size=18,
							env_size = env.action_space.n,
							session=session,
							summary_writer=journalist,
							exploration_period=1000,    # original 1 million
							minibatch_size=32,
							discount_factor=0.99,
							experience_replay_buffer=sozo,  # original is 1000000, to make it fast, i changed to 100000
							target_qnet_update_frequency=20000,
							# 30000 if UseDoubleDQN else 10000, ## Tuned DDQN,ori is 20000,i changed 10000
							initial_exploration_epsilon=1.0,
							final_exploration_epsilon=0.1,
							reward_clipping=1.0,
							DoubleDQN=UseDoubleDQN,
							reg = regression)

	print ENV_NAME,"action space:",env.action_space.n
	session.run(tf.initialize_all_variables())
	journalist.add_graph(session.graph)
	saver = tf.train.Saver(tf.all_variables())
	logger = logging.getLogger()
	logging.disable(logging.INFO)

	# If an argument is supplied, load the specific checkpoint.
	test_mode = True   # default is False
	if collection or testing:
		saver.restore(session, outdir + "/" + 'Assault-v0' + "/final.ckpt")  #default is sys.argv[1], loading teacher model

	num_frames = 0
	num_games = 0
	current_game_frames = 0
	init_no_ops = np.random.randint(MAX_NOOP_START + 1)
	last_time = time.time()
	last_frame_count = 0.0
	state = env.reset()

	# training start
	if collection:
		TOTAL_FRAMES = sozo
	while num_frames <= TOTAL_FRAMES + 1:
		if test_mode and testing:
			env.render()

		num_frames += 1
		current_game_frames += 1

		# Pick action given current state
		if not test_mode:
			action, q_values = agent.action(state, training=True)  # added q_values
		else:
			action, q_values = agent.action(state, training=False)
		if current_game_frames < init_no_ops:
			action = 0

		# Perform the selected action on the environment
		next_state, reward, done, _ = env.step(action)

	# Store experience, for training DQN
		# if current_game_frames >= init_no_ops:
		# 	agent.store(state, action, reward, next_state, done, q_values)

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store experience into file, Duan
		if collection and current_game_frames >= init_no_ops:
			TD = agent.Cal_TD(state, action, reward, next_state, done)
			# print "TD error is:",TD, q_values
			agent.store_file(state, action, reward, next_state, done, q_values, TD)

	# sampling form teacher's memory and training model ues regression methods
		if regression:
			lost = agent.train_samples()
			cost.append(lost)
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		state = next_state

		# # Train agent
		# if num_frames >= TRAIN_AFTER_FRAMES:
		# 	agent.train()
		if done or current_game_frames > MAX_TRAINING_STEPS:
			state = env.reset()
			current_game_frames = 0
			num_games += 1
			init_no_ops = np.random.randint(MAX_NOOP_START + 1)

		# Print an update
		if num_frames % 10000 == 0:
			new_time = time.time()
			diff = new_time - last_time
			last_time = new_time
			elapsed_frames = num_frames - last_frame_count
			last_frame_count = num_frames
			print "frames: ", num_frames, "    games: ", num_games, "    speed: ", (elapsed_frames / diff), " frames/second"

		# # Save the network's parameters after every epoch
		# if num_frames % epoch_size == 0 and num_frames > TRAIN_AFTER_FRAMES:
		# 	saver.save(session, outdir + "/" + ENV_NAME + "/model_" + str(num_frames / 1000) + "k.ckpt")
		# 	print '\n', "epoch:  frames=", num_frames, "   games=", num_games

		## Testing -- it's kind of slow, so we're only going to test every 2 epochs
		if collection == False and num_frames % (2 * epoch_size) == 0 and num_frames > TRAIN_AFTER_FRAMES:
			total_reward = 0
			avg_steps = 0
			for i in xrange(TESTING_GAMES):
				state = env.reset()
				init_no_ops = np.random.randint(MAX_NOOP_START + 1)
				frm = 0
				while frm < MAX_TESTING_STEPS:
					frm += 1
					# env.render()
					action, q_values = agent.action(state, training=False)  # direct action for test
					if current_game_frames < init_no_ops:
						action = 0
					state, reward, done, _ = env.step(action)
					total_reward += reward
					if done:
						break
				avg_steps += frm
			avg_reward = float(total_reward) / TESTING_GAMES
			# str_ = session.run(tf.summary.scalar('test reward (' + str(epoch_size / 1000) + 'k)', avg_reward))
			# journalist.add_summary(str_, num_frames)  # np.round(num_frames/epoch_size)) # in no. of epochs, as in Mnih
			print '  --> Evaluation Average Reward: ', avg_reward, '   avg steps: ', (avg_steps / TESTING_GAMES)
			state = env.reset()
			reward_s.append(avg_reward)  # record avg_reward

	# graph close
	journalist.close()

	# Save the final network
	if regression:
		# fw = open(ENV_NAME+'regscore.txt','wb')
		# pickle.dump([cost,reward_s],fw,-1)
		saver.save(session, outdir + "/" + "Assault_v1" + "/final.ckpt")#+ENV_NAME
	return 0

if __name__=='__main__':
	main()


