	def create_sample(self):    # softmax, Duan
		with self.g2.as_default():
			# Pick action given state ->   action = argmax( qnet(state) )
			with tf.name_scope("pick_action"):
				self.state = tf.placeholder(tf.float32, (None,)+self.state_size , name="state")
				self.q_values = tf.identity(self.qnet(self.state) , name="q_values")	# input state and get q_value
				self.predicted_actions = tf.argmax(self.q_values, dimension=1 , name="predicted_actions")
				self.target_q_values = tf.placeholder(tf.float32, (32,18), name='target_values')
				# tf.summary.histogram("Q values", tf.reduce_mean(tf.reduce_max(self.q_values, 1))) # save max q-values to track learning

			# Gradient descent
			with tf.name_scope("optimization_step"):
				self.y = tf.nn.softmax(logits=self.target_q_values/0.1)
				self.s = tf.nn.softmax(logits=self.q_values)
				self.loss = tf.reduce_sum(self.y * (tf.log(self.y)-tf.log(self.s)))

				qnet_gradients = self.qnet_optimizer.compute_gradients(self.loss, self.qnet.variables())
				for i, (grad, var) in enumerate(qnet_gradients):
					if grad is not None:
						qnet_gradients[i] = (tf.clip_by_norm(grad, 10), var)
				self.qnet_optimize = self.qnet_optimizer.apply_gradients(qnet_gradients)
				self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

			with tf.name_scope("target_network_update"):
				self.hard_copy_to_target = DQN.copy_to_target_network(self.qnet, self.target_qnet)
			self.summarize = tf.summary.merge_all()
		size = 10000
		area = 100
		experience = [None] * size  # deque(maxlen = self.memory_size)
		# replay_buffer1 = deque()
		i = 0
		while i <= size:
			try:
				# replay_buffer1.append(pickle.load(f1))
				# minibatch = replay_buffer1.popleft()

				state_value = stu[i]
				state_value =state_value[5]
				state_value = state_value.reshape(np.size(state_value))
				experience[i] = max(state_value)
				i = i + 1
			except:
				i = size + 100

		state = np.array(experience, dtype=float)
		print "Max of state:", max(state), "Min of state:", min(state)
		max_n = max(state)
		min_n = min(state)

		space = (max_n - min_n) / area
		bar = [0.0] * (area + 1)
		for i in xrange(len(state)):
			# print state[i]
			x = (state[i] - min_n) / space
			x = int(x)
			# print x
			bar[x] += 1
		# print 'student state distribution',sum(bar), bar
		return bar

	def regression(self, tea_bar):
		minibatch = self.experience_replay.buffer()
		stu_bar = self.Cal_stu(minibatch)
		choosed = self.Sort_Index(stu_bar, tea_bar)
		replay = deque()
		for i in range(50000):
			temp = self.experience[i]
			state_value = temp[5]
			state_value = state_value.reshape(np.size(state_value))

			x = (max(state_value) - self.min_n) / self.space
			x = int(x)

			if x in choosed:
				replay.append(temp)
		print 'length of regression replay memory:',len(replay)

		counter = 0
		while counter <= 1000:
			samples_index = np.floor(np.random.random((32,))*1000)
			samples = [replay[int(i)] for i in samples_index]

			# Build the bach states
			batch_states = np.asarray( [d[0] for d in samples] )  #observation
			value_batch = np.asanyarray([d[5] for d in samples])    # q_values from teacher, regression target
			teacher = value_batch.reshape(32, 18)

			# Perform training
			#----------------------------------------------------------------------------------
			self.session.run(self.optimizer, {self.state: batch_states, self.target_q_values_reg: teacher})
			counter = counter+1
		self.num_training_steps += 1000
		
	def restore(self, Pkl):
		# self.experience = [None] * 10000  # deque(maxlen = self.memory_size)
		f1 = open('/home/anny/Desktop/DQN-master-Ori/' + Pkl)
		replay_buffer1 = deque()
		i = 0
		while i <= tea_size:
			try:
				replay_buffer1.append(pickle.load(f1))
				minibatch = replay_buffer1.popleft()

				# state_value = minibatch[5]
				# state_value = state_value.reshape(np.size(state_value))
				# state_value = np.hstack([state_value, np.zeros([18 - np.size(state_value)])])
				# state_value = state_value.reshape(1, 18)
				# minibatch[5] = state_value

				self.experience[i] = minibatch
				i = i + 1
			except:
				i = tea_size + 100
				f1.close()
		print 'length of self.experience:', len(self.experience)  #self.experience is ok

	def Cal_bar(self):
		area = 100
		# experience = [None] * tea_size  # deque(maxlen = self.memory_size)
		# replay_buffer1 = deque()
		i = 0
		while i <= tea_size:
			try:
				# replay_buffer1.append(pickle.load(f1))
				# minibatch = replay_buffer1.popleft()
				temp = self.experience[i]
				state_value = temp[5]
				state_value = state_value.reshape(np.size(state_value))
				self.statevalues[i] = max(state_value)
				i = i + 1
			except:
				i = tea_size + 100

		state = np.array(self.statevalues, dtype=float)
		# print "Max of state:", max(state), "Min of state:", min(state)
		self.max_n = max(state)
		self.min_n = min(state)

		self.space = (self.max_n - self.min_n) / area
		bar = [0.0] * (area + 1)
		for i in xrange(len(state)):
			x = (state[i] - self.min_n) / self.space
			x = int(x)
			# print x
			bar[x] += 1
		print 'Teacher state distribution',sum(bar), bar
		return bar

	def Cal_stu(self, stu):
		size = 10000
		area = 100
		experience = [None] * size  # deque(maxlen = self.memory_size)
		# replay_buffer1 = deque()
		i = 0
		while i <= size:
			try:
				# replay_buffer1.append(pickle.load(f1))
				# minibatch = replay_buffer1.popleft()

				state_value = stu[i]
				state_value =state_value[5]
				state_value = state_value.reshape(np.size(state_value))
				experience[i] = max(state_value)
				i = i + 1
			except:
				i = size + 100

		state = np.array(experience, dtype=float)
		print "Max of state:", max(state), "Min of state:", min(state)
		max_n = max(state)
		min_n = min(state)

		space = (max_n - min_n) / area
		bar = [0.0] * (area + 1)
		for i in xrange(len(state)):
			# print state[i]
			x = (state[i] - min_n) / space
			x = int(x)
			# print x
			bar[x] += 1
		# print 'student state distribution',sum(bar), bar
		return bar

	def Sort_Index(self, Stu, Tea):
		Length = 10000
		# print sum(Seaquest)
		# print max(Seaquest),min(Seaquest)
		S = sorted(Stu)
		# print type(S)

		D = []

		for i in xrange(len(Stu)):
			# print S[i],Seaquest.index(S[i])
			Temp = Stu.index(S[i])
			D.append(Temp)
			Stu[Temp] = -1
		# print D

		Sum = []
		T = []

		for i in xrange(len(Tea)):
			x = Tea[D[i]]
			Sum.append(x)
			T.append(D[i])
			# print Tea[D[i]], D[i]
			if sum(Sum) >= Length:
				break
		print len(T), T, sum(Sum)
		return T



