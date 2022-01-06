import random
from numpy.lib.utils import info
random.seed(1)

class Environment(object):

	""" Klasa kojom je interpretirano i implementirano zadato okruzenje. """

	def __init__(self, bad_reward: int=-1, good_reward: int=3, random_state: int=1) -> None:

		""" 
            Konstruktor klase.

            :params:
                - bad_reward: vrednost "lose" nagrade
				- good_reward: vrednost "dobre" nagrade
				- random_state: vrednost seed-a random funkcija

            :return:
                - None
        """

		

		# definisanje svih stanja
		self.states = ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B3', 'B5']
		self.start_state = 'A1'
		self.terminal_states = ['B1', 'B3', 'B5']
		self.bad_terminal_states = ['B1', 'B3']
		self.good_terminal_states = ['B5']

		# definisanje svih akcija
		self.actions = ['up', 'down', 'left', 'right']

		# definisanje verovatnoca za preduzimanje pojedinih akcija
		self.action_probabilities = {
									 'choosen_direction': 0.6,
									 'choosen_direction_plus_90_degs': 0.2,
									 'choosen_direction_minus_90_degs': 0.2 
									}

		# definisanje mogucih izmena u kretanju
		self.get_possible_action_changes()

		# definisanje kretanja u zavisnosti od stanja i akcije
		self.get_possible_moves()

		self.random_state = random_state

		# definisanje nagrada za sva stanja
		self.populate_rewards(bad_reward, good_reward)

		# resetovanje okruzenja
		self.current_state = self.start_state
		self.steps_cnt = 0
		self.history = []

	def get_possible_action_changes(self):

		""" 
            Funkcija za mapiranje promena u akcija.

            :params:
                - None

            :return:
                - None
        """

		self.possible_action_changes = {
										'up'    : {
													 'choosen_direction'               : 'up',
													 'choosen_direction_plus_90_degs'  : 'left',
													 'choosen_direction_minus_90_degs' : 'right'
											      },
									    'down'  : {
													 'choosen_direction'               : 'down',
													 'choosen_direction_plus_90_degs'  : 'right',
													 'choosen_direction_minus_90_degs' : 'left'
											      },
									    'left'  : {
													 'choosen_direction'               : 'left',
													 'choosen_direction_plus_90_degs'  : 'down',
													 'choosen_direction_minus_90_degs' : 'up'
											      },
									    'right' : {
													 'choosen_direction'               : 'right',
													 'choosen_direction_plus_90_degs'  : 'up',
													 'choosen_direction_minus_90_degs' : 'down'
											      }
									   }

	def get_possible_moves(self) -> None:

		""" 
            Funkcija za definisanje dozvoljenih kretanja.

            :params:
                - None

            :return:
                - None
        """

		self.possible_moves = { 
								'A1': {
										'up'    : 'A1',
										'down'  : 'B1',
										'left'  : 'A1',
										'right' : 'A2'
									  },
							    'A2': {
										'up'    : 'A2',
										'down'  : 'A2',
										'left'  : 'A1',
										'right' : 'A3'
									  },
								'A3': {
										'up'    : 'A3',
										'down'  : 'B3',
										'left'  : 'A2',
										'right' : 'A4'
									  },
								'A4': {
										'up'    : 'A4',
										'down'  : 'A4',
										'left'  : 'A3',
										'right' : 'A5'
									  },	
							    'A5': {
										'up'    : 'A5',
										'down'  : 'B5',
										'left'  : 'A4',
										'right' : 'A5'
									  },  
							    'B1': {
										'up'    : 'A1',
										'down'  : 'B1',
										'left'  : 'B1',
										'right' : 'B1'
									  },
							    'B3': {
										'up'    : 'A3',
										'down'  : 'B3',
										'left'  : 'B3',
										'right' : 'B3'
									  },
							    'B5': {
										'up'    : 'A5',
										'down'  : 'B5',
										'left'  : 'B5',
										'right' : 'B5'
									  },
							   }


	def reset(self) -> str:

		""" 
            Funkcija za resetovanje varijabilnih parametera okruzenja.

            :params:
                - None

            :return:
                - current_state: pocetno/trenutno stanje agenta
        """

		self.current_state = self.start_state
		self.steps_cnt = 0
		self.history = []

		return self.current_state


	def populate_rewards(self, bad_reward: int, good_reward: int) -> None:

		""" 
            Funkcija za formiranje recnika koji sadrzi nagrade za svako stanje.

            :params:
                - bad_reward: vrednost "lose" nagrade
				- good_reward: vrednost "dobre" nagrade

            :return:
                - None
        """

		self.rewards = dict.fromkeys(self.states)

		for state in self.states:
			if state not in self.terminal_states:
				self.rewards[state] = 0

			elif state in self.bad_terminal_states:
				self.rewards[state] = bad_reward

			elif state in self.good_terminal_states:
				self.rewards[state] = good_reward

	def get_action_space(self) -> list:

		""" 
            Fukcija za dohvatanje svih akcija iz okruzenja.

            :params:
                - None

            :return:
                - actions: lista svih akcija
        """

		return self.actions

	def get_state_space(self) -> list:

		""" 
            Fukcija za dohvatanje svih stanja okruzenja.

            :params:
                - None

            :return:
                - states: lista svih stanja
        """

		return self.states

	def get_terminal_states(self) -> tuple:

		""" 
            Fukcija za dohvatanje terminalnih stanja okruzenja.

            :params:
                - None

            :return:
                - bad_terminal_states: lista "losih" terminalnih stanja
				- good_terminal_states: lista "dobrih" terminalnih stanja 
        """

		return self.bad_terminal_states, self.good_terminal_states

	def get_current_state(self) -> str:

		""" 
            Fukcija za dohvatanje trenutnog stanja.

            :params:
                - None

            :return:
                - current_state: string koji govori koje je trenutno stanje
        """

		return self.current_state

	def get_history(self) -> list:

		""" 
            Fukcija za dohvatanje istorije agenta pre resetovanja okruzenja.

            :params:
                - None

            :return:
                - history: lista sa recnicima za svaku preduzetu akciju
        """

		return self.history


	def choose_action(self, provided_action: str) -> str:

		""" 
            Fukcija za interpretaciju stohasticnosti okruzenja, slucajnim izborom mogucih akcija.

            :params:
                - provided_action: izabrana akcija od strane agenta, za koju okruzenje stoh. odlucuje 
								   da li nju da primeni ili eku drugu

            :return:
                - choosen_action: finalno izabrana akcija koja se primenjuje
        """

		random.seed(self.random_state)

		choosen = random.choices(list(self.action_probabilities.keys()), weights=list(self.action_probabilities.values()), k=1)[0]
		choosen_action = self.possible_action_changes[provided_action][choosen]

		return choosen_action


	def move(self, action: str) -> tuple:

		""" 
            Funkcija za pomeranje trenutnog stanja u zavisnosti 
			od trenutnog stanja i akcije.

            :params:
                - action: izabrana akcija od strane agenta i okruzenja

            :return:
                - reward: nagrada za akciju
				- new_state: novo stanje u koje prelazi agent
				- previous_state: prethodno stanje
				- done: indikator, da li je presao u terminalno stanje
				- info: informacija o njegovom stanju
        """

		previous_state = self.current_state
		self.current_state = self.possible_moves[self.current_state][action]
		new_state = self.current_state

		reward = self.rewards[new_state]
		done = True if new_state in self.terminal_states else False
		
		if done and new_state in self.bad_terminal_states:
			info = f'Nazalost, agent je zavrsio epizodu u stanju {new_state} :('
		elif done and new_state in self.good_terminal_states:
			info = f'Bravo! Agent je zavrsio epizodu u stanju {new_state} :)'
		else:
			info = None

		self.steps_cnt += 1

		return reward, new_state, previous_state, done, info


	def step(self, action: str) -> tuple:

		""" 
            Funkcija za interakciju sa okruzenjem, tj. preduzimanjem akcije.

            :params:
                - action: ackija koju je agent izabrao kao najadekvatniju

            :return:
                - reward: nagrada za akciju
				- new_state: novo stanje u koje prelazi agent
				- done: indikator, da li je presao u terminalno stanje
				- info: informacija o njegovom stanju
        """

		if action.lower() in self.actions:

			choosen_action = self.choose_action(action.lower())
				
			reward, new_state, prev_state, done, info = self.move(choosen_action)

			self.history.append({
								 'steps_taken'     : self.steps_cnt,
								 'previous_state'  : prev_state,
								 'provided_action' : action.lower(),
								 'choosen_action'  : choosen_action,
								 'new_state'       : new_state
								})

		return  reward, new_state, done, info
