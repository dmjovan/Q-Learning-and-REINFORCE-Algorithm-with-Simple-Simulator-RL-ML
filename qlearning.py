import numpy as np
import random
import time
import matplotlib.pyplot as plt
from collections import namedtuple
from environment import Environment

EpisodeStats = namedtuple('EpisodeStats',['episode_lengths', 'episode_rewards'])

class QLearningAgent(object):

    """ Klasa u kojoj je implementran Q-Learning algoritam iz oblasti Reinforcement Learning-a. """

    def __init__(self, num_episodes: int=200, max_iter: int=2000, eval_episodes: int=10, 
                 gamma: float=0.9, alpha: float=None, epsilon: float=0.6, print_infos: bool=True) -> None:

        """ 
            Konstruktor klase.

            :params:
                - num_episodes: broj epizoda za treniranje,
                - max_iter: maksimalan broj iteracija u toku jedne epizode
                - eval_episodes: broj epizoda evaluacije
                - gamma: faktor umanjenja buducih nagrada
                - alpha: konstanta ucenja
                - epsilon: verovatnoca uzorkovanja slucajnih akcija
                - print_infos: indikator za printovanje informacija na kraju epizode

            :return:
                - None
        """

        np.random.seed(1)
        random.seed(1)

        #  definisanje ukupnog broja epizoda
        self.num_episodes = num_episodes

        # defiisanje maksimalnog broja itearcija u okviru jedne epizode
        self.max_iter = max_iter

        # definisanje maksimalnog broj epizoda za evaluaciju
        self.eval_episodes = eval_episodes

        # definisanje faktora umanjenja
        self.gamma = gamma

        # definisanje konstante ucenja
        self.alpha = alpha

        # postavaljenje flag-a ukoliko je konstanta ucenja prosledjena konstruktoru, tj. konstantna je
        self.alpha_flag = True if self.alpha is not None else False

        # definisanje epsilon verovatnoce za istrazivanje
        self.epsilon = epsilon

        self.print_infos = print_infos

        # inicijalizacija strukture za cuvanje epizodnih statistika tokom treniranja
        self.train_episode_stats = EpisodeStats(episode_lengths = np.zeros(self.num_episodes), episode_rewards = np.zeros(self.num_episodes)) 

        # inicijalizacija strukture za cuvanje epizodnih statistika tokom evaluacije
        self.eval_episode_stats = EpisodeStats(episode_lengths = np.zeros(self.eval_episodes), episode_rewards = np.zeros(self.eval_episodes))

        # inicijalizacija okruzenja za Q-Learning agenta
        self.env = Environment()

        # dohvatanje svih mogucih stanja iz okruzenja
        self.states = self.env.get_state_space()

        # dohvatanje "losih" i "dobrih" terminalnih stanja
        self.bad_terminal_states, self.good_terminal_states = self.env.get_terminal_states()

        # dohvatanje neterminalnih stanja
        self.non_terminal_states = list(set(self.states) - set(self.bad_terminal_states) - set(self.good_terminal_states))
        self.non_terminal_states.sort()

        # inicijalizacija strukture za cuvanje iterativnih statistika prilikom treniranja
        self.iter_values = dict.fromkeys(self.non_terminal_states)
        for key in self.non_terminal_states:
            self.iter_values[key] = []

        # dohvatanje svih mogucih akcija iz okruzenja
        self.actions = self.env.get_action_space()

        # inicijalizacija Q-tabele
        self.Q_table = {state: {action: 0.0 for action in self.actions} for state in self.states}


    def update_alpha(self, episode: int) -> None:

        """ 
            Funkcija za azuriranje konstante ucenja, ukoliko ona nije konstruktorom definisana kao konstanta.

            :params:
                - episode: trenutna epizoda prilikom treniranja

            :return:
                - None
        """

        if not self.alpha_flag:
            self.alpha = np.log(episode + 1)/(episode + 1)


    def update_epsilon(self, episode: int) -> None:

        """ 
            Funkcija za azuriranje epsilon verovatnoce.

            :params:
                - episode: trenutna epizoda prilikom treniranja

            :return:
                - None
        """

        if episode != 0:
            self.epsilon *= max(0.0, (1 - episode/self.num_episodes))


    def policy(self, state: str, train: bool=True) -> str:

        """ 
            Funckija za uzorkovanje akcija, prema politici ili nasumicno.

            :params:
                - state:   trenutno stanje
                - train:   indikator, da li je treniranje ili evaluacija u pitanju

            :return:
                - action: izabrana akcija
        """

        if train and random.random() <= self.epsilon:
            action = random.choice(self.actions)
        else:
            action = max(self.Q_table[state], key=self.Q_table[state].get)

        return action


    def train(self) -> None:

        """ 
            Funkcija za treniranje agenta upotrebom Q-Learning algoritma.

            :params:
                - None

            :return:
                - None
        """

        print('------------------------ TRENIRANJE AGENTA ------------------------')

        # iteriranje po epizodama
        for episode in range(self.num_episodes):
           
            # resetovanje okruzenja i dohvatanje trenutnog stanja / startne pozicije
            state = self.env.reset()

            # azuriranje konstante ucenja u svakoj epizodi, ukoliko ona nije konstantna
            self.update_alpha(episode)

            # azuriranje epsilon verovatnoce u svakoj epizode
            self.update_epsilon(episode)
            
            # iteriranje po iteracijama u okviru jedne epizode
            for i in range(self.max_iter):
                
                # uzorkovanje adekvatne ili istrazivacke akcije prema sacuvanoj politici i prema trenutnom stanju
                action = self.policy(state)

                # izvrsavanje akcije u okruzenju i tranzicija stanja
                reward, next_state, done, info = self.env.step(action)
    
                # azuriranje epizodnih statistika ucenja
                self.train_episode_stats.episode_rewards[episode] += reward
                self.train_episode_stats.episode_lengths[episode] = i

                # azuriranje iterativnih statistika ucenja
                for s in self.non_terminal_states:
                    self.iter_values[s].append(max(self.Q_table[s].values()))
                
                # azuriranje Q-tabele
                greedy_action = max(self.Q_table[next_state], key=self.Q_table[next_state].get)
                self.Q_table[state][action] += self.alpha * (reward + self.gamma * self.Q_table[next_state][greedy_action] - self.Q_table[state][action])
    
                # terminiranje epizode ukoliko je okruzenje reklo da je agent dosao u terminalno stanje 
                if done:
                    if self.print_infos:
                        print(info)
                    break

                # tranzicija iz prehodnog u novo stanje
                state = next_state

            if not done and self.print_infos:
                print(f'Maksimalan broj iteracija je dostignut! Agent je zavrsio epizodu u stanju {state}')

        print('Naucena optimalna politika: ')
        self.print_policy()
        self.plot_stats()


    def evaluate(self) -> None:

        """ 
            Funkcija za evaluaciju agent nad n_episodes u okruzenju.

            :params:
                - None

            :return:
                - None
        """

        print('------------------------ EVALUACIJA AGENTA ------------------------')

        # iteriranje po epizodama
        for episode in range(self.eval_episodes):

            # resetovanje okruzenja i dohvatanje trenutnog stanja / startne pozicije
            state = self.env.reset()
            
            # iteriranje po iteracijama u okviru jedne epizode
            for i in range(self.max_iter):
                
                # uzorkovanje adekvatne akcije prema sacuvanoj politici i prema trenutnom stanju
                action = self.policy(state, train=False)
    
                # izvrsavanje akcije u okruzenju i tranzicija stanja
                reward, next_state, done, info = self.env.step(action)
    
                # azuriranje statistika ucenja
                self.eval_episode_stats.episode_rewards[episode] += reward
                self.eval_episode_stats.episode_lengths[episode] = i

                # terminacija epizode
                if done:
                    if self.print_infos:
                        print(info)
                    break

                # tranzija stanja
                state = next_state

            if not done and self.print_infos:
                print(f'Maksimalan broj iteracija je dostignut! Agent je zavrsio epizodu u stanju {state}')

        print(f'Prosecna ukupna epizodna nagrada je: {np.sum(self.eval_episode_stats.episode_rewards)/self.eval_episodes}')


    def print_policy(self) -> None:

        """
            Funkcija za ispisvanje najbolje/optimalne politike 
            u neterminalnim stanjima.

            :params:
                - None

            :return:
                - None
        """

        for state in self.non_terminal_states:
            print(f'Stanje {state} ------> Akcija {self.policy(state, train=False).upper()}')


    def plot_stats(self):

        """ 
            Funkcija za iscrtavanje potrebnih grafika.

            :params:
                - None

            :return:
                - None
        """

        if not self.alpha_flag:
            alpha_str = "volatile"
        else:
            alpha_str = str(self.alpha)

        # --------------- TRENIRANJE - EPIZODE ---------------
        fig, ax = plt.subplots(figsize=(12,10))
        plt.title(f'Treniranje agenta \nPromena ukupne nagrade po epizodama \nbr. epizoda={self.num_episodes}, max. iter={self.max_iter}, $\\gamma$={self.gamma}, $\\alpha$={alpha_str}')

        ax.plot(np.arange(0, self.num_episodes), self.train_episode_stats.episode_rewards, color='blue', marker='o')
        ax.set_xlabel('#No. epizode')
        ax.set_ylabel('Ukupna epizodna nagrada', color='blue')
        ax.grid()
        ax1 = ax.twinx()
        ax1.plot(np.arange(0, self.num_episodes), self.train_episode_stats.episode_lengths, color='red')
        ax1.axhline(self.max_iter, color='green', linestyle='-.', label='Max. Iter')
        ax1.set_ylabel('Iteracije', color='red')
        ax1.grid()
        ax1.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

        # --------------- TRENIRANJE - ITERACIJE ---------------
        plt.figure(figsize=(12,10))
        plt.title(f'Treniranje agenta \nPromena V vrednosti po iteracijama \nbr. epizoda={self.num_episodes}, max. iter={self.max_iter}, $\\gamma$={self.gamma}, $\\alpha$={alpha_str}')
        
        for state, col in zip(self.non_terminal_states, ['blue', 'red', 'green', 'gray', 'tomato']):
            plt.plot(np.arange(0,len(self.iter_values[state])), self.iter_values[state], color=col, label=f'Stanje {state}')

        plt.xlabel('#No. iteracije')
        plt.ylabel('V vrednost')
        plt.grid()
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
