from math import e
import numpy as np
import random
import time
from collections import namedtuple
from environment import Environment

random.seed(1)

EpisodeStats = namedtuple('Stats',['episode_lengths', 'episode_rewards'])

class QLearningAgent(object):

    """ Klasa u kojoj je implementran Q-Learning algoritam iz oblasti Reinforcement Learning-a. """

    def __init__(self, num_episodes: int=100, max_iter: int=10000, gamma: float=0.9, alpha: float=None, epsilon: float=0.6, random_state: int=1) -> None:

        """ 
            Konstruktor klase.

            :params:
                - num_episodes: broj epizoda za treniranje,
                - max_iter: maksimalan broj iteracija u toku jedne epizode
                - gamma: faktor umanjenja buducih nagrada
                - alpha: konstanta ucenja
                - epsilon: verovatnoca uzorkovanja slucajnih akcija
                - random_state: seed za random funkcije

            :return:
                - None
        """

        #  definisanje ukupnog broja epizoda
        self.num_episodes = num_episodes

        # defiisanje maksimalnog broja itearcija u okviru jedne epizode
        self.max_iter = max_iter

        # definisanje faktora umanjenja
        self.gamma = gamma

        # definisanje konstante ucenja
        self.alpha = alpha

        # postavaljenje flag-a ukoliko je konstanta ucenja prosledjena konstruktoru, tj. konstantna je
        self.alpha_flag = True if self.alpha is not None else False

        # definisanje epsilon verovatnoce za istrazivanje
        self.epsilon = epsilon

        # inicijalizacija strukture za cuvanje statistika tokom treniranja
        self.train_episode_stats = EpisodeStats(episode_lengths = np.zeros(self.num_episodes), episode_rewards = np.zeros(self.num_episodes)) 

        # inicijalizacija strukture za cuvanje statistika tokom evaluacije
        self.eval_episode_stats = EpisodeStats(episode_lengths = np.zeros(self.num_episodes), episode_rewards = np.zeros(self.num_episodes)) 

        # inicijalizacija okruzenja za Q-Learning agenta
        self.env = Environment()

        # dohvatanje svih mogucih stanja iz okruzenja
        self.states = self.env.get_state_space()

        # dohvatanje svih mogucih akcija iz okruzenja
        self.actions = self.env.get_action_space()

        # inicijalizacija Q-tabele
        self.Q_table = {state: {action: 0.0 for action in self.actions} for state in self.states}

        # seed-ovanje random funkcija
        self.random_state = random_state
        random.seed(self.random_state)


    def reset(self, num_episodes: int=100, max_iter: int=10000, gamma: float=0.9, alpha: float=None, epsilon: float=0.6) -> None:

        """ 
            Funkcija za resetovanje agenta sa novim, prosledjenim parameterima.

            :params:
                - num_episodes: broj epizoda za treniranje,
                - max_iter: maksimalan broj iteracija u toku jedne epizode
                - gamma: faktor umanjenja buducih nagrada
                - alpha: konstanta ucenja
                - epsilon: verovatnoca uzorkovanja slucajnih akcija

            :return:
                - None
        """

        #  definisanje ukupnog broja epizoda
        self.num_episodes = num_episodes

        # defiisanje maksimalnog broja itearcija u okviru jedne epizode
        self.max_iter = max_iter

        # definisanje faktora umanjenja
        self.gamma = gamma

        # definisanje konstante ucenja
        self.alpha = alpha

        # postavaljenje flag-a ukoliko je konstanta ucenja prosledjena konstruktoru, tj. konstantna je
        self.alpha_flag = True if self.alpha is not None else False

        # definisanje epsilon verovatnoce za istrazivanje
        self.epsilon = epsilon

        # inicijalizacija strukture za cuvanje statistika tokom treniranja
        self.train_episode_stats = EpisodeStats(episode_lengths = np.zeros(self.num_episodes), episode_rewards = np.zeros(self.num_episodes)) 

        # inicijalizacija strukture za cuvanje statistika tokom evaluacije
        self.eval_episode_stats = EpisodeStats(episode_lengths = np.zeros(self.num_episodes), episode_rewards = np.zeros(self.num_episodes)) 

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
        random.seed(time.time())

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
    
                # azuriranje statistika ucenja
                self.train_episode_stats.episode_rewards[episode] += reward
                self.train_episode_stats.episode_lengths[episode] = i
                
                # azuriranje Q-tabele
                greedy_action = max(self.Q_table[next_state], key=self.Q_table[next_state].get)
                self.Q_table[state][action] += self.alpha * (reward + self.gamma * self.Q_table[next_state][greedy_action] - self.Q_table[state][action])
    
                # terminiranje epizode ukoliko je okruzenje reklo da je agent dosao u terminalno stanje 
                if done:
                    print(info)
                    break

                # tranzicija iz prehodnog u novo stanje
                state = next_state

        #plot_stats()


    def evaluate(self, n_episodes: int=10) -> None:

        """ 
            Funkcija za evaluaciju agent nad n_episodes u okruzenju.

            :params:
                - n_episodes: broj epizoda za evaluaciju

            :return:
                - None
        """

        print('------------------------ EVALUACIJA AGENTA ------------------------')

        # iteriranje po epizodama
        for episode in range(n_episodes):

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
                    print(info)
                    break

                # tranzija stanja
                state = next_state

        #plot_stats(train=False)


    def plot_stats(self, train: bool=True):

        """ 
            Funkcija za iscrtavanje potrebnih grafika.

            :params:
                - train: indikator, da li je treniranje ili evaluacija u pitanju

            :return:
                - None
        """

        pass


agent = QLearningAgent()
agent.train()
agent.evaluate()

