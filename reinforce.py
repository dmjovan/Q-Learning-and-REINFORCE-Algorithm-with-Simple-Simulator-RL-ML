import numpy as np
import random
import time
import matplotlib.pyplot as plt
from environment import Environment


class ReinforceAgent(object):

    """ Klasa u kojoj je implementran REINFORCE algoritam iz oblasti Reinforcement Learning-a. """

    def __init__(self, num_episodes: int=200, max_iter: int=2000, eval_episodes: int=10, 
                 gamma: float=0.9, alpha: float=None) -> None:

        """ 
            Konstruktor klase.

            :params:
                - num_episodes: broj epizoda za treniranje,
                - max_iter: maksimalan broj iteracija u toku jedne epizode
                - eval_episodes: broj epizoda evaluacije
                - gamma: faktor umanjenja buducih nagrada
                - alpha: konstanta ucenja

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

        # inicijalizacija okruzenja za Q-Learning agenta
        self.env = Environment()

        # dohvatanje svih mogucih stanja iz okruzenja
        self.states = self.env.get_state_space()

        # dohvatanje "losih" i "dobrih" terminalnih stanja
        self.bad_terminal_states, self.good_terminal_states = self.env.get_terminal_states()

        # dohvatanje neterminalnih stanja
        self.non_terminal_states = list(set(self.states) - set(self.bad_terminal_states) - set(self.good_terminal_states))
        self.non_terminal_states.sort()

        # dohvatanje svih mogucih akcija iz okruzenja
        self.actions = self.env.get_action_space()

        # uzorkovanje jednog primera obelezja, radi dohvatanja dimenzija parametera
        sample_features = self.get_features((self.env.start_state, random.choice(self.actions), None))

        # dohvatanje broja obelezja
        self.n_features = sample_features.shape[0]

        # slucajna inicijalizacija parametara theta za procenu kriterijumske funkcije
        self.theta = np.random.normal(size=sample_features.shape)*0.1


    def get_features(self, recorded_tuple: tuple) -> np.ndarray:

        """ 
            Funkcija za generisanje obelezja na osnovu sacuvane "trojke" prilikom interakcije
            za okruzenjem.

            :params:
                - recorded_tuple: sacuvana trojka (st, at, rt)

            :return:
                - features: generisana obelezja 
        """

        # raspakovavanje 
        state, action, _ = recorded_tuple

        """ 
            Obelezja su posmatrana kao binarna, odnosno n-arna (n=3), kako ne bi imalo
            potrebe za selektivnom standardizacijom obelezja.
        """

        # dohvatanje sledeceg stanja prema kojem tezi agent
        next_state = self.env.possible_moves[state][action]

        # Prvo obelezje: Da li bi se kretao prema cilju ako u stanju 'state' primeni akciju 'action' i pomeri se u 'next_state'?
        f1 = 0 # ne bi se kretao se prema cilju (stoji ili odlazi od cilja)
        if self.manhattan_distance(state, self.good_terminal_states[0]) - \
                self.manhattan_distance(next_state, self.good_terminal_states[0]) > 0:
            f1 = 1 # kretao bi se prema cilju 

        # Drugo obelezje: Da li bi osvajio nagradu ako u stanju 'state' primeni akciju 'action' i pomeri se u terminalno stanje
        f2 = 0 # ne bi osvojio nagradu
        if next_state in self.good_terminal_states:
            f2 = 1 # osvojio bi nagradu

        # Trece obelezje: Da li bi usao u "lose" terminalno stanje ako u stanju 'state' primeni akciju 'action' 
        f3 = 0 # ne bi usao
        if next_state in self.bad_terminal_states:
            f3 = 1 # usao bi

        # generisana obelezja
        features = np.array([f1,f2,f3]).reshape(-1,1)

        return features


    def manhattan_distance(self, state1: str, state2: str) -> int:

        """ 
            Funkcija za racunanje Menhetn distance izmedju dva stanja.

            :params:
                - state1: prvo stanje, npr. 'B1'
                - state2: drugo stanje, npr. 'B3'

            :return:
                - distance: preracunata distanca
        """

        # dohvatanje koordinata stanja 1
        letter1 = state1[0]
        number1 = state1[1]

        # dohvatanje koordinata stanja 2
        letter2 = state2[0]
        number2 = state2[1]

        # racunanje ukupnog rastojanja
        distance = abs(ord(letter1.upper()) - ord(letter2.upper())) + abs(int(number1) - int(number2))

        return distance


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

    def policy(self, state: str = None, best: bool=False) -> str:

        """
            Funkcija za uzorkovanje akcije na osnovu Softmax raspodele za politiku
            ili uzorkovanjem najbolje akcije.

            :params:
                - state: trenutno stanje
                - best: indikator za uzorkovanje najbolje akcije

            :return:
                - action: uzorkovana akcija

        """
        
        # raspodela verovatnoce izbora akcije
        psd = self.policy_softmax(state)

        # uzorkovanje najbolje akcije
        if best:
            action = self.actions[np.argmax(psd)]
        else:
            # nasumicno uzorkovanje akcije, respektivno prema raspodeli
            action = random.choices(self.actions, psd, k=1)[0]

        return action


    def policy_softmax(self, state: str) -> list:

        """
            Funkcija za racunanje Softmax raspodele (kako je problem sa diskretnim akcijama)
            za politiku zaa trenutno stanje.

            :params:
                - state: trenutno stanje

            :return:
                - psd: policy softmax distribution 

        """

        # formiranje softmax verovatnoca za svaku akciju u trenutnom stanju
        psd = [np.exp(self.theta.T @ self.get_features((state, action, None))).item() for action in self.actions]

        # skaliranje verovatnoca da u zbiru cine 1
        psd = [pr / sum(psd) for pr in psd]

        return psd


    def run_and_record_episode(self) -> None:

        """ 
            Funkcija za pokretanje 1 epizode i cuvanje tranzicija.

            :params:
                - None

            :return:
                - None
        """

        # inicijalizacija memorije za cuvanje tranzicija
        self.buffer = []

        # resetovanje okruzenja i dohvatanje trenutnog stanja / startne pozicije
        state = self.env.reset()
        
        # iteriranje po iteracijama u okviru jedne epizode
        for i in range(self.max_iter):
            
            # uzorkovanje adekvatne ili istrazivacke akcije prema sacuvanoj politici i prema trenutnom stanju
            action = self.policy(state)

            # izvrsavanje akcije u okruzenju i tranzicija stanja
            reward, next_state, done, info = self.env.step(action)

             # cuvanje tranzicija
            self.buffer.append((state, action, reward))

            # terminiranje epizode ukoliko je okruzenje reklo da je agent dosao u terminalno stanje 
            if done:
                # print(info)
                break

            # tranzicija iz prehodnog u novo stanje
            state = next_state


    def evaluate(self) -> tuple:

        """ 
            Funkcija za pokretanje evaluacije nakon par epizoda treniranja.

            :params:
                - None

            :return:
                - avg_reward: procesna ukupna nagrada
                - policy_params: parameteri politike
        """

        # inicijalizacija ukupne nagrade
        avg_reward = 0

        # iteriranje evaluacije
        for ep in range(self.eval_episodes):

            # pokretanje jedne epizode
            self.run_and_record_episode()

            # dohvatanje ukupne epizodne nagrade
            avg_reward += np.sum(np.array([rec[2] for rec in (self.buffer)]))

        avg_reward /= self.eval_episodes
        print(f'Prosecna ukupna nagrada u toku jedne epizode je {avg_reward}')

        # dohvatanje parametara politike
        policy_params = np.array([self.policy_softmax(state) for state in self.non_terminal_states])

        return avg_reward, policy_params

    
    def score(self, state: str, action: str):

        """
            Funkcija za racunanje Softmax skora, gradijenta logaritma politike.

            :params:
                - state: trenutno stanje
                - action: akcija izabrana u trenutnom stanju

            :return:
                - score: vrednost skora
        """

        # racunanje svih obelezja -> za trenutno stanje i sve moguce akcije
        features = [self.get_features((state, action_, None)) for action_ in self.actions]

        # racunanje softmax verovatnoca politike 
        psd = self.policy_softmax(state)

        # racunanje skora
        score = self.get_features((state, action, None)) - np.sum(np.array(psd).reshape(1,-1) @ np.array(features).reshape(len(self.actions), self.n_features))

        return score


    def update_params(self, episode: int) -> None:

        """ 
            Funkcija za azuriranje parametera theta.

            :params:
                - episode: trenutna epizoda treniranja

            :return:
                - None
        """


        def calculate_earnings(t: int, T: int):

            """ 
                Funkcija za racunanje ukupne diskontovane nagrade od trenutka t
                pa do kraja epizode.

                :params:
                    - t: trenutna iteracija
                    - T: poslednja iteracija

                :return:
                    - earnings: ukupna zarada 
            """

            gammas = self.gamma ** np.arange(0, T-t)
            rewards = np.array([rec[2] for i, rec in enumerate(self.buffer)  if i >= t])

            assert gammas.size == rewards.size, 'Racunanje zarade invalidno!'

            earnings = gammas.reshape(1,-1) @ rewards.reshape(-1,1)
            return earnings

        # azuriranje konstante ucenja u svakoj epizodi, ukoliko ona nije konstantna
        self.update_alpha(episode)

        # broj iteracija u epizodi
        T = len(self.buffer)

        # iteriranje po svih iteracijama iz epizode
        for t in range(T):

            # racunanje skora
            score_ = self.score(self.buffer[t][0], self.buffer[t][1])

            # racunanje ukupne diskontovane nagrade u trenutku t (zarada)
            vt = calculate_earnings(t, T)

            # azuriranje parametera
            self.theta += self.alpha * score_ * vt

    
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
            print(f'Stanje {state} ------> Akcija {self.policy(state, best=True).upper()}')

    
    def learn(self, episode: int) -> None:

        """
            Funkcija za jedno-epizodno ucenje parametara i agenta.

            :params:
                - episode: trenutna epizoda

            :return:
                - None
        """

        # ucenje agenta
        self.run_and_record_episode()
        
        # azuriranje parametara
        self.update_params(episode)


    def do_all(self, num_slices: int=4) -> None:

        """
            Funkcija za izvrsavanje svega - treniranje, testiranje (evaluaciju), iscrtavanje itd...

            :params:
                - num_slices: broj podela ukupnog broj epizoda

            :return:
                - None
        """

        episode_slices = [ (i*(self.num_episodes // num_slices), (i+1)*(self.num_episodes // num_slices)) for i in range(num_slices)] 

        # inicijalizacija listi
        avg_rewards, params = [], []

        # iteriranje po slice-ovima epizoda
        for ep_slice in episode_slices:

            # epizodno ucenje
            for episode in range(ep_slice[0], ep_slice[1]):
                print(f'Izvrsavanje epizode {episode}/{self.num_episodes}', end='\r')
                self.learn(episode)

             # ispisivanje optimalne politike koju je agent naucio
            self.print_policy()
            
            # testiranje agenta
            avg_reward, policy_params = self.evaluate()

            # cuvanje rezultata
            avg_rewards.append(avg_reward)
            params.append(policy_params)


        fig = plt.figure(figsize=(10,8))
        plt.plot([sl[1] for sl in episode_slices], avg_rewards)
        plt.title('Promena prosecne ukupne nagrade')
        plt.ylabel('Prosecna ukupna nagrada')
        plt.xlabel('#No. epizoda')
        plt.grid()
        plt.show()

        params = np.array(params)
        
        fig, axes = plt.subplots(ncols=len(self.non_terminal_states), figsize=(40,10))
        ax = axes.ravel()
        action_colors = ['blue', 'red', 'green', 'gray']
        for i, state in enumerate(self.non_terminal_states):
            for j, action in enumerate(self.actions):
                ax[i].plot([sl[1] for sl in episode_slices], params[:,i,j], color=action_colors[j], label=f"$\\theta\_{state}\_{action}$")
                ax[i].set_title(f'Stanje {state}')
                ax[i].legend()
                ax[i].set_xlabel('#No. epizode')
                ax[i].set_ylabel('Vrednosti parametara')
            ax[i].grid()
        plt.show()
        plt.tight_layout()


# agent = ReinforceAgent()
# agent.do_all()

# agent = ReinforceAgent(alpha=0.6)
# agent.do_all()

# agent = ReinforceAgent(alpha=0.2)
# agent.do_all()
