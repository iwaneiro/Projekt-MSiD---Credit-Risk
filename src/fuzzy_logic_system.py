import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class CreditRiskFuzzySystem:
    def __init__(self, num_attributes=2, num_mf=3, mf_type='trimf'):
        """
        num_attributes: 2 (wiek, kwota) lub 3 (wiek, kwota, czas trwania)
        num_mf: liczba przedziałów (np. 3, 5, 7)
        mf_type: rodzaj funkcji przynależności ('trimf', 'trapmf', 'gaussmf')
        """
        self.num_attributes = num_attributes
        self.num_mf = num_mf
        self.mf_type = mf_type

        # 1. Inicjalizacja zmiennych wejściowych (Antecedents)
        self.age = ctrl.Antecedent(np.arange(18, 101, 1), 'age')
        self.amount = ctrl.Antecedent(np.arange(200, 20001, 100), 'credit_amount')

        # Jeśli wybrano 3 atrybuty, dodajemy czas trwania kredytu w miesiącach (duration)
        if self.num_attributes >= 3:
            self.duration = ctrl.Antecedent(np.arange(4, 73, 1), 'duration')

            # Wyjście: ryzyko (Consequent)
        self.risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

        # 2. i 3. Definiowanie funkcji i reguł
        self._setup_membership_functions()
        self._build_rules()

        self.fuzzy_ctrl = ctrl.ControlSystem(self.rules)
        self.simulator = ctrl.ControlSystemSimulation(self.fuzzy_ctrl)

    def _add_mfs(self, variable):
        """Pomocnicza metoda do automatycznego generowania kształtów funkcji"""
        universe = variable.universe
        min_val = universe.min()
        max_val = universe.max()
        # Odległość między środkami kolejnych funkcji
        step = (max_val - min_val) / max(1, (self.num_mf - 1))

        for i in range(self.num_mf):
            name = f'level_{i}'  # Nazywamy je ogólnie: level_0, level_1, itd.
            center = min_val + i * step

            # Rysowanie odpowiednich kształtów za pomocą biblioteki skfuzzy
            if self.mf_type == 'trimf':
                # Trójkąt [lewy róg, środek, prawy róg]
                variable[name] = fuzz.trimf(universe, [center - step, center, center + step])
            elif self.mf_type == 'gaussmf':
                # Dzwon Gaussa [środek, szerokość/odchylenie]
                sigma = step / 2.5
                variable[name] = fuzz.gaussmf(universe, center, sigma)
            elif self.mf_type == 'trapmf':
                # Trapez [lewy dół, lewa góra, prawa góra, prawy dół]
                variable[name] = fuzz.trapmf(universe,
                                             [center - step, center - step / 4, center + step / 4, center + step])

    def _setup_membership_functions(self):
        """Podpina wygenerowane funkcje pod każdą zmienną"""
        self._add_mfs(self.age)
        self._add_mfs(self.amount)
        if self.num_attributes >= 3:
            self._add_mfs(self.duration)
        self._add_mfs(self.risk)

    def _build_rules(self):
        """Automatyczny generator bazy reguł dla dowolnej liczby funkcji i atrybutów"""
        self.rules = []

        # Pętle przechodzą przez wszystkie możliwe kombinacje funkcji (tzw. siatka)
        for a_idx in range(self.num_mf):
            for am_idx in range(self.num_mf):

                # Logika ekspercka:
                # Wiek: Młody (niski indeks) = większe ryzyko. Odwracamy indeks.
                age_risk = (self.num_mf - 1) - a_idx

                if self.num_attributes == 2:
                    # Uśredniamy ryzyko z 2 atrybutów
                    risk_idx = int(round((age_risk + am_idx) / 2))

                    antecedent = self.age[f'level_{a_idx}'] & self.amount[f'level_{am_idx}']
                    self.rules.append(ctrl.Rule(antecedent, self.risk[f'level_{risk_idx}']))

                elif self.num_attributes >= 3:
                    for d_idx in range(self.num_mf):
                        # Czas trwania: Dłuższy kredyt = większe ryzyko (d_idx)
                        risk_idx = int(round((age_risk + am_idx + d_idx) / 3))

                        antecedent = self.age[f'level_{a_idx}'] & self.amount[f'level_{am_idx}'] & self.duration[
                            f'level_{d_idx}']
                        self.rules.append(ctrl.Rule(antecedent, self.risk[f'level_{risk_idx}']))

    def predict(self, age_val, amount_val, duration_val=None):
        """Wylicza ryzyko. Jeśli system jest 3-atrybutowy, wymaga podania duration_val."""
        try:
            self.simulator.input['age'] = age_val
            self.simulator.input['credit_amount'] = amount_val
            if self.num_attributes >= 3 and duration_val is not None:
                self.simulator.input['duration'] = duration_val

            self.simulator.compute()
            return self.simulator.output['risk']
        except Exception as e:
            # Zwraca średnie ryzyko w razie jakiegokolwiek błędu biblioteki
            return 50.0