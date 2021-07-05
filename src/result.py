class Result(object):
    def __init__(self, event, driver, end_position, team=None, start_position=0, num_racers=0, dnf_category='', laps=0):
        self._event = event
        self._driver = driver
        self._team = team
        self._start_position = int(start_position)
        self._end_position = int(end_position)
        self._num_racers = int(num_racers)
        self._laps = int(laps)
        self._dnf_category = dnf_category
        self._probability_fail_at_n = None
        self._probability_fail_after_n = None
        self._probability_succeed_through_n = None

    def event(self):
        return self._event

    def driver(self):
        return self._driver

    def team(self):
        return self._team

    def start_position(self):
        return self._start_position

    def end_position(self):
        return self._end_position

    def num_racers(self):
        return self._num_racers

    def position_from_back(self):
        if self._num_racers and self._start_position:
            return self._num_racers - self._start_position
        return 0

    def dnf_category(self):
        return self._dnf_category

    def laps(self):
        return self._laps

    def calculate_lap_reliability(self, num_laps, lap_distance_km):
        if self._probability_fail_at_n is not None:
            return
        self._probability_fail_at_n = [0] * (num_laps + 1)
        self._probability_fail_after_n = [0] * (num_laps + 1)
        self._probability_succeed_through_n = [1] * (num_laps + 1)
        if self._event.type() == 'Q':
            return
        self._probability_fail_at_n[0] = 1 - self.probability_survive_opening()
        self._probability_succeed_through_n[0] = self.probability_survive_opening()
        per_lap_success_probability = self._driver.rating().probability_finishing(race_distance_km=lap_distance_km)
        per_lap_success_probability *= self._team.rating().probability_finishing(race_distance_km=lap_distance_km)
        per_lap_failure_probability = 1 - per_lap_success_probability
        current_success_probability = self._probability_succeed_through_n[0]
        for n in range(1, num_laps + 1):
            # Odds of completing N-1 laps and then failing at lap N
            self._probability_fail_at_n[n] = current_success_probability * per_lap_failure_probability
            # Odds of completing N laps
            current_success_probability *= per_lap_success_probability
            self._probability_succeed_through_n[n] = current_success_probability
        # The probability they fail after N laps is:
        #   The probability they fail by the end
        #     MINUS
        #   The probability they did NOT fail after N laps (i.e., 1 - probability of completing N laps)
        # (1-succeed[ALL]) - (1-succeed[N])
        # 1 - succeed[ALL] - 1 + succeed[N]
        # succeed[N] - succeed[ALL]
        for n in range(1, num_laps + 1):
            self._probability_fail_after_n[n] = self._probability_succeed_through_n[n] - current_success_probability

    def probability_survive_opening(self):
        if self._event.type() == 'Q':
            return 1.0
        if self._start_position <= 10:
            return 0.982 - (0.0024 * self._start_position)
        else:
            return 0.95

    def probability_complete_n_laps(self, num_laps):
        self.calculate_lap_reliability(self._event.num_laps(), self._event.lap_distance_km())
        return self._probability_succeed_through_n[num_laps]

    def probability_fail_at_n(self, num_laps):
        self.calculate_lap_reliability(self._event.num_laps(), self._event.lap_distance_km())
        return self._probability_fail_at_n[num_laps]

    def probability_fail_after_n(self, num_laps):
        self.calculate_lap_reliability(self._event.num_laps(), self._event.lap_distance_km())
        return self._probability_fail_after_n[num_laps]
