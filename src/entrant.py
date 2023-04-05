import math


class Entrant(object):
    def __init__(self, event, driver, team, start_position=0, num_racers=0):
        self._event = event
        self._driver = driver
        self._team = team
        self._result = None
        self._start_position = int(start_position)
        self._num_racers = int(num_racers)
        self._probability_fail_at_n = None
        self._probability_fail_after_n = None
        self._probability_succeed_through_n = None
        self._condition_multiplier_km = 1
        self._id = self._create_id()

    def set_result(self, result):
        self._result = result

    def set_start_position(self, start_position):
        self._start_position = start_position

    def id(self):
        return self._id

    def event(self):
        return self._event

    def driver(self):
        return self._driver

    def team(self):
        return self._team

    def start_position(self):
        return self._start_position

    def num_racers(self):
        return self._num_racers

    def has_result(self):
        return self._result is not None

    def result(self):
        return self._result

    def position_from_back(self):
        if self._result is None:
            return 0
        if self._num_racers and self._result.start_position():
            return self._num_racers - self._result.start_position()
        return 0

    def dnf_category(self):
        if self._result is not None:
            return self._result.dnf_category()
        return None

    def set_condition_multiplier_km(self, condition_multiplier_km):
        self._condition_multiplier_km = condition_multiplier_km

    def reset_condition_multiplier_km(self):
        self._condition_multiplier_km = 1

    def calculate_lap_reliability(self, num_laps, lap_distance_km):
        if self._probability_fail_at_n is not None:
            return
        self._probability_fail_at_n = [0] * (num_laps + 1)
        self._probability_fail_after_n = [0] * (num_laps + 1)
        self._probability_succeed_through_n = [1] * (num_laps + 1)
        self._probability_fail_at_n[0] = 1 - self.probability_survive_opening()
        self._probability_succeed_through_n[0] = self.probability_survive_opening()
        current_success_probability = self._probability_succeed_through_n[0]
        for n in range(1, num_laps + 1):
            this_lap_success = self._driver.rating().probability_finishing(start_km=(lap_distance_km * (n - 1)),
                                                                           race_distance_km=(lap_distance_km * n))
            this_lap_success *= self._team.rating().probability_finishing(start_km=(lap_distance_km * (n - 1)),
                                                                          race_distance_km=(lap_distance_km * n))
            this_lap_success *= (self._condition_multiplier_km ** lap_distance_km)
            this_lap_failure = 1 - this_lap_success
            # Odds of completing N-1 laps and then failing at lap N
            self._probability_fail_at_n[n] = current_success_probability * this_lap_failure
            # Odds of completing N laps
            current_success_probability *= this_lap_success
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
        # TODO: Remove this hack.
        if self._event.type() == 'Q':
            return 1.0
        grid_row = math.floor(self._start_position / 2) + 1
        if self._event.season() >= 2000:
            if grid_row <= 8:
                return 0.989 - (0.0035 * grid_row)
            else:
                return 0.965
        else:
            if grid_row <= 6:
                return 0.973 - (0.0052 * grid_row)
            else:
                return 0.919

    def probability_complete_n_laps(self, num_laps):
        if self._probability_fail_at_n is None:
            self.calculate_lap_reliability(self._event.num_laps(), self._event.lap_distance_km())
        return self._probability_succeed_through_n[num_laps]

    def probability_fail_at_n(self, num_laps):
        if self._probability_fail_at_n is None:
            self.calculate_lap_reliability(self._event.num_laps(), self._event.lap_distance_km())
        return self._probability_fail_at_n[num_laps]

    def probability_fail_after_n(self, num_laps):
        if self._probability_fail_at_n is None:
            self.calculate_lap_reliability(self._event.num_laps(), self._event.lap_distance_km())
        return self._probability_fail_after_n[num_laps]

    def _create_id(self):
        team_uuid = 'TeamXXXXX'
        if self._team is not None:
            team_uuid = self._team.uuid()
        return '%s:%s' % (self._driver.id(), team_uuid)
