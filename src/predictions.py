from collections import defaultdict

import copy
import math
import random
import ratings


_ALL = 'All'
_CAR = 'Car'
_DRIVER = 'Driver'


def elo_rating_from_entrant(car_factor, entrant):
    rating = (1 - car_factor) * entrant.driver().rating().elo()
    rating += (car_factor * entrant.team().rating().elo())
    return rating


def elo_win_probability(r_a, r_b, denominator):
    """Standard logistic calculator, per
       https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    q_a = 10 ** (r_a / denominator)
    q_b = 10 ** (r_b / denominator)
    return q_a / (q_a + q_b)


class HeadToHeadPrediction(object):

    def __init__(self, event, entrant_this, entrant_other, elo_denominator, k_factor_adjust, car_factor,
                 base_points_per_position, position_base_factor, debug_file):
        self._event = event
        self._entrant_this = entrant_this
        self._entrant_other = entrant_other
        self._elo_denominator = elo_denominator
        self._k_factor_adjust = k_factor_adjust
        self._car_factor = car_factor
        self._base_points_per_position = base_points_per_position
        self._position_base_factor = position_base_factor
        self._rating_this = None
        self._rating_other = None
        self._k_factor = None
        self._this_elo_start_position_advantage = None
        self._this_win_probability = None
        self._this_elo_probability = None
        self._this_further_than_other = None
        self._tie_probability = None
        self._double_dnf_probability = None
        self._debug_file = debug_file

    def this_won(self):
        return 1 if self._entrant_this.result().end_position() < self._entrant_other.result().end_position() else 0

    def same_team(self):
        return self._entrant_this.team().uuid() == self._entrant_other.team().uuid()

    def this_win_probability(self, get_other=False):
        if self._k_factor == ratings.KFactor.INVALID:
            return None
        if self._this_win_probability is not None:
            if get_other:
                return 1 - self._this_win_probability
            else:
                return self._this_win_probability
        if abs(1. - self.probability_double_dnf()) < 1e-3:
            return 0
        num_laps = self._event.num_laps()
        this_elo_probability_raw, _, _ = self.this_elo_probability()
        if this_elo_probability_raw is None:
            return None
        probability_this_complete = self._entrant_this.probability_complete_n_laps(num_laps)
        probability_other_complete = self._entrant_other.probability_complete_n_laps(num_laps)
        probability_both_complete = probability_this_complete * probability_other_complete
        probability_this_not_other = probability_this_complete * (1 - probability_other_complete)
        probability_this_further_than_other = self.probability_this_further_than_other()
        this_elo_probability = this_elo_probability_raw * probability_both_complete
        full_this_probability = this_elo_probability + probability_this_not_other + probability_this_further_than_other
        full_this_probability += (0.5 * self._tie_probability)
        if self._debug_file is not None:
            print('%s ThisElo: %.7f x %.7f x %.7f = %.7f and PTNO: %.7f PTFTO: %.7f PThisWin: %.7f %s:%s vs %s:%s' % (
                self._event.id(), this_elo_probability_raw, probability_this_complete, probability_other_complete,
                this_elo_probability, probability_this_not_other, probability_this_further_than_other,
                full_this_probability, self._entrant_this.driver().id(), self._entrant_this.team().uuid(),
                self._entrant_other.driver().id(), self._entrant_other.team().uuid()),
                  file=self._debug_file)
        self._this_win_probability = full_this_probability
        if self._this_win_probability == 1.0:
            self._this_win_probability = 1.0 - 1e-6
        elif self._this_win_probability == 0.0:
            self._this_win_probability = 1e-6
        if get_other:
            return 1 - self._this_win_probability
        else:
            return self._this_win_probability

    def this_elo_probability(self, get_other=False):
        if self.combined_k_factor() == ratings.KFactor.INVALID:
            return None, None, None
        if self._this_elo_probability is not None:
            if get_other:
                return 1 - self._this_elo_probability, self._rating_other, self._rating_this
            else:
                return self._this_elo_probability, self._rating_this, self._rating_other
        self._rating_this = elo_rating_from_entrant(self._car_factor, self._entrant_this)
        self._rating_this += self.this_elo_start_position_advantage()
        self._rating_other = elo_rating_from_entrant(self._car_factor, self._entrant_other)
        self._this_elo_probability = elo_win_probability(self._rating_this, self._rating_other, self._elo_denominator)
        if self._debug_file is not None:
            print('%s TEP: (%.1f + %.1f) vs (%.1f) = %.6f' % (
                self._event.id(), self._rating_this, self.this_elo_start_position_advantage(),
                self._rating_other, self._this_elo_probability), file=self._debug_file)
        if get_other:
            return 1 - self._this_elo_probability, self._rating_other, self._rating_this
        else:
            return self._this_elo_probability, self._rating_this, self._rating_other

    def this_elo_deltas(self, get_other=False):
        elo_win_probability_this, _, _ = self.this_elo_probability()
        if elo_win_probability_this is None:
            return None, None
        win_actual_this = self.this_won()
        k_factor = self.combined_k_factor()
        delta_all_this = k_factor * (win_actual_this - elo_win_probability_this)
        if get_other:
            delta_all_this *= -1
        if self.same_team():
            delta_car_this = 0
            delta_driver_this = delta_all_this
        else:
            delta_car_this = self._car_factor * delta_all_this
            delta_driver_this = delta_all_this - delta_car_this
        return delta_car_this, delta_driver_this

    def elo_rating(self, entrant):
        if self.combined_k_factor() == ratings.KFactor.INVALID:
            return None
        if entrant == self._entrant_this:
            return self._rating_this
        elif entrant == self._entrant_other:
            return self._rating_other
        else:
            return None

    def probability_double_dnf(self):
        if self._double_dnf_probability is not None:
            return self._double_dnf_probability
        num_laps = self._event.num_laps()
        probability_this_dnf = 1 - self._entrant_this.probability_complete_n_laps(num_laps)
        probability_other_dnf = 1 - self._entrant_other.probability_complete_n_laps(num_laps)
        self._double_dnf_probability = probability_this_dnf * probability_other_dnf
        return self._double_dnf_probability

    def probability_this_further_than_other(self):
        if self._this_further_than_other is not None:
            return self._this_further_than_other
        total_this_further = 0
        total_tie_probability = 0
        # This is the probability that 'other' fails on lap N AND 'this' fails after N laps but before the end.
        for current_lap in range(0, self._event.num_laps()):
            probability_this_fail_at_n = self._entrant_this.probability_fail_at_n(current_lap)
            probability_this_fail_after_n = self._entrant_this.probability_fail_after_n(current_lap)
            probability_other_fail_at_n = self._entrant_other.probability_fail_at_n(current_lap)
            total_this_further += (probability_this_fail_after_n * probability_other_fail_at_n)
            total_tie_probability += (probability_this_fail_at_n * probability_other_fail_at_n)
        self._this_further_than_other = total_this_further
        self._tie_probability = total_tie_probability
        return self._this_further_than_other

    def combined_k_factor(self):
        if self._k_factor is not None:
            return self._k_factor
        k_factor_driver_this = self._entrant_this.driver().rating().k_factor().factor()
        k_factor_team_this = self._entrant_this.team().rating().k_factor().factor()
        k_factor_driver_other = self._entrant_other.driver().rating().k_factor().factor()
        k_factor_team_other = self._entrant_other.team().rating().k_factor().factor()
        if not k_factor_driver_this and not k_factor_team_this:
            self._k_factor = ratings.KFactor.INVALID
            return self._k_factor
        if not k_factor_driver_other and not k_factor_team_other:
            self._k_factor = ratings.KFactor.INVALID
            return self._k_factor
        if self.same_team():
            # Only average the K-Factor of the two drivers
            self._k_factor = (k_factor_driver_this + k_factor_driver_other) / 2
        else:
            k_factor_this = (self._car_factor * k_factor_team_this) + ((1 - self._car_factor) * k_factor_driver_this)
            k_factor_other = (self._car_factor * k_factor_team_other) + ((1 - self._car_factor) * k_factor_driver_other)
            self._k_factor = (k_factor_this + k_factor_other) / 2
        self._k_factor *= self._k_factor_adjust
        return self._k_factor

    def this_elo_start_position_advantage(self):
        """
        Summing a geometric series where a=the base points per position and n=number of positions ahead.
        https://en.wikipedia.org/wiki/Geometric_series#Sum
        We exclude the nth term from the summation. E.g., a 10 Elo point base benefit starting 2 spots ahead with a
        base factor of 0.85 is:
        advantage = 10 * ((1 - 0.85^2) / (1 - 0.85))
                  = 10 * (0.2775 / 0.15)
                  = 10 * 1.85
                  = 18.5
        """
        if self._this_elo_start_position_advantage is not None:
            return self._this_elo_start_position_advantage
        start_position_difference = self._entrant_other.start_position() - self._entrant_this.start_position()
        if not start_position_difference:
            self._this_elo_start_position_advantage = 0
            return self._this_elo_start_position_advantage
        if abs(self._position_base_factor - 1.0) > 1e-2:
            factor = 1 - math.pow(self._position_base_factor, abs(start_position_difference))
            factor /= 1 - self._position_base_factor
        else:
            factor = self._position_base_factor * abs(start_position_difference)
        if self._entrant_other.start_position() < self._entrant_this.start_position():
            factor *= -1.0
        self._this_elo_start_position_advantage = self._base_points_per_position * factor
        return self._this_elo_start_position_advantage


class EventSimulator(object):

    def __init__(self, event_prediction):
        self._prediction = event_prediction
        self._num_iterations = event_prediction.num_iterations()
        self._event = event_prediction.event()
        self._num_entrants = len(self._event.entrants())
        # Raw results: [entrant][position] = count
        self._all_simulated_results = dict()
        self._init_simulated_results()
        self._tmp_one_simulation_ordering = [None] * self._num_entrants

    def _init_simulated_results(self):
        for entrant in self._event.entrants():
            self._all_simulated_results[entrant] = {int(n + 1): 0 for n in range(self._num_entrants)}

    def simulate(self, num_iterations=None):
        if num_iterations is None:
            num_iterations = self._num_iterations
        for _ in range(num_iterations):
            self._simulate_one()
        self._normalize_simulated_positions()

    def position_probabilities(self):
        return self._all_simulated_results

    def _simulate_one(self):
        # Figure out how far each one gets
        distances = defaultdict(list)
        self._calculate_distances(distances)
        # Then compare all within each position
        self._determine_positions(distances)

    def _calculate_distances(self, distances):
        for entrant in self._event.entrants():
            distances[self._calculate_num_laps(entrant)].append(entrant)

    def _calculate_num_laps(self, entrant):
        failure_probability = random.random()
        # Did we make it to the end?
        if failure_probability < entrant.probability_complete_n_laps(self._event.num_laps()):
            return self._event.num_laps()
        # Binary search the rest
        high = self._event.num_laps()
        low = 0
        while high - 1 > low:
            mid = int((high + low) / 2)
            mid_prob = entrant.probability_complete_n_laps(mid)
            if failure_probability < mid_prob:
                # We went further than the midpoint
                low = mid
            else:
                # We didn't make it to the midpoint
                high = mid
        # Either high == low or high = low + 1
        if high == low:
            if failure_probability < entrant.probability_complete_n_laps(high):
                return high
            else:
                return high - 1
        else:
            if failure_probability < entrant.probability_complete_n_laps(high):
                return high
            elif failure_probability < entrant.probability_complete_n_laps(low):
                return low
            else:
                return low - 1

    def _determine_positions(self, distances):
        curr_start = 0
        for lap_num in sorted(distances, reverse=True):
            curr_end = curr_start + len(distances[lap_num])
            self._tmp_one_simulation_ordering[curr_start:curr_end] = self._order_entrant_results(distances[lap_num])
            curr_start += len(distances[lap_num])
        pos = 1
        for entrant in self._tmp_one_simulation_ordering:
            self._all_simulated_results[entrant][pos] += 1
            pos += 1

    def _normalize_simulated_positions(self):
        for positions in self._all_simulated_results.values():
            for position in positions.keys():
                positions[position] /= self._num_iterations

    def _order_entrant_results(self, same_lap_entrant_results):
        num_entrants = len(same_lap_entrant_results)
        if num_entrants <= 1:
            return same_lap_entrant_results
        return_array = [None] * num_entrants
        pivot_idx = random.randrange(num_entrants - 1)
        pivot_entrant = same_lap_entrant_results[pivot_idx]
        ahead_idx = 0
        behind_idx = num_entrants - 1
        for e in same_lap_entrant_results:
            if e == pivot_entrant:
                continue
            c = self._compare(e, pivot_entrant)
            if c < 0:
                # 'r' finished ahead of the pivot_entrant
                return_array[ahead_idx] = e
                ahead_idx += 1
            else:
                return_array[behind_idx] = e
                behind_idx -= 1
        return_array[ahead_idx] = pivot_entrant
        if ahead_idx > 1:
            return_array[0:ahead_idx] = self._order_entrant_results(return_array[0:ahead_idx])
        if num_entrants - behind_idx > 1:
            return_array[behind_idx + 1:num_entrants] = \
                    self._order_entrant_results(return_array[behind_idx + 1:num_entrants])
        return return_array

    def _compare(self, entrant_a, entrant_b):
        """Return a value < 0 if 'a' finished ahead of 'b'
        """
        elo_win_prob, _, _ = self._prediction.get_elo_win_probability(entrant_a, entrant_b)
        if elo_win_prob is None:
            return 0
        return random.random() - elo_win_prob


class EventPrediction(object):

    def __init__(self, event, num_iterations, elo_denominator, k_factor_adjust, car_factor, base_points_per_position,
                 position_base_factor, base_car_reliability, base_new_car_reliability, base_driver_reliability,
                 team_reliability_new_events, debug_file):
        self._event = event
        self._num_iterations = num_iterations
        self._elo_denominator = elo_denominator
        self._k_factor_adjust = k_factor_adjust
        self._car_factor = car_factor
        self._base_points_per_position = base_points_per_position
        self._position_base_factor = position_base_factor
        self._base_car_reliability = base_car_reliability
        self._base_new_car_reliability = base_new_car_reliability
        self._base_driver_reliability = base_driver_reliability
        self._team_reliability_new_events = team_reliability_new_events
        self._win_probabilities = dict()
        self._podium_probabilities = dict()
        self._finish_probabilities = dict()
        self._naive_probabilities = dict()
        self._position_base_dict = None
        self._team_share_dict = None
        self._head_to_head = defaultdict(dict)
        self._debug_file = debug_file
        self._driver_cache = None
        self._team_cache = None

    def cache_ratings(self):
        self._driver_cache = {driver.id(): copy.deepcopy(driver) for driver in self._event.drivers()}
        self._team_cache = {team.id(): copy.deepcopy(team) for team in self._event.teams()}

    def driver_before(self, driver_id):
        return self._driver_cache.get(driver_id)

    def team_before(self, team_id):
        return self._team_cache.get(team_id)

    def start_updates(self):
        for driver in sorted(self._event.drivers(), key=lambda d: d.id()):
            driver.start_update(self._event.id(), self._base_driver_reliability)
        for team in sorted(self._event.teams(), key=lambda t: t.id()):
            team.start_update(self._event.id(), self._base_new_car_reliability)
            if team.rating().k_factor().num_events() >= self._team_reliability_new_events:
                team.rating().reliability().set_template(self._base_car_reliability)
        # It's safe to call this here since we can guarantee it will only be called once.
        # IOW we won't accidentally decay the rate unnecessarily.
        self._base_car_reliability.start_update()
        self._base_new_car_reliability.start_update()
        self._base_driver_reliability.start_update()

    def maybe_force_regress(self):
        for driver in sorted(self._event.drivers(), key=lambda d: d.id()):
            driver.maybe_regress()
        for team in sorted(self._event.teams(), key=lambda t: t.id()):
            team.maybe_regress()

    def commit_updates(self):
        for driver in self._event.drivers():
            driver.commit_update()
        for team in self._event.teams():
            team.commit_update()

    def predict_winner(self, do_simulate_results):
        """
        Predict the probability of each entrant winning. Use the odds ratio.
        https://en.wikipedia.org/wiki/Odds_ratio
        """
        self.predict_all_head_to_head()
        # Calculate the naive probabilities to finish this event.
        distance_km = self._event.total_distance_km()
        naive_car_probability = self._base_car_reliability.probability_finishing(race_distance_km=distance_km)
        naive_driver_probability = self._base_driver_reliability.probability_finishing(race_distance_km=distance_km)
        naive_all_probability = naive_car_probability * naive_driver_probability
        self._naive_probabilities = {
            _ALL: naive_all_probability, _CAR: naive_car_probability, _DRIVER: naive_driver_probability
        }
        # Simulate the event
        simulator = EventSimulator(self)
        if do_simulate_results:
            simulator.simulate()
        for entrant, position_probabilities in simulator.position_probabilities().items():
            driver_id = entrant.driver().id()
            driver_finish = entrant.driver().rating().probability_finishing(race_distance_km=distance_km)
            car_finish = entrant.team().rating().probability_finishing(race_distance_km=distance_km)
            finish_probability = driver_finish * car_finish
            self.finish_probabilities()[driver_id] = {
                'All': finish_probability, 'Car': car_finish, 'Driver': driver_finish
            }
            if do_simulate_results:
                self.win_probabilities()[driver_id] = position_probabilities.get(1, 0)
                self.podium_probabilities()[driver_id] = position_probabilities.get(1, 0)
                self.podium_probabilities()[driver_id] += position_probabilities.get(2, 0)
                self.podium_probabilities()[driver_id] += position_probabilities.get(3, 0)

    def predict_podium(self):
        """
        https://math.stackexchange.com/questions/625611/given-every-horses-chance-of-winning-a-race-what-is-the-probability-that-a-spe
        """
        win_probs = self.win_probabilities()
        second = defaultdict(float)
        third = defaultdict(float)
        for driver_id_a, win_prob_a in win_probs.items():
            for driver_id_b, win_prob_b in win_probs.items():
                if driver_id_a == driver_id_b:
                    continue
                prob_a_then_b = win_prob_a * (win_prob_b / (1 - win_prob_a))
                for driver_id_c, win_prob_c in win_probs.items():
                    if driver_id_c == driver_id_a or driver_id_c == driver_id_b:
                        continue
                    prob_a_then_b_then_c = prob_a_then_b * (win_prob_c / (1 - (win_prob_a + win_prob_b)))
                    third[driver_id_c] += prob_a_then_b_then_c
                second[driver_id_b] += prob_a_then_b
        for driver_id, win_prob in win_probs.items():
            self.podium_probabilities()[driver_id] = \
                win_prob + second.get(driver_id, 0) + third.get(driver_id, 0)
            if self._debug_file is not None:
                print('AddPodium\t%s\t%7.5f\t%7.5f\t%7.5f\t%7.5f\t%s' % (
                    self._event.id(), self.podium_probabilities().get(driver_id), win_prob,
                    second.get(driver_id, 0), third.get(driver_id, 0), driver_id),
                      file=self._debug_file)

    def predict_all_head_to_head(self):
        for entrant_a in self._event.entrants():
            for entrant_b in self._event.entrants():
                # Allow us to calculate the odds of an entrant against themselves. When calculating win probability we
                # need to calculate the odds of a person against themselves (it should be 50/50) so don't skip over
                # that one.
                if entrant_a.driver().id() > entrant_b.driver().id():
                    continue
                head_to_head = HeadToHeadPrediction(self._event, entrant_a, entrant_b, self._elo_denominator,
                                                    self._k_factor_adjust, self._car_factor,
                                                    self._base_points_per_position, self._position_base_factor,
                                                    self._debug_file)
                self._head_to_head[entrant_a][entrant_b] = head_to_head

    def get_win_probability(self, entrant_a, entrant_b):
        if entrant_a in self._head_to_head:
            if entrant_b in self._head_to_head[entrant_a]:
                return self._head_to_head[entrant_a][entrant_b].this_win_probability()
        if entrant_b in self._head_to_head:
            if entrant_a in self._head_to_head[entrant_b]:
                return self._head_to_head[entrant_b][entrant_a].this_win_probability(get_other=True)
        print('ERROR no (%s:%s) or (%s:%s) in GWP' % (entrant_a.driver().id(), entrant_a.team().uuid(),
                                                      entrant_b.driver().id(), entrant_b.team().uuid()),
              file=self._debug_file)
        return None

    def get_elo_win_probability(self, entrant_a, entrant_b):
        if entrant_a in self._head_to_head:
            if entrant_b in self._head_to_head[entrant_a]:
                return self._head_to_head[entrant_a][entrant_b].this_elo_probability()
        if entrant_b in self._head_to_head:
            if entrant_a in self._head_to_head[entrant_b]:
                return self._head_to_head[entrant_b][entrant_a].this_elo_probability(get_other=True)
        print('ERROR no (%s:%s) or (%s:%s) in GEWP' % (entrant_a.driver().id(), entrant_a.team().uuid(),
                                                       entrant_b.driver().id(), entrant_b.team().uuid()),
              file=self._debug_file)
        return None

    def get_elo_deltas(self, entrant_a, entrant_b):
        if entrant_a in self._head_to_head:
            if entrant_b in self._head_to_head[entrant_a]:
                return self._head_to_head[entrant_a][entrant_b].this_elo_deltas()
        if entrant_b in self._head_to_head:
            if entrant_b in self._head_to_head[entrant_b]:
                return self._head_to_head[entrant_b][entrant_a].this_elo_deltas(get_other=True)
        print('ERROR no (%s:%s) or (%s:%s) in GED' % (entrant_a.driver().id(), entrant_a.team().uuid(),
                                                      entrant_b.driver().id(), entrant_b.team().uuid()),
              file=self._debug_file)
        return None, None

    def event(self):
        return self._event

    def num_iterations(self):
        return self._num_iterations

    def elo_denominator(self):
        return self._elo_denominator

    def win_probabilities(self):
        return self._win_probabilities

    def podium_probabilities(self):
        return self._podium_probabilities

    def finish_probabilities(self):
        return self._finish_probabilities

    def naive_finish_probabilities(self):
        return self._naive_probabilities
