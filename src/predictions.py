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

    def reset_predictions(self):
        # Reset the predictions but not the ratings
        self._rating_this = None
        self._rating_other = None
        self._this_elo_start_position_advantage = None
        self._this_win_probability = None
        self._this_elo_probability = None
        self._tie_probability = None

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
            # TODO: Parameterize this
            delta_driver_this = delta_all_this * 0.8
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
        self._simulation_log_file = self._prediction.simulation_log_file()
        # Raw and normalized results: [entrant][position] = count
        self._all_simulated_results = dict()
        self._position_probabilities = dict()
        self._is_normalized = False
        self._simulation_log = list()
        self._tmp_one_simulation_ordering = [None] * self._num_entrants
        self._tmp_one_simulation_num_laps = [None] * self._num_entrants
        self._init_simulated_results()

    def _init_simulated_results(self):
        # Force there to be the correct number of spots so even 0-count results are accounted for
        for entrant in self._event.entrants():
            self._all_simulated_results[entrant] = {int(n + 1): 0 for n in range(self._num_entrants)}

    def simulate(self, num_iterations=None, idx_offset=0):
        if num_iterations is None:
            num_iterations = self._num_iterations
        for idx in range(num_iterations):
            self._simulate_one(idx + idx_offset)

    def position_counts(self):
        return self._all_simulated_results

    def position_probabilities(self):
        self.normalize_simulated_positions()
        return self._position_probabilities

    def simulation_log(self):
        return self._simulation_log

    def _simulate_one(self, idx):
        self._is_normalized = False
        # Figure out how far each one gets
        distances = defaultdict(list)
        self._calculate_distances(distances)
        # Then compare all within each position
        self._determine_positions(distances)
        if self._simulation_log_file is not None:
            self.log_results(idx)

    def _calculate_distances(self, distances):
        for entrant in self._event.entrants():
            laps_completed = max([self._calculate_num_laps(entrant), 0])
            distances[laps_completed].append(entrant)

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
        for lap_num in sorted(distances.keys(), reverse=True):
            curr_end = curr_start + len(distances[lap_num])
            self._tmp_one_simulation_ordering[curr_start:curr_end] = self._order_entrant_results(distances[lap_num])
            self._tmp_one_simulation_num_laps[curr_start:curr_end] = [lap_num] * len(distances[lap_num])
            curr_start += len(distances[lap_num])
        # Append this outcome to the simulation log
        self._simulation_log.append('|'.join([e.driver().id() for e in self._tmp_one_simulation_ordering]))
        pos = 1
        for entrant in self._tmp_one_simulation_ordering:
            self._all_simulated_results[entrant][pos] += 1
            pos += 1

    def normalize_simulated_positions(self):
        if self._is_normalized:
            return
        for entrant, positions in self._all_simulated_results.items():
            entrant_probabilities = dict()
            for position in positions.keys():
                entrant_probabilities[position] = positions[position] / self._num_iterations
            self._position_probabilities[entrant] = entrant_probabilities
        self._is_normalized = True

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

    def log_results(self, idx):
        order_string = '|'.join([self.simulation_log_string(idx) for idx in range(self._num_entrants)])
        print('%s,%s,%s' % (self._event.id(), idx, order_string), file=self._simulation_log_file)

    def simulation_log_string(self, idx):
        entrant = self._tmp_one_simulation_ordering[idx]
        return 'P%02d:%s:%d:%d' % (idx + 1, entrant.id(), entrant.start_position(),
                                   self._tmp_one_simulation_num_laps[idx])


class EventPrediction(object):

    def __init__(self, event, args, elo_denominator, k_factor_adjust, car_factor, base_points_per_position,
                 base_car_reliability, base_new_car_reliability, base_driver_reliability,
                 debug_file, starting_positions=None, simulation_log=None):
        self._args = args
        self._event = event
        self._sorted_entrants = sorted(self._event.entrants(), key=lambda e: e.driver().id())
        self._num_iterations = self._args.num_iterations
        self._elo_denominator = elo_denominator
        self._k_factor_adjust = k_factor_adjust
        self._car_factor = car_factor
        self._base_points_per_position = base_points_per_position
        # TODO: Remove this/make it empirical
        if 'Monaco' in event.name():
            self._base_points_per_position *= 1.5
        elif event.is_street_course():
            self._base_points_per_position *= 1.1
        self._position_base_factor = self._args.position_base_factor
        self._base_car_reliability = base_car_reliability
        self._base_new_car_reliability = base_new_car_reliability
        self._base_driver_reliability = base_driver_reliability
        self._team_reliability_new_events = self._args.team_reliability_new_events
        self._win_probabilities = dict()
        self._podium_probabilities = dict()
        self._finish_probabilities = dict()
        self._naive_probabilities = dict()
        self._position_base_dict = None
        self._team_share_dict = None
        # Mapping of [entrant_a][entrant_b] to HeadToHeadPrediction
        self._head_to_head = defaultdict(dict)
        self._debug_file = debug_file
        self._driver_cache = None
        self._team_cache = None
        # These are only used in only_simulate_outcomes
        self._starting_positions = starting_positions
        self._simulation_log_file = simulation_log

    def cache_ratings(self):
        self._driver_cache = {driver.id(): copy.deepcopy(driver) for driver in self._event.drivers()}
        self._team_cache = {team.id(): copy.deepcopy(team) for team in self._event.teams()}

    def driver_before(self, driver_id):
        return self._driver_cache.get(driver_id)

    def team_before(self, team_id):
        return self._team_cache.get(team_id)

    def start_updates(self):
        driver_reliability_multiplier_km = 1
        if 'Monaco' in self._event.name():
            driver_reliability_multiplier_km = 0.9995
        if self._event.weather() == 'wet':
            driver_reliability_multiplier_km *= self._args.reliability_km_multiplier_wet
        if self._event.is_street_course():
            driver_reliability_multiplier_km *= self._args.reliability_km_multiplier_street
        for entrant in self._sorted_entrants:
            entrant.set_condition_multiplier_km(driver_reliability_multiplier_km)
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
        for entrant in self._sorted_entrants:
            entrant.reset_condition_multiplier_km()
        for driver in self._event.drivers():
            driver.commit_update()
        for team in self._event.teams():
            team.commit_update()

    def predict_winner(self, do_simulate_results):
        """
        Predict the probability of each entrant winning. Use the Odds Ratio.
        https://en.wikipedia.org/wiki/Odds_ratio
        """
        # Calculate the naive probabilities to finish this event.
        distance_km = self._event.total_distance_km()
        naive_car_probability = self._base_car_reliability.probability_finishing(race_distance_km=distance_km)
        naive_driver_probability = self._base_driver_reliability.probability_finishing(race_distance_km=distance_km)
        naive_all_probability = naive_car_probability * naive_driver_probability
        self._naive_probabilities = {
            _ALL: naive_all_probability, _CAR: naive_car_probability, _DRIVER: naive_driver_probability
        }
        # Predict all head-to-head and then simulate the event
        self.predict_all_head_to_head()
        simulator = EventSimulator(self)
        if do_simulate_results:
            simulator.simulate()
        for entrant, position_probabilities in simulator.position_probabilities().items():
            driver_id = entrant.driver().id()
            driver_finish = entrant.driver().rating().probability_finishing(race_distance_km=distance_km)
            car_finish = entrant.team().rating().probability_finishing(race_distance_km=distance_km)
            finish_probability = entrant.probability_complete_n_laps(self._event.num_laps())
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

    def apply_fuzz(self, fuzz, idx):
        self._fuzz_internal(fuzz, idx, 1)

    def remove_fuzz(self, fuzz, idx):
        self._fuzz_internal(fuzz, idx, -1)

    def _fuzz_internal(self, fuzz, idx, multiplier):
        for driver in self._event.drivers():
            driver_fuzz = fuzz.get(driver.id())
            if driver_fuzz is None:
                continue
            event_fuzz = driver_fuzz.get(self._event.id())
            if event_fuzz is None:
                continue
            total_fuzz = event_fuzz[idx]
            if not total_fuzz:
                continue
            driver.start_update(self._event.id(), None)
            driver.rating().update(total_fuzz * multiplier)
            driver.commit_update()
        for team in self._event.teams():
            team_fuzz = fuzz.get(team.id())
            if team_fuzz is None:
                continue
            event_fuzz = team_fuzz.get(self._event.id())
            if event_fuzz is None:
                continue
            total_fuzz = event_fuzz[idx]
            if not total_fuzz:
                continue
            team.start_update(self._event.id(), None)
            team.rating().update(total_fuzz * multiplier)
            team.commit_update()

    @staticmethod
    def apply_grid_penalties(starting_positions, grid_penalties):
        if grid_penalties is None or not grid_penalties:
            return
        # The raw temporary grid, with collisions allowed
        temporary_grid_0 = defaultdict(set)
        penalized = set()
        for penalty in grid_penalties:
            driver_id = penalty[0]
            penalized.add(driver_id)
            new_place = penalty[1] + starting_positions[driver_id]
            temporary_grid_0[new_place].add(driver_id)
        unpenalized_order = [driver_id for driver_id in sorted(starting_positions.keys(), key=lambda x: starting_positions[x])
                if driver_id not in penalized]
        # The updated temporary grid, with no collisions allowed
        # [position] = driver_id
        temporary_grid_1 = dict()
        for place in sorted(temporary_grid_0.keys(), reverse=True):
            curr_place = place
            for driver_id in sorted(temporary_grid_0[place], key=lambda d_id: starting_positions[d_id], reverse=True):
                while curr_place in temporary_grid_1:
                    curr_place -= 1
                temporary_grid_1[curr_place] = driver_id
        # Now start at the front of the grid and work our way back.
        # If the current spot is taken by a penalized driver, use that.
        # Otherwise pull the next unpenalized driver
        # [driver_id] = position
        updated_starting_positions = dict()
        for position in range(1, len(starting_positions) + 1):
            if position in temporary_grid_1:
                updated_starting_positions[temporary_grid_1[position]] = position
                del temporary_grid_1[position]
            elif unpenalized_order:
                updated_starting_positions[unpenalized_order[0]] = position
                del unpenalized_order[0]
            # There are no unpenalized drivers to move in front of the penalized drivers.
            # Just fill them in
        current_position = len(updated_starting_positions) + 1
        for _, driver_id in sorted(temporary_grid_1.items()):
            updated_starting_positions[driver_id] = current_position
            current_position += 1
        starting_positions.update(updated_starting_positions)

    def only_simulate_outcomes(self, fuzz, grid_penalties=None):
        if self._starting_positions is None:
            # This is a function call signature mismatch
            print('ERROR: Starting positions is None in simulation-only mode.', file=self._debug_file)
            return
        if not self._starting_positions:
            # This is a qualifying event and there are no starting positions
            self.set_starting_positions(None)
            simulator = EventSimulator(self)
            for idx in range(self._num_iterations):
                self.apply_fuzz(fuzz, idx)
                self.predict_all_head_to_head()
                simulator.simulate(idx_offset=idx, num_iterations=1)
                self.remove_fuzz(fuzz, idx)
            self._starting_positions.extend(simulator.simulation_log())
        else:
            # By default, only run one simulation per starting position ordering. However, if only one starting
            # position ordering is specified, this means it was generated artificially from a previous event.  E.g., the
            # last seen actual/real event was qualifying, and now we need to simulate the race given the outcome of
            # that qualifying session; in that case do the full set of simulations.
            if len(self._starting_positions) == 1:
                self._starting_positions = self._starting_positions * self._num_iterations
            elif len(self._starting_positions) != self._num_iterations:
                print('ERROR: In simulation-only mode were only given %d starting positions but expected %d.' % (
                    len(self._starting_positions), self._num_iterations
                ), file=self._debug_file)
                return
            simulation_outcomes = list()
            simulator = EventSimulator(self)
            for idx in range(self._num_iterations):
                self.set_starting_positions(self._starting_positions[idx], grid_penalties=grid_penalties)
                self.apply_fuzz(fuzz, idx)
                self.predict_all_head_to_head()
                simulator.simulate(num_iterations=1, idx_offset=idx)
                self.remove_fuzz(fuzz, idx)
                idx += 1
            simulation_outcomes.extend(simulator.simulation_log())
            self._starting_positions.clear()
            if self._event.type() != 'R':
                # We don't carry race results over but otherwise (qualifying and sprint) carry them over.
                self._starting_positions.extend(simulation_outcomes)

    def set_starting_positions(self, starting_order, grid_penalties=None):
        driver_to_start_position = dict()
        if starting_order is not None:
            p = 1
            for driver_id in starting_order.split('|'):
                driver_to_start_position[driver_id] = p
                p += 1
        self.apply_grid_penalties(driver_to_start_position, grid_penalties)
        for entrant in self._sorted_entrants:
            start_position = driver_to_start_position.get(entrant.driver().id(), 0)
            entrant.set_start_position(start_position)

    def predict_all_head_to_head(self):
        for idx_a in range(len(self._sorted_entrants) - 1):
            entrant_a = self._sorted_entrants[idx_a]
            for idx_b in range(idx_a, len(self._sorted_entrants)):
                entrant_b = self._sorted_entrants[idx_b]
                # Allow us to calculate the odds of an entrant against themselves. When calculating win probability we
                # need to calculate the odds of a person against themselves (it should be 50/50) so don't skip over
                # that one.
                if entrant_a in self._head_to_head and entrant_b in self._head_to_head[entrant_a]:
                    self._head_to_head[entrant_a][entrant_b].reset_predictions()
                else:
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
        if entrant_a == entrant_b:
            return 0.5
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
        if entrant_a == entrant_b:
            return 0.5
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
        if entrant_a == entrant_b:
            return 0, 0
        print('ERROR no (%s:%s) or (%s:%s) in GED' % (entrant_a.driver().id(), entrant_a.team().uuid(),
                                                      entrant_b.driver().id(), entrant_b.team().uuid()),
              file=self._debug_file)
        return None, None

    def event(self):
        return self._event

    def num_iterations(self):
        return self._num_iterations

    def simulation_log_file(self):
        return self._simulation_log_file

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
