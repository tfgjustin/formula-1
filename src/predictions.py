from collections import defaultdict

import copy
import math
import random
import ratings


_ALL = 'All'
_CAR = 'Car'
_DRIVER = 'Driver'


def elo_rating_from_result(car_factor, result):
    rating = (1 - car_factor) * result.driver().rating().elo()
    rating += (car_factor * result.team().rating().elo())
    return rating


def elo_win_probability(r_a, r_b, denominator):
    """Standard logistic calculator, per
       https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    q_a = 10 ** (r_a / denominator)
    q_b = 10 ** (r_b / denominator)
    return q_a / (q_a + q_b)


def select_reference_results(event, car_factor, num_results=4):
    if event.type() == 'Q':
        performance = sorted([result for result in event.results()],
                             key=lambda r: elo_rating_from_result(car_factor, r), reverse=True)
        return performance[:num_results]
    else:
        # Use this for sprint qualifying and races.
        reliability = sorted([result for result in event.results()], key=lambda r: r.probability_complete_n_laps(1),
                             reverse=True)
        return reliability[:num_results]


def was_performance_win(result_this, result_other):
    return result_this.dnf_category() == '-' and result_other.dnf_category() == '-'


class HeadToHeadPrediction(object):

    def __init__(self, event, result_this, result_other, elo_denominator, k_factor_adjust, car_factor,
                 base_points_per_position, position_base_factor, debug_file):
        self._event = event
        self._result_this = result_this
        self._result_other = result_other
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
        return 1 if self._result_this.end_position() < self._result_other.end_position() else 0

    def same_team(self):
        return self._result_this.team().uuid() == self._result_other.team().uuid()

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
        probability_this_complete = self._result_this.probability_complete_n_laps(num_laps)
        probability_other_complete = self._result_other.probability_complete_n_laps(num_laps)
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
                full_this_probability, self._result_this.driver().id(), self._result_this.team().uuid(),
                self._result_other.driver().id(), self._result_other.team().uuid()),
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
        self._rating_this = elo_rating_from_result(self._car_factor, self._result_this)
        self._rating_this += self.this_elo_start_position_advantage()
        self._rating_other = elo_rating_from_result(self._car_factor, self._result_other)
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
        if not was_performance_win(self._result_this, self._result_other):
            # One of these DNF'ed
            return None, None
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

    def elo_rating(self, result):
        if self.combined_k_factor() == ratings.KFactor.INVALID:
            return None
        if result == self._result_this:
            return self._rating_this
        elif result == self._result_other:
            return self._rating_other
        else:
            return None

    def probability_double_dnf(self):
        if self._double_dnf_probability is not None:
            return self._double_dnf_probability
        num_laps = self._event.num_laps()
        probability_this_dnf = 1 - self._result_this.probability_complete_n_laps(num_laps)
        probability_other_dnf = 1 - self._result_other.probability_complete_n_laps(num_laps)
        self._double_dnf_probability = probability_this_dnf * probability_other_dnf
        return self._double_dnf_probability

    def probability_this_further_than_other(self):
        if self._this_further_than_other is not None:
            return self._this_further_than_other
        total_this_further = 0
        total_tie_probability = 0
        # This is the probability that 'other' fails on lap N AND 'this' fails after N laps but before the end.
        for current_lap in range(0, self._event.num_laps()):
            probability_this_fail_at_n = self._result_this.probability_fail_at_n(current_lap)
            probability_this_fail_after_n = self._result_this.probability_fail_after_n(current_lap)
            probability_other_fail_at_n = self._result_other.probability_fail_at_n(current_lap)
            total_this_further += (probability_this_fail_after_n * probability_other_fail_at_n)
            total_tie_probability += (probability_this_fail_at_n * probability_other_fail_at_n)
        self._this_further_than_other = total_this_further
        self._tie_probability = total_tie_probability
        return self._this_further_than_other

    def combined_k_factor(self):
        if self._k_factor is not None:
            return self._k_factor
        k_factor_driver_this = self._result_this.driver().rating().k_factor().factor()
        k_factor_team_this = self._result_this.team().rating().k_factor().factor()
        k_factor_driver_other = self._result_other.driver().rating().k_factor().factor()
        k_factor_team_other = self._result_other.team().rating().k_factor().factor()
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
        start_position_difference = self._result_other.start_position() - self._result_this.start_position()
        if not start_position_difference:
            self._this_elo_start_position_advantage = 0
            return self._this_elo_start_position_advantage
        if abs(self._position_base_factor - 1.0) > 1e-2:
            factor = 1 - math.pow(self._position_base_factor, abs(start_position_difference))
            factor /= 1 - self._position_base_factor
        else:
            factor = self._position_base_factor * abs(start_position_difference)
        if self._result_other.start_position() < self._result_this.start_position():
            factor *= -1.0
        self._this_elo_start_position_advantage = self._base_points_per_position * factor
        return self._this_elo_start_position_advantage


class EventSimulator(object):

    def __init__(self, event_prediction):
        self._prediction = event_prediction
        self._num_iterations = event_prediction.num_iterations()
        self._event = event_prediction.event()
        # Raw results: [result][position] = count
        self._results = dict()

    def simulate(self):
        for _ in range(self._num_iterations):
            self._simulate_one()
        self._normalize_results()

    def position_probabilities(self):
        return self._results

    def _simulate_one(self):
        # Figure out how far each one gets
        distances = defaultdict(list)
        self._calculate_distances(distances)
        # Then compare all within each position
        self._determine_positions(distances)

    def _calculate_distances(self, distances):
        for result in self._event.results():
            distances[self._calculate_num_laps(result)].append(result)

    def _calculate_num_laps(self, result):
        distance = random.random()
        for laps_from_end in range(self._event.num_laps()):
            complete_n = self._event.num_laps() - laps_from_end
            if result.probability_complete_n_laps(complete_n) > distance:
                return complete_n
        return 0

    def _determine_positions(self, distances):
        ordered_results = list()
        for lap_num in sorted(distances, reverse=True):
            ordered_results.extend(self._order_results(distances[lap_num]))
        pos = 1
        for result in ordered_results:
            positions = self._results.get(result, dict())
            if pos in positions:
                positions[pos] += 1
            else:
                positions[pos] = 1
            self._results[result] = positions
            pos += 1

    def _normalize_results(self):
        for result, positions in self._results.items():
            for position in positions.keys():
                positions[position] /= self._num_iterations

    def _order_results(self, same_lap_results):
        if len(same_lap_results) <= 1:
            return same_lap_results
        random.shuffle(same_lap_results)
        pivot_result = same_lap_results[0]
        ahead = list()
        behind = list()
        for r in same_lap_results[1:]:
            cmp = self._compare(pivot_result, r)
            if cmp < 0:
                ahead.append(r)
            else:
                behind.append(r)
        return self._order_results(ahead) + [pivot_result] + self._order_results(behind)

    def _compare(self, a, b):
        elo_win_prob, _, _ = self._prediction.get_elo_win_probability(a, b)
        if elo_win_prob is None:
            return 0
        return math.copysign(1, elo_win_prob - random.random())


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

    def commit_updates(self):
        for driver in self._event.drivers():
            driver.commit_update()
        for team in self._event.teams():
            team.commit_update()

    def predict_winner(self):
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
        simulator.simulate()
        for result, position_probabilities in simulator.position_probabilities().items():
            driver_id = result.driver().id()
            driver_finish = result.driver().rating().probability_finishing(race_distance_km=distance_km)
            car_finish = result.team().rating().probability_finishing(race_distance_km=distance_km)
            finish_probability = driver_finish * car_finish
            self.finish_probabilities()[driver_id] = {
                'All': finish_probability, 'Car': car_finish, 'Driver': driver_finish
            }
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
        for result_a in self._event.results():
            for result_b in self._event.results():
                # Allow us to calculate the odds of an entrant against themselves. When calculating win probability we
                # need to calculate the odds of a person against themselves (it should be 50/50) so don't skip over
                # that one.
                if result_a.driver().id() > result_b.driver().id():
                    continue
                head_to_head = HeadToHeadPrediction(self._event, result_a, result_b, self._elo_denominator,
                                                    self._k_factor_adjust, self._car_factor,
                                                    self._base_points_per_position, self._position_base_factor,
                                                    self._debug_file)
                self._head_to_head[result_a][result_b] = head_to_head
                # print('Added (%s:%s) and (%s:%s)' % (result_a.driver().id(), result_a.team().uuid(),
                #                                      result_b.driver().id(), result_b.team().uuid()))

    def get_win_probability(self, result_a, result_b):
        if result_a in self._head_to_head:
            if result_b in self._head_to_head[result_a]:
                return self._head_to_head[result_a][result_b].this_win_probability()
        if result_b in self._head_to_head:
            if result_a in self._head_to_head[result_b]:
                return self._head_to_head[result_b][result_a].this_win_probability(get_other=True)
        print('ERROR no (%s:%s) or (%s:%s) in GWP' % (result_a.driver().id(), result_a.team().uuid(),
                                                      result_b.driver().id(), result_b.team().uuid()),
              file=self._debug_file)
        return None

    def get_elo_win_probability(self, result_a, result_b):
        if result_a in self._head_to_head:
            if result_b in self._head_to_head[result_a]:
                return self._head_to_head[result_a][result_b].this_elo_probability()
        if result_b in self._head_to_head:
            if result_a in self._head_to_head[result_b]:
                return self._head_to_head[result_b][result_a].this_elo_probability(get_other=True)
        print('ERROR no (%s:%s) or (%s:%s) in GEWP' % (result_a.driver().id(), result_a.team().uuid(),
                                                       result_b.driver().id(), result_b.team().uuid()),
              file=self._debug_file)
        return None

    def get_elo_deltas(self, result_a, result_b):
        if result_a in self._head_to_head:
            if result_b in self._head_to_head[result_a]:
                return self._head_to_head[result_a][result_b].this_elo_deltas()
        if result_b in self._head_to_head:
            if result_a in self._head_to_head[result_b]:
                return self._head_to_head[result_b][result_a].this_elo_deltas(get_other=True)
        print('ERROR no (%s:%s) or (%s:%s) in GED' % (result_a.driver().id(), result_a.team().uuid(),
                                                      result_b.driver().id(), result_b.team().uuid()),
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
