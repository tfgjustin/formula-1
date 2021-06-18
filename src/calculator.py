import argparse
import copy
import datetime
import math
import numpy as np
import ratings

from collections import defaultdict
from ratings import CarReliability, DriverReliability, Reliability
from scipy.stats import skew
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


_ALL = 'All'
_CAR = 'Car'
_DRIVER = 'Driver'


def validate_factors(argument):
    parts = argument.split('_')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError('Invalid factor: %s' % argument)
    try:
        initial = int(parts[0])
    except ValueError:
        raise argparse.ArgumentTypeError(
            'Factor %s has non-numeric initial value %s' % (argument, parts[0]))
    else:
        if initial < 0 or initial > 100:
            raise argparse.ArgumentTypeError(
                'Factor %s has invalid initial value %s' % (argument, parts[0]))

    try:
        year_step = int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            'Factor %s has non-numeric year-step value %s' % (argument, parts[1]))
    else:
        if year_step < 0:
            raise argparse.ArgumentTypeError(
                'Factor %s has invalid year-step value %s' % (argument, parts[1]))
    try:
        step_value = int(parts[2])
    except ValueError:
        raise argparse.ArgumentTypeError(
            'Factor %s has non-numeric step value %s' % (argument, parts[2]))
    else:
        if step_value < 0 or step_value > 100:
            raise argparse.ArgumentTypeError(
                'Factor %s has invalid step value %s' % (argument, parts[2]))
    current_year = datetime.datetime.now().year
    num_seasons = current_year - 1950
    if year_step > 0:
        max_value = initial + (int(num_seasons / year_step) * step_value)
    else:
        max_value = initial
    if max_value >= 100:
        raise argparse.ArgumentTypeError(
            'Factor spec %s will result in a final factor of %d' % (argument, max_value))
    return argument


def select_reference_results(event, car_factor, num_results=4):
    if event.type() == 'R':
        reliability = sorted([result for result in event.results()], key=lambda r: r.probability_complete_n_laps(1),
                             reverse=True)
        return reliability[:num_results]
    else:
        performance = sorted([result for result in event.results()],
                             key=lambda r: elo_rating_from_result(car_factor, r), reverse=True)
        return performance[:num_results]


def was_performance_win(result_this, result_other):
    return result_this.dnf_category() == '-' and result_other.dnf_category() == '-'


def elo_win_probability(r_a, r_b, denominator):
    """Standard logistic calculator, per
       https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    q_a = 10 ** (r_a / denominator)
    q_b = 10 ** (r_b / denominator)
    return q_a / (q_a + q_b)


def elo_rating_from_result(car_factor, result):
    rating = (1 - car_factor) * result.driver().rating().elo()
    rating += (car_factor * result.team().rating().elo())
    return rating


def assign_year_value_dict(spec, divisor, value_dict):
    parts = spec.split('_')
    initial = float(parts[0]) / divisor
    window = int(parts[1])
    step = float(parts[2]) / divisor
    current_year = datetime.datetime.now().year
    count = 1
    value_dict[1950] = initial
    current_value = initial
    for year in range(1951, current_year + 1):
        if count >= window:
            current_value += step
            count = 1
        else:
            count += 1
        value_dict[year] = current_value


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


class EventPrediction(object):

    def __init__(self, event, elo_denominator, k_factor_adjust, car_factor, base_points_per_position,
                 position_base_factor, base_car_reliability, base_new_car_reliability, base_driver_reliability,
                 team_reliability_new_events, debug_file):
        self._event = event
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
        if self._event.type() == 'R':
            CarReliability.fit_opening_lap(self._event.id())
            CarReliability.regress_opening_lap()
            DriverReliability.fit_opening_lap(self._event.id())
            DriverReliability.regress_opening_lap()
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
        # Identify a reference entrant and compare each entrant to that person.
        reference_results = select_reference_results(self._event, self._car_factor)
        if reference_results is None:
            return
        drivers = list()
        probabilities = np.ones([len(self._event.results())], dtype='float')
        idx = 0
        distance_km = self._event.total_distance_km()
        naive_car_probability = self._base_car_reliability.probability_finishing(race_distance_km=distance_km)
        naive_driver_probability = self._base_driver_reliability.probability_finishing(race_distance_km=distance_km)
        naive_all_probability = naive_car_probability * naive_driver_probability
        self._naive_probabilities = {
            _ALL: naive_all_probability, _CAR: naive_car_probability, _DRIVER: naive_driver_probability
        }
        for result in self._event.results():
            driver_finish = result.driver().rating().probability_finishing(race_distance_km=distance_km)
            car_finish = result.team().rating().probability_finishing(race_distance_km=distance_km)
            finish_probability = driver_finish * car_finish
            self.finish_probabilities()[result.driver().id()] = {
                'All': finish_probability, 'Car': car_finish, 'Driver': driver_finish
            }
            drivers.append(result.driver().id())
            win_probability = 1
            use_last = False
            count = 0
            for ref_result in reference_results:
                if result == ref_result:
                    use_last = True
                    continue
                if ref_result == reference_results[-1] and not use_last:
                    continue
                this_probability = self.get_win_probability(result, ref_result)
                if this_probability is None:
                    continue
                count += 1
                win_probability *= this_probability
            if win_probability == 1:
                probabilities[idx] = win_probability
            elif count:
                probabilities[idx] = win_probability / (1 - win_probability)
            else:
                probabilities[idx] = 0
            idx += 1
        probability_sum = np.sum(probabilities)
        # If the collective probability is greater than 0 (which it should be, since comparing someone against
        # themselves gives a 0.500 odds, which has an odds ratio of 1.0) then normalize the odds. Otherwise give
        # everyone the same odds.
        if probability_sum > 1e-3:
            probabilities /= probability_sum
        else:
            probabilities.fill(1.0 / len(self._event.results()))
        for idx in range(0, len(self._event.results())):
            self.win_probabilities()[drivers[idx]] = probabilities[idx]
        # Now that we have the odds of each one winning, predict the odds each one finishes in the top 3.
        self.predict_podium()

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


class Calculator(object):

    def __init__(self, args, base_filename):
        self._args = args
        self._driver_rating_file = open(base_filename + '.driver_ratings', 'w')
        self._team_rating_file = open(base_filename + '.team_ratings', 'w')
        self._summary_file = open(base_filename + '.summary', 'w')
        self._logfile = None
        if self._args.print_progress:
            self._logfile = open(base_filename + '.log', 'w')
        self._debug_file = None
        if self._args.print_debug:
            self._debug_file = open(base_filename + '.debug', 'w')
        self._predict_file = None
        if self._args.print_predictions:
            self._predict_file = open(base_filename + '.predict', 'w')
        self._position_base_dict = dict()
        self.create_position_base_dict()
        self._team_share_dict = dict()
        self.create_team_share_dict()
        self._oversample_rates = dict(
            {'Q': dict({'195': 3, '196': 2}),
             'R': dict({'195': 7, '196': 6, '197': 3, '198': 3, '199': 3, '200': 2})
             })
        self._base_car_reliability = CarReliability(
            default_decay_rate=self._args.team_reliability_decay,
            regress_percent=self._args.team_reliability_regress,
            regress_numerator=(self._args.team_reliability_lookback * Reliability.DEFAULT_KM_PER_RACE))
        self._base_new_car_reliability = Reliability(other=self._base_car_reliability)
        self._base_driver_reliability = DriverReliability(
            default_decay_rate=self._args.driver_reliability_decay,
            regress_percent=self._args.driver_reliability_regress,
            regress_numerator=(self._args.driver_reliability_lookback * Reliability.DEFAULT_KM_PER_RACE))
        self._full_h2h_log = list()
        self._elo_h2h_log = list()
        self._podium_odds_log = list()
        self._win_odds_log = list()
        self._finish_odds_log = {_ALL: list(), _CAR: list(), _DRIVER: list()}

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        On exit of this class make sure we close all the opened files.
        """
        self._driver_rating_file.close()
        self._team_rating_file.close()
        self._summary_file.close()
        if self._logfile is not None:
            self._logfile.close()
        if self._debug_file is not None:
            self._debug_file.close()
        if self._predict_file is not None:
            self._predict_file.close()

    def create_team_share_dict(self):
        """
        Create the dictionary which maps year to the percent of the performance which is due to the team (car).
        """
        assign_year_value_dict(self._args.team_share_spec, 100., self._team_share_dict)

    def create_position_base_dict(self):
        """
        Create the dictionary which maps year to the Elo boost each position on the starting grid gives you. This is
        somewhat analogous to home field advantage.
        """
        assign_year_value_dict(self._args.position_base_spec, 1., self._position_base_dict)

    def get_oversample_rate(self, event_type, event_id):
        """
        For a given event figure out what the oversample rate should be when calculating the overall error. This is
        because earlier seasons of F1 had fewer events and fewer entrants. This mitigates against us creating a model
        which works best for later era F1 but poorly for the earlier decades.
        """
        return self._oversample_rates[event_type].get(event_id[:3], 1)

    def run_all_ratings(self, loader):
        """
        Run the ratings for each year and print the output to the relevant files.
        """
        np.set_printoptions(precision=5, linewidth=300)
        print(('RaceID\tDriverID\tPlaced\tNumDrivers\tDnfReason\tEloPre\tEloPost\tEloDiff\tEffectPre\tEffectPost'
               + '\tKFEventsPre\tKFEventsPost\tKmSuccessPre\tKmSuccessPost\tKmFailurePre\tKmFailurePost'
               + '\tSuccessPre\tSuccessPost'
               ),
              file=self._driver_rating_file)
        print(('RaceID\tTeamUUID\tTeamID\tNumTeams\tEloPre\tEloPost\tEloDiff\tEffectPre\tEffectPost'
               + '\tKFEventsPre\tKFEventsPost\tKmSuccessPre\tKmSuccessPost\tKmFailurePre\tKmFailurePost'
               + '\tSuccessPre\tSuccessPost'
               ),
              file=self._team_rating_file)
        for year in sorted(loader.seasons().keys()):
            self.run_one_year(year, loader.seasons()[year])
        self.log_summary_errors()

    def run_one_year(self, year, season):
        """
        Run the model for one year.
        """
        events_dict = season.events()
        if self._logfile is not None:
            print('Running %d' % year, file=self._logfile)
        self._base_new_car_reliability.regress()
        for event_id in sorted(events_dict.keys()):
            event = events_dict[event_id]
            # Skip all the Indianapolis races since they were largely disjoint
            # from the rest of Formula 1.
            if 'Indianapolis' in event.name() and event.season() <= 1960:
                if self._logfile is not None:
                    print('Skipping %s (%s)' % (event.id(), event.name()), file=self._logfile)
                continue
            self.run_one_event(event)

    def run_one_event(self, event):
        """
        Run the model for a single event. This could be one qualifying session or one race.
        """
        if self._logfile is not None:
            print('  Event %s' % (event.id()), file=self._logfile)
        k_factor_adjust = 1
        elo_denominator = self._args.elo_exponent_denominator_race
        if event.type() == 'Q':
            # If this is a qualifying session, adjust all K-factors by a constant multiplier so fewer points flow
            # between the drivers and teams. Also use a different denominator since the advantage is much more
            # pronounced in qualifying than in races (e.g., a 100 point Elo advantage in qualifying gives you a much
            # better chance of finishing near the front than a 100 point Elo advantage in a race).
            k_factor_adjust = self._args.qualifying_kfactor_multiplier
            elo_denominator = self._args.elo_exponent_denominator_qualifying
        predictions = EventPrediction(event, elo_denominator, k_factor_adjust, self._team_share_dict[event.season()],
                                      self._position_base_dict[event.season()], self._args.position_base_factor,
                                      self._base_car_reliability, self._base_new_car_reliability,
                                      self._base_driver_reliability, self._args.team_reliability_new_events,
                                      self._debug_file)
        predictions.cache_ratings()
        # Predict the odds of each entrant either winning or finishing on the podium.
        predictions.predict_winner()
        predictions.start_updates()
        # Do the full pairwise comparison of each driver. In order to not calculate A vs B and B vs A (or A vs A) only
        # compare when the ID of A<B.
        for result_a in event.results():
            for result_b in event.results():
                if result_a.driver().id() < result_b.driver().id():
                    self.compare_results(event, predictions, result_a, result_b)
        self.update_all_reliability(event)
        predictions.commit_updates()
        self.log_results(predictions)

    def update_all_reliability(self, event):
        _OPENING_LAP_COUNT = 1
        if event.type() == 'Q':
            return
        driver_crash_laps = [
            result.laps() for result in event.results()
            if result.dnf_category() == 'driver' and result.laps() >= _OPENING_LAP_COUNT
        ]
        crash_laps = {lap: driver_crash_laps.count(lap) for lap in driver_crash_laps}
        opening_distance = _OPENING_LAP_COUNT * event.lap_distance_km()
        for result in event.results():
            if result.laps() >= _OPENING_LAP_COUNT or result.dnf_category() != 'car':
                base_probability = result.driver().rating().reliability().probability_finishing(
                    race_distance_km=opening_distance)
                num_laps = result.laps()
                did_fail = 0
                if num_laps < _OPENING_LAP_COUNT:
                    did_fail = 1
                else:
                    num_laps = _OPENING_LAP_COUNT
                DriverReliability.add_observation(base_probability, result.start_position(), num_laps, did_fail)
            if result.laps() >= _OPENING_LAP_COUNT or result.dnf_category() != 'driver':
                base_probability = result.team().rating().reliability().probability_finishing(
                    race_distance_km=opening_distance)
                num_laps = result.laps()
                did_fail = 0
                if num_laps < _OPENING_LAP_COUNT:
                    did_fail = 1
                else:
                    num_laps = _OPENING_LAP_COUNT
                CarReliability.add_observation(base_probability, num_laps, did_fail)
            if result.laps() < _OPENING_LAP_COUNT:
                continue
            km_success = event.lap_distance_km() * result.laps()
            km_car_failure = 0
            km_driver_failure = 0
            if result.dnf_category() == 'car':
                km_car_failure = self._args.team_reliability_failure_constant
            elif result.dnf_category() == 'driver':
                km_driver_failure = self._args.driver_reliability_failure_constant / crash_laps[result.laps()]
            if self._debug_file is not None:
                print(('Event %s Laps: %3d KmPerLap: %5.2f Driver: %s ThisKmSuccess: %5.1f ThisKmFailure: %5.1f '
                       + 'TotalKmSuccess: %8.1f TotalKmFailure: %8.3f ProbFinish: %.4f') % (
                          event.id(), event.num_laps(), event.lap_distance_km(), result.driver().id(), km_success,
                          km_driver_failure, result.driver().rating().reliability().km_success(),
                          result.driver().rating().reliability().km_failure(),
                          result.driver().rating().reliability().probability_finishing()
                      ),
                      file=self._debug_file)
            team_num_events = result.team().rating().k_factor().num_events()
            if team_num_events >= self._args.team_reliability_new_events:
                self._base_car_reliability.update(km_success, km_car_failure)
            else:
                self._base_new_car_reliability.update(km_success, km_car_failure)
            self._base_driver_reliability.update(km_success, km_driver_failure)
            result.team().rating().update_reliability(km_success, km_car_failure)
            result.driver().rating().update_reliability(km_success, km_driver_failure)

    def compare_results(self, event, predictions, result_a, result_b):
        # First do the overall win probabilities
        win_actual_a = 1 if result_a.end_position() < result_b.end_position() else 0
        win_prob_a = predictions.get_win_probability(result_a, result_b)
        if win_prob_a is None:
            if self._debug_file is not None:
                print('      Skip: Win Prob is None', file=self._debug_file)
            return
        # if result_a.laps() > 1 and result_b.laps() > 1:
        self.add_full_h2h_error(event, win_actual_a, win_prob_a)
        self.add_full_h2h_error(event, 1 - win_actual_a, 1 - win_prob_a)
        # Now do the performance (Elo-based) probabilities
        if not was_performance_win(result_a, result_b):
            if self._debug_file is not None:
                print('      Skip: At least one entrant DNF\'ed', file=self._debug_file)
            return
        elo_win_prob_a, rating_a, rating_b = predictions.get_elo_win_probability(result_a, result_b)
        if elo_win_prob_a is None:
            if self._debug_file is not None:
                print('      Skip: Elo Prob is None', file=self._debug_file)
            return
        car_delta, driver_delta = predictions.get_elo_deltas(result_a, result_b)
        if car_delta is None or driver_delta is None:
            if self._debug_file is not None:
                print('      Skip: Use', file=self._debug_file)
            return
        self.add_elo_h2h_error(event, win_actual_a, elo_win_prob_a)
        self.add_elo_h2h_error(event, 1 - win_actual_a, 1 - elo_win_prob_a)
        if not self.should_compare(rating_a, rating_b, predictions.elo_denominator()):
            if self._debug_file is not None:
                print('      Skip: SC', file=self._debug_file)
            return
        self.update_ratings(result_a, car_delta, driver_delta)
        self.update_ratings(result_b, -car_delta, -driver_delta)

    def should_compare(self, rating_a, rating_b, elo_denominator):
        return (abs(rating_a - rating_b) / elo_denominator) <= self._args.elo_compare_window

    def add_full_h2h_error(self, event, actual, prob):
        if event.type() == 'Q':
            return
        error_array = [event.id(), 0.5, prob, actual]
        for _ in range(self.get_oversample_rate(event.type(), event.id())):
            self._full_h2h_log.append(error_array)
            if self._predict_file is not None:
                print('FullH2H\t%s\t%.4f\t%.4f\t%d' % (event.id(), 0.5, prob, actual), file=self._predict_file)

    def add_elo_h2h_error(self, event, actual, prob):
        error_array = [event.id(), 0.5, prob, actual]
        for _ in range(self.get_oversample_rate(event.type(), event.id())):
            self._elo_h2h_log.append(error_array)
            if self._predict_file is not None:
                print('EloH2H\t%s\t%.4f\t%.4f\t%d' % (event.id(), 0.5, prob, actual), file=self._predict_file)

    def update_ratings(self, result, car_delta, driver_delta):
        result.driver().rating().update(driver_delta)
        result.team().rating().update(car_delta)
        if self._debug_file is not None:
            print('        DD: %5.2f CD: %5.2f D: %s T: %s' % (
                driver_delta, car_delta, result.driver().id(), result.team().uuid()),
                  file=self._debug_file)

    def log_results(self, predictions):
        event = predictions.event()
        for team in sorted(event.teams(), key=lambda t: t.id()):
            self.log_team_results(event, predictions, team)
        self.log_average_team(event, predictions, self._base_car_reliability, 'TeamBase')
        self.log_average_team(event, predictions, self._base_new_car_reliability, 'TeamNew')
        driver_results = {result.driver().id(): result for result in event.results()}
        num_drivers = len(driver_results)
        for driver_id, result in sorted(driver_results.items()):
            self.log_driver_results(event, predictions, num_drivers, result)
            self.log_finish_probabilities(event, predictions, driver_id, result)
            self.log_win_probabilities(event, predictions, driver_id, result)
            self.log_podium_probabilities(event, predictions, driver_id, result)

    def log_team_results(self, event, predictions, team):
        rating_before = predictions.team_before(team.id()).rating()
        rating_after = team.rating()
        elo_diff = rating_after.elo() - rating_before.elo()
        before_effect = (rating_before.elo() - self._args.team_elo_initial) * self._team_share_dict[event.season()]
        after_effect = (rating_after.elo() - self._args.team_elo_initial) * self._team_share_dict[event.season()]
        reliability_before = rating_before.reliability()
        if reliability_before is None:
            reliability_before = Reliability()
        reliability_after = rating_after.reliability()
        if reliability_after is None:
            reliability_after = Reliability()
        print(('S%s\t%s\t%s\t%d\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%5.1f\t%5.1f\t%8.2f\t%8.2f'
               + '\t%8.4f\t%8.4f\t%.6f\t%.6f') % (
            event.id(), team.uuid(), team.id(), len(event.teams()), rating_before.elo(), rating_after.elo(), elo_diff,
            before_effect, after_effect, rating_before.k_factor().num_events(), rating_after.k_factor().num_events(),
            reliability_before.km_success(), reliability_after.km_success(),
            reliability_before.km_failure(), reliability_after.km_failure(),
            rating_before.probability_finishing(), rating_after.probability_finishing()
        ),
              file=self._team_rating_file)

    def log_average_team(self, event, predictions, base_reliability, base_name):
        matching_teams = [t for t in event.teams()
                          if t.rating().reliability() is not None
                          and t.rating().reliability().template() == base_reliability]
        rating_before = sum([predictions.team_before(t.id()).rating().elo() for t in matching_teams])
        rating_after = sum([t.rating().elo() for t in matching_teams])
        before_effect = 0
        after_effect = 0
        if matching_teams:
            rating_before /= len(matching_teams)
            rating_after /= len(matching_teams)
            before_effect = (rating_before - self._args.team_elo_initial) * self._team_share_dict[event.season()]
            after_effect = (rating_after - self._args.team_elo_initial) * self._team_share_dict[event.season()]
        elo_diff = rating_after - rating_before
        print(('S%s\t%s\t%s\t%d\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%5.1f\t%5.1f\t%8.2f\t%8.2f'
               + '\t%8.4f\t%8.4f\t%.6f\t%.6f') % (
            event.id(), base_name, base_name, len(matching_teams), rating_before, rating_after, elo_diff,
            before_effect, after_effect, 0, 0,
            0, base_reliability.km_success(),
            0, base_reliability.km_failure(),
            0, base_reliability.probability_finishing()
        ),
              file=self._team_rating_file)
        return

    def log_driver_results(self, event, predictions, num_drivers, result):
        driver = result.driver()
        placed = result.end_position()
        dnf = result.dnf_category()
        rating_before = predictions.driver_before(result.driver().id()).rating()
        rating_after = result.driver().rating()
        elo_diff = rating_after.elo() - rating_before.elo()
        before_effect = (rating_before.elo() - self._args.driver_elo_initial)
        before_effect *= (1 - self._team_share_dict[event.season()])
        after_effect = (rating_after.elo() - self._args.driver_elo_initial)
        after_effect *= (1 - self._team_share_dict[event.season()])
        reliability_before = rating_before.reliability()
        if reliability_before is None:
            reliability_before = Reliability()
        reliability_after = rating_after.reliability()
        if reliability_after is None:
            reliability_after = Reliability()
        print(('S%s\t%s\t%d\t%d\t%s\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%5.1f\t%5.1f\t%8.2f\t%8.2f'
               + '\t%8.4f\t%8.4f\t%.6f\t%.6f') % (
            event.id(), driver.id(), placed, num_drivers, dnf, rating_before.elo(), rating_after.elo(), elo_diff,
            before_effect, after_effect, rating_before.k_factor().num_events(), rating_after.k_factor().num_events(),
            reliability_before.km_success(), reliability_after.km_success(),
            reliability_before.km_failure(), reliability_after.km_failure(),
            rating_before.probability_finishing(), rating_after.probability_finishing()
        ),
              file=self._driver_rating_file)
        return

    def log_finish_probabilities(self, event, predictions, driver_id, result):
        if event.type() != 'R':
            return
        # Identifiers for the entrant, team (car), and driver
        full_identifier = '%s:%s' % (result.team().uuid(), result.driver().id())
        identifiers = {
            _ALL: full_identifier, _CAR: result.team().uuid(), _DRIVER: result.driver().id()
        }
        # The probabilities that the average entrant will finish and that this entrant will finish
        naive_full_probabilities = predictions.naive_finish_probabilities()
        full_probabilities = predictions.finish_probabilities().get(driver_id)
        # Regardless of whether they finished the race or not this is the probability that they got as far as they
        # did in the race.
        distance_success_km = result.laps() * event.lap_distance_km()
        car_probability = result.team().rating().probability_finishing(race_distance_km=distance_success_km)
        driver_probability = result.driver().rating().probability_finishing(race_distance_km=distance_success_km)
        all_probability = car_probability * driver_probability
        # The naive (baseline) odds that they got as far as they did
        team_num_events = result.team().rating().k_factor().num_events()
        if team_num_events >= self._args.team_reliability_new_events:
            naive_car_probability = self._base_car_reliability.probability_finishing(
                race_distance_km=distance_success_km)
        else:
            naive_car_probability = self._base_new_car_reliability.probability_finishing(
                race_distance_km=distance_success_km)
        naive_driver_probability = self._base_driver_reliability.probability_finishing(
            race_distance_km=distance_success_km)
        naive_all_probability = naive_driver_probability * naive_car_probability
        completed_error_arrays = {
            _ALL: [event.id(), naive_all_probability, all_probability, 1],
            _CAR: [event.id(), naive_car_probability, car_probability, 1],
            _DRIVER: [event.id(), naive_driver_probability, driver_probability, 1]
        }
        # These are in case the entrant fails to get the full distance
        failed_error_arrays = {
            _ALL: [event.id(), naive_full_probabilities[_ALL], full_probabilities[_ALL], 0],
            _CAR: [event.id(), naive_full_probabilities[_CAR], full_probabilities[_CAR], 0],
            _DRIVER: [event.id(), naive_full_probabilities[_DRIVER], full_probabilities[_DRIVER], 0]
        }
        for _ in range(self.get_oversample_rate(event.type(), event.id())):
            for mode in [_ALL, _CAR, _DRIVER]:
                self._finish_odds_log[mode].append(completed_error_arrays[mode])
                self.log_one_finish_probability(
                    event, mode, identifiers.get(mode), completed_error_arrays[mode][-2], 1)
                if self._debug_file is not None:
                    for km in range(math.floor(distance_success_km)):
                        print('Reliable\t%s\t%d\t1' % (mode, km), file=self._debug_file)
            # If they didn't finish the race, then either the car succeeded up until it crapped out and the driver
            # succeeded until they didn't, or vice versa.
            failed_modes = []
            if result.dnf_category() == 'car':
                # The car didn't make it to the end.
                failed_modes = [_ALL, _CAR]
            elif result.dnf_category() == 'driver':
                # Blame the driver are the entire package for not making it to the end.
                failed_modes = [_ALL, _DRIVER]
            for mode in failed_modes:
                self._finish_odds_log[mode].append(failed_error_arrays[mode])
                self.log_one_finish_probability(
                    event, mode, identifiers.get(mode), failed_error_arrays[mode][-2], 0)
                if self._debug_file is not None:
                    print('Reliable\t%s\t%d\t0' % (mode, math.ceil(distance_success_km)), file=self._debug_file)

    def log_one_finish_probability(self, event, mode, identifier, probability, correct):
        if self._predict_file is None:
            return
        print('Finish\t%s\t%s\t%s\t%.6f\t%d' % (event.id(), mode, identifier, probability, correct),
              file=self._predict_file)

    def log_win_probabilities(self, event, predictions, driver_id, result):
        win_odds = predictions.win_probabilities().get(driver_id)
        won = 1 if result.end_position() == 1 else 0
        naive_odds = 1.0 / len(event.drivers())
        error_array = [event.id(), naive_odds, win_odds, won]
        for _ in range(self.get_oversample_rate(event.type(), event.id())):
            self._win_odds_log.append(error_array)
            if self._predict_file is not None:
                print('WinOR\t%s\t%s\t%.6f\t%.6f\t%d' % (
                    event.id(), driver_id, naive_odds, win_odds, won
                ),
                      file=self._predict_file)

    def log_podium_probabilities(self, event, predictions, driver_id, result):
        podium_odds = predictions.podium_probabilities().get(driver_id)
        podium = 1 if result.end_position() <= 3 else 0
        naive_odds = 1.0 / len(event.drivers())
        error_array = [event.id(), naive_odds, podium_odds, podium]
        for _ in range(self.get_oversample_rate(event.type(), event.id())):
            self._podium_odds_log.append(error_array)
            if self._predict_file is not None:
                print('PodiumOR\t%s\t%s\t%.6f\t%.6f\t%d' % (
                    event.id(), driver_id, naive_odds, podium_odds, podium
                ),
                      file=self._predict_file)

    def log_summary_errors(self):
        for decade in range(195, 203):
            self.log_summary_errors_for_decade(decade=str(decade))
        self.log_summary_errors_for_decade()

    def log_summary_errors_for_decade(self, decade=''):
        self.log_reliability_summary(decade=decade)
        self.log_win_summary(decade=decade)
        self.log_podium_summary(decade=decade)
        self.log_elo_summary(decade=decade)
        self.log_full_summary(decade=decade)

    @staticmethod
    def get_matching_errors(error_log, decade, event_type):
        matching = [
            error_array for error_array in error_log
            if (event_type is None or error_array[0].endswith(event_type)) and error_array[0].startswith(decade)
        ]
        return matching

    def log_full_summary(self, decade=''):
        matching_errors = self.get_matching_errors(self._full_h2h_log, decade, None)
        self.log_one_error_log('FullH2H', matching_errors, decade, None)
        self.log_one_pr_auc('FullH2H', matching_errors, decade, None)

    def log_elo_summary(self, decade=''):
        for event_type in ['Q', 'R']:
            matching_errors = self.get_matching_errors(self._elo_h2h_log, decade, event_type)
            self.log_one_error_log('EloH2H', matching_errors, decade, event_type)
            self.log_one_pr_auc('EloH2H', matching_errors, decade, event_type)

    def log_win_summary(self, decade=''):
        for event_type in ['Q', 'R']:
            matching_errors = self.get_matching_errors(self._win_odds_log, decade, event_type)
            self.log_one_error_log('Win', matching_errors, decade, event_type)
            self.log_one_pr_auc('Win', matching_errors, decade, event_type)

    def log_podium_summary(self, decade=''):
        for event_type in ['Q', 'R']:
            matching_errors = self.get_matching_errors(self._podium_odds_log, decade, event_type)
            self.log_one_error_log('Podium', matching_errors, decade, event_type)
            self.log_one_pr_auc('Podium', matching_errors, decade, event_type)

    def log_reliability_summary(self, decade=''):
        # For these we use ROC curves
        for mode in [_ALL, _CAR, _DRIVER]:
            matching_errors = self.get_matching_errors(self._finish_odds_log[mode], decade, None)
            self.log_one_error_log('Fin' + mode, matching_errors, decade, None)
            self.log_one_roc_auc('Fin' + mode, matching_errors, decade, None)

    def log_one_error_log(self, tag, error_log, decade, event_type):
        squared_errors = [
            (error_array[-2] - error_array[-1]) ** 2 for error_array in error_log
        ]
        error_sum = sum(squared_errors)
        error_count = len(squared_errors)
        if event_type is None:
            event_type = ''
        elif not event_type:
            event_type = '-All'
        else:
            event_type = '-' + event_type
        if not decade:
            decade = 'Total'
        else:
            decade = decade + '0'
        print('%s%s\tMSE\t%s\t%.6f\t%7.1f\t%5d' % (
            tag, event_type, decade, error_sum / error_count, error_sum, error_count),
              file=self._summary_file)

    @staticmethod
    def get_true_probs_naive_residuals(error_log):
        is_true = [error_array[-1] for error_array in error_log]
        probs = [error_array[-2] for error_array in error_log]
        naive = [error_array[-3] for error_array in error_log]
        residuals = [error_array[-1] - error_array[-2] for error_array in error_log]
        return is_true, probs, naive, residuals

    def log_one_pr_auc(self, tag, error_log, decade, event_type):
        is_true, probs, naive, residuals = self.get_true_probs_naive_residuals(error_log)
        precision, recall, _ = precision_recall_curve(is_true, probs)
        area_under_curve = auc(recall, precision)
        if decade:
            decade = decade + '0'
        else:
            decade = 'Total'
        if event_type is None:
            event_type = ''
        elif not event_type:
            event_type = '-All'
        else:
            event_type = '-' + event_type
        skew_value = float(skew(np.asarray(residuals)))
        print('%s%s\tSkew\t%s\t%.6f\t---\t%5d' % (tag, event_type, decade, skew_value, len(is_true)),
              file=self._summary_file)
        print('%s%s\tAUC\t%s\t%.6f\t--\t%5d' % (tag, event_type, decade, area_under_curve, len(is_true)),
              file=self._summary_file)
        naive_precision, naive_recall, _ = precision_recall_curve(is_true, naive)
        naive_area_under_curve = auc(naive_recall, naive_precision)
        print('%s%s\tAUCN\t%s\t%.6f\t--\t%5d' % (tag, event_type, decade, naive_area_under_curve, len(is_true)),
              file=self._summary_file)
        area_under_curve_above_naive = (area_under_curve - naive_area_under_curve) / (1 - naive_area_under_curve)
        print('%s%s\tAUCAN\t%s\t%.6f\t--\t%5d' % (tag, event_type, decade, area_under_curve_above_naive, len(is_true)),
              file=self._summary_file)

    def log_one_roc_auc(self, tag, error_log, decade, event_type):
        is_true, probs, naive, residuals = self.get_true_probs_naive_residuals(error_log)
        if decade:
            decade = decade + '0'
        else:
            decade = 'Total'
        if not event_type:
            event_type = 'All'
        count = len(is_true)
        skew_value = float(skew(np.asarray(residuals)))
        print('%s%s\tSkew\t%s\t%.6f\t---\t%5d' % (tag, event_type, decade, skew_value, count),
              file=self._summary_file)
        area_under_curve = roc_auc_score(is_true, probs)
        print('%s\tAUC\t%s\t%.6f\t--\t%5d' % (tag, decade, area_under_curve, count),
              file=self._summary_file)
        naive_area_under_curve = roc_auc_score(is_true, naive)
        print('%s\tAUCN\t%s\t%.6f\t--\t%5d' % (tag, decade, naive_area_under_curve, count),
              file=self._summary_file)
        area_under_curve_above_naive = (area_under_curve - naive_area_under_curve) / (1 - naive_area_under_curve)
        print('%s\tAUCAN\t%s\t%.6f\t--\t%5d' % (tag, decade, area_under_curve_above_naive, count),
              file=self._summary_file)
