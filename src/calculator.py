import argparse
import datetime
import math
import numpy as np
import ratings

from collections import defaultdict
from ratings import CarReliability, DriverReliability, Reliability
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


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


def select_reference_result(event):
    reliability = sorted([result for result in event.results()], key=lambda r: r.probability_complete_n_laps(1),
                         reverse=True)
    max_reliability = reliability[0].probability_complete_n_laps(1)
    reference_reliability_results = list([r for r in reliability
                                          if abs(r.probability_complete_n_laps(1) - max_reliability) < 1e-6])
    return reference_reliability_results[0]


def elo_win_probability(r_a, r_b, denominator):
    """Standard logistic calculator, per
       https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    q_a = 10 ** (r_a / denominator)
    q_b = 10 ** (r_b / denominator)
    return q_a / (q_a + q_b)


def elo_rating_from_result(car_factor, result):
    rating = (1 - car_factor) * result.driver().rating().rating()
    rating += (car_factor * result.team().rating().rating())
    return rating


def have_same_team(result_a, result_b):
    if result_a.team() is None or result_b.team() is None:
        return False
    return result_a.team().uuid() == result_b.team().uuid()


def allocate_results(result_a, result_b, outfile):
    """
    xxxx    Finished    Crashed     Car
    xxxx    Sa  Di      Sa  Di      Sa  Di
    --------------------------------------
    Fin.    D   D+C     D   D       x   x
    Crash   D   D       D   D       x   x
    Car     x   x       x   x       x   x
    """
    if result_a.dnf_category() == 'car' or result_b.dnf_category() == 'car':
        return None, None
    if result_a.dnf_category() == 'driver' or result_b.dnf_category() == 'driver':
        return None, None
    same_team = have_same_team(result_a, result_b)
    use_driver = True
    use_team = True
    if same_team:
        use_team = False
        if outfile is not None:
            print('--xtx', file=outfile)
    else:
        use_team = True
        if outfile is not None:
            print('--xTx', file=outfile)
    return use_driver, use_team


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
        return self._result_this.end_position() < self._result_other.end_position()

    def was_performance_win(self):
        return self._result_this.dnf_category() != '-' and self._result_other.dnf_category() != '_'

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
        this_elo_probability = self.this_elo_probability()
        probability_this_complete = self._result_this.probability_complete_n_laps(num_laps)
        probability_other_complete = self._result_other.probability_complete_n_laps(num_laps)
        probability_both_complete = probability_this_complete * probability_other_complete
        probability_this_not_other = probability_this_complete * (1 - probability_other_complete)
        probability_this_further_than_other = self.probability_this_further_than_other()
        this_elo_probability *= probability_both_complete
        full_this_probability = this_elo_probability + probability_this_not_other + probability_this_further_than_other
        full_this_probability += (0.5 * self._tie_probability)
        print('%s ThisElo: %.7f x %.7f x %.7f = %.7f and PTNO: %.7f PTFTO: %.7f PThisWin: %.7f %s:%s vs %s:%s' % (
            self._event.id(), self.this_elo_probability(), probability_this_complete, probability_other_complete,
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
            return None
        if self._this_elo_probability is not None:
            if get_other:
                return 1 - self._this_elo_probability
            else:
                return self._this_elo_probability
        self._rating_this = elo_rating_from_result(self._car_factor, self._result_this)
        self._rating_this += self.this_elo_start_position_advantage()
        self._rating_other = elo_rating_from_result(self._car_factor, self._result_other)
        self._this_elo_probability = elo_win_probability(self._rating_this, self._rating_other, self._elo_denominator)
        print('%s TEP: (%.1f + %.1f) vs (%.1f) = %.6f' % (
            self._event.id(), self._rating_this, self.this_elo_start_position_advantage(),
            self._rating_other, self._this_elo_probability), file=self._debug_file)
        if get_other:
            return 1 - self._this_elo_probability
        else:
            return self._this_elo_probability

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
        if have_same_team(self._result_this, self._result_other):
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
                 position_base_factor, debug_file):
        self._event = event
        self._elo_denominator = elo_denominator
        self._k_factor_adjust = k_factor_adjust
        self._car_factor = car_factor
        self._base_points_per_position = base_points_per_position
        self._position_base_factor = position_base_factor
        self._drivers_before = dict()
        self._drivers_after = dict()
        self._teams_before = dict()
        self._teams_after = dict()
        self._win_probabilities = dict()
        self._podium_probabilities = dict()
        self._finish_probabilities = dict()
        self._position_base_dict = None
        self._team_share_dict = None
        self._head_to_head = defaultdict(dict)
        self._debug_file = debug_file

    def predict_winner(self):
        """
        Predict the probability of each entrant winning. Use the odds ratio.
        https://en.wikipedia.org/wiki/Odds_ratio
        """
        self.predict_all_head_to_head()
        # Identify a reference entrant and compare each entrant to that person.
        reference_result = select_reference_result(self._event)
        if reference_result is None:
            return
        drivers = list()
        probabilities = np.ones([len(self._event.results())], dtype='float')
        idx = 0
        distance_km = self._event.total_distance_km()
        for result in self._event.results():
            driver_finish = result.driver().rating().probability_finishing(race_distance_km=distance_km)
            car_finish = result.team().rating().probability_finishing(race_distance_km=distance_km)
            finish_probability = driver_finish * car_finish
            self.finish_probabilities()[result.driver().id()] = {
                'All': finish_probability, 'Car': car_finish, 'Driver': driver_finish
            }
            drivers.append(result.driver().id())
            # win_probability_new_elo = predictions.get_elo_win_probability(result, reference_result)
            win_probability_new_all = self.get_win_probability(result, reference_result)
            if win_probability_new_all is None:
                continue
            probabilities[idx] = win_probability_new_all / (1 - win_probability_new_all)
            idx += 1
        probability_sum = np.sum(probabilities)
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
                                                      result_b.driver().id(), result_b.team().uuid()))
        return None

    def get_elo_win_probability(self, result_a, result_b):
        if result_a in self._head_to_head:
            if result_b in self._head_to_head[result_a]:
                return self._head_to_head[result_a][result_b].this_elo_probability()
        if result_b in self._head_to_head:
            if result_a in self._head_to_head[result_b]:
                return self._head_to_head[result_b][result_a].this_elo_probability(get_other=True)
        print('ERROR no (%s:%s) or (%s:%s) in GEWP' % (result_a.driver().id(), result_a.team().uuid(),
                                                       result_b.driver().id(), result_b.team().uuid()))
        return None

    def event(self):
        return self._event

    def drivers_before(self):
        return self._drivers_before

    def teams_before(self):
        return self._teams_before

    def drivers_after(self):
        return self._drivers_after

    def teams_after(self):
        return self._teams_after

    def win_probabilities(self):
        return self._win_probabilities

    def podium_probabilities(self):
        return self._podium_probabilities

    def finish_probabilities(self):
        return self._finish_probabilities


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
        self._team_share_dict = dict()
        self._year_error_sum = dict({'Q': defaultdict(float), 'R': defaultdict(float)})
        self._year_error_count = dict({'Q': defaultdict(int), 'R': defaultdict(int)})
        self._decade_error_sum = dict({'Q': defaultdict(float), 'R': defaultdict(float)})
        self._decade_error_count = dict({'Q': defaultdict(int), 'R': defaultdict(int)})
        self._total_error_sum = 0
        self._total_error_count = 0
        self._finish_pred_true = dict({'All': list(), 'Car': list(), 'Driver': list()})
        self._finish_pred_prob = dict({'All': list(), 'Car': list(), 'Driver': list()})
        self._win_error_sum = dict({'Q': 0.0, 'R': 0.0})
        self._win_error_count = dict({'Q': 0, 'R': 0})
        self._podium_error_sum = dict({'Q': 0.0, 'R': 0.0})
        self._podium_error_count = dict({'Q': 0, 'R': 0})
        self._podium_pred_prob = dict({'Q': list(), 'R': list()})
        self._podium_pred_true = dict({'Q': list(), 'R': list()})
        self._calibrate_num_correct = defaultdict(int)
        self._calibrate_num_total = defaultdict(int)
        self.create_position_base_dict()
        self.create_team_share_dict()
        self._oversample_rates = dict(
            {'Q': dict({'195': 3, '196': 2}),
             'R': dict({'195': 7, '196': 6, '197': 3, '198': 3, '199': 3, '200': 2})
             })
        self._base_car_reliability = CarReliability(
            default_decay_rate=self._args.team_reliability_decay,
            regress_percent=self._args.team_reliability_regress,
            regress_numerator=(self._args.team_reliability_lookback * Reliability.DEFAULT_KM_PER_RACE))
        self._base_driver_reliability = DriverReliability(
            default_decay_rate=self._args.driver_reliability_decay,
            regress_percent=self._args.driver_reliability_regress,
            regress_numerator=(self._args.driver_reliability_lookback * Reliability.DEFAULT_KM_PER_RACE))

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
               + '\tProbFinish\tProbDriverFinish\tProbCarFinish'),
              file=self._driver_rating_file)
        print('RaceID\tTeamUUID\tTeamID\tNumTeams\tEloPre\tEloPost\tEffectPre\tEffectPost\tProbFinish',
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
                                      self._debug_file)
        # Predict the odds of each entrant either winning or finishing on the podium.
        predictions.predict_winner()
        event.collect_ratings(predictions.drivers_before(), predictions.teams_before())
        event.start_updates(self._base_car_reliability, self._base_driver_reliability)
        # Do the full pairwise comparison of each driver. In order to not calculate A vs B and B vs A (or A vs A) only
        # compare when the ID of A<B.
        for result_a in event.results():
            for result_b in event.results():
                if result_a.driver().id() >= result_b.driver().id():
                    continue
                self.compare_results(event, predictions, result_a, result_b, k_factor_adjust, elo_denominator)
        self.update_all_reliability(event)
        event.commit_updates()
        event.collect_ratings(predictions.drivers_after(), predictions.teams_after())
        self.log_results(event, predictions)

    def update_all_reliability(self, event):
        if event.type() == 'Q':
            return
        for result in event.results():
            if result.laps() <= 1:
                continue
            km_success = event.lap_distance_km() * result.laps()
            km_car_failure = 0
            km_driver_failure = 0
            if result.dnf_category() == 'car':
                km_car_failure = self._args.team_reliability_failure_constant
            elif result.dnf_category() == 'driver':
                km_driver_failure = self._args.driver_reliability_failure_constant
            if self._debug_file is not None:
                print(('Event %s Laps: %3d KmPerLap: %5.2f Driver: %s ThisKmSuccess: %5.1f ThisKmFailure: %5.1f '
                       + 'TotalKmSuccess: %8.1f TotalKmFailure: %8.3f ProbFinish: %.4f') % (
                          event.id(), event.num_laps(), event.lap_distance_km(), result.driver().id(), km_success,
                          km_driver_failure, result.driver().rating().reliability().km_success(),
                          result.driver().rating().reliability().km_failure(),
                          result.driver().rating().reliability().probability_finishing()
                      ),
                      file=self._debug_file)
            self._base_car_reliability.update(km_success, km_car_failure)
            self._base_driver_reliability.update(km_success, km_driver_failure)
            result.team().rating().update_reliability(km_success, km_car_failure)
            result.driver().rating().update_reliability(km_success, km_driver_failure)

    def create_merged_rating(self, season, result):
        car_factor = self._team_share_dict[season]
        rating = result.driver().rating().rating()
        rating *= (1 - car_factor)
        rating += (car_factor * result.team().rating().rating())
        k_factor = result.driver().rating().k_factor().factor()
        k_factor *= (1 - car_factor)
        k_factor += (car_factor * result.team().rating().k_factor().factor())
        if self._debug_file is not None and False:  # TODO: Fix this
            print('        R: %4d KF: %2d NR: %2d SP: %2d EP: %2d PFB: %2d DNF: %6s D: %s T: %s' % (
                rating, k_factor, result.num_racers(), result.start_position(), result.end_position(),
                result.position_from_back(), result.dnf_category(), result.driver().id(), result.team().uuid()),
                  file=self._debug_file)
        return rating, k_factor, result.position_from_back()

    def compare_results(self, event, predictions, result_a, result_b, k_factor_adjust, elo_denominator,
                        update_ratings=True, update_errors=True):
        rating_a, k_factor_a, pfb_a = self.create_merged_rating(event.season(), result_a)
        rating_b, k_factor_b, pfb_b = self.create_merged_rating(event.season(), result_b)
        if not k_factor_b or not k_factor_b:
            if self._debug_file is not None:
                print('      Skip: K', file=self._debug_file)
            return None
        k_factor = ((k_factor_a + k_factor_b) / 2) * k_factor_adjust
        start_position_advantage = self.start_position_advantage(event, pfb_a, pfb_b)
        win_prob_a = elo_win_probability(rating_a + start_position_advantage, rating_b, elo_denominator)
        if self._debug_file is not None:
            print('%s OEP: (%.1f + %.1f) vs (%.1f) = %.6f' % (
                event.id(), rating_a, start_position_advantage, rating_b, win_prob_a
            ),
                  file=self._debug_file)
        win_prob_a = predictions.get_elo_win_probability(result_a, result_b)
        if win_prob_a is None:
            print('Invalid comparison XXX')
            return None
        win_actual_a = 1 if result_a.end_position() < result_b.end_position() else 0
        use_drivers, use_team = allocate_results(result_a, result_b, None)  # Possibly set this back to self._debug_file
        if use_drivers is None or use_team is None:
            if self._debug_file is not None and False:  # TODO: Fix this
                print('      Skip: Use', file=self._debug_file)
            return win_prob_a
        if update_errors:
            self.add_h2h_error(event, rating_a, rating_b, win_actual_a, win_prob_a)
            self.add_h2h_error(event, rating_b, rating_a, 1 - win_actual_a, 1 - win_prob_a)
        if not self.should_compare(rating_a, rating_b):
            if self._debug_file is not None and False:  # TODO: Fix this
                print('      Skip: SC', file=self._debug_file)
            return win_prob_a
        if not update_ratings:
            return win_prob_a
        delta_a = k_factor * (win_actual_a - win_prob_a)
        self.update_ratings(event.season(), result_a, use_drivers, use_team, delta_a)
        self.update_ratings(event.season(), result_b, use_drivers, use_team, -delta_a)
        return win_prob_a

    def start_position_advantage(self, event, from_back_a, from_back_b):
        season = event.season()
        position_base = self._position_base_dict[season]
        position_diffs = abs(from_back_a - from_back_b)
        factor = position_diffs
        if abs(self._args.position_base_factor - 1.0) > 1e-2:
            factor = 1 - math.pow(self._args.position_base_factor, position_diffs)
            factor /= 1 - self._args.position_base_factor
        if from_back_a > from_back_b:
            return factor * position_base
        else:
            return -factor * position_base

    def should_compare(self, rating_a, rating_b):
        return abs(rating_a - rating_b) <= self._args.elo_compare_window

    def add_h2h_error(self, event, rating_a, rating_b, actual, prob):
        event_id = event.id()
        event_type = event.type()
        year = int(event_id[:4])
        decade = int(year / 10) * 10
        error = actual - prob
        squared_error = error * error
        bucket = int(round(100 * prob) / 2) * 2
        for _ in range(self.get_oversample_rate(event_type, event_id)):
            self._calibrate_num_total[bucket] += 1
            self._calibrate_num_correct[bucket] += actual
            self._year_error_sum[event_type][year] += squared_error
            self._year_error_count[event_type][year] += 1
            self._decade_error_sum[event_type][decade] += squared_error
            self._decade_error_count[event_type][decade] += 1
            self._total_error_sum += squared_error
            self._total_error_count += 1
            if self._predict_file is not None:
                print('H2H\t%s\t%.1f\t%.1f\t%.4f\t%d' % (event.id(), rating_a, rating_b, prob, actual),
                      file=self._predict_file)

    def update_ratings(self, season, result, use_drivers, use_teams, delta):
        if use_teams:
            car_factor = self._team_share_dict[season]
            car_delta = car_factor * delta
            result.driver().rating().update(delta - car_delta)
            result.team().rating().update(car_delta)
            if self._debug_file is not None:
                print('        DD: %5.2f CD: %5.2f D: %s T: %s' % (
                    delta - car_delta, car_delta, result.driver().id(), result.team().uuid()),
                      file=self._debug_file)
        else:
            # Only drivers
            if self._debug_file is not None:
                print('        DD: %5.2f CD: ----- D: %s T: %s' % (
                    delta, result.driver().id(), result.team().uuid()),
                      file=self._debug_file)
            result.driver().rating().update(delta)

    def log_results(self, event, predictions):
        driver_results = {result.driver().id(): result for result in event.results()}
        num_drivers = len(driver_results)
        for driver_id, result in sorted(driver_results.items()):
            self.log_driver_results(event, predictions, num_drivers, result)
            self.log_finish_probabilities(event, predictions, driver_id, result)
            self.log_win_probabilities(event, predictions, driver_id, result)
            self.log_podium_probabilities(event, predictions, driver_id, result)
        for team in sorted(predictions.teams_after().keys(), key=lambda t: t.id()):
            self.log_team_results(event, predictions, team)

    def log_driver_results(self, event, predictions, num_drivers, result):
        driver = result.driver()
        placed = result.end_position()
        dnf = result.dnf_category()
        before = predictions.drivers_before().get(driver)
        after = predictions.drivers_after().get(driver)
        before_effect = (before - self._args.driver_elo_initial) * (1 - self._team_share_dict[event.season()])
        after_effect = (after - self._args.driver_elo_initial) * (1 - self._team_share_dict[event.season()])
        finish_probabilities = predictions.finish_probabilities().get(driver.id())
        print('S%s\t%s\t%d\t%d\t%s\t%.1f\t%.1f\t%6.1f\t%6.1f\t%6.1f\t%.5f\t%.5f\t%.5f' % (
            event.id(), driver.id(), placed, num_drivers, dnf, before, after, after - before,
            before_effect, after_effect, finish_probabilities.get('All'), finish_probabilities.get('Car'),
            finish_probabilities.get('Driver')),
              file=self._driver_rating_file)
        return

    def log_team_results(self, event, predictions, team):
        before = predictions.teams_before().get(team)
        after = predictions.teams_after().get(team)
        before_effect = (before - self._args.team_elo_initial) * self._team_share_dict[event.season()]
        after_effect = (after - self._args.team_elo_initial) * self._team_share_dict[event.season()]
        prob_finish = team.rating().reliability().probability_finishing()
        print('S%s\t%s\t%s\t%d\t%.1f\t%.1f\t%6.1f\t%6.1f\t%6.1f\t%.5f' % (
            event.id(), team.uuid(), team.id(), len(predictions.teams_after()), before, after, after - before,
            before_effect, after_effect, prob_finish
        ), file=self._team_rating_file)

    def log_finish_probabilities(self, event, predictions, driver_id, result):
        if event.type() != 'R':
            return
        if result.laps() <= 1:
            # Ignore opening-lap scraps for right now.
            return
        full_probabilities = predictions.finish_probabilities().get(driver_id)
        # Regardless of whether they finished the race or not this is the probability that they got as far as they
        # did in the race.
        distance_success_km = result.laps() * event.lap_distance_km()
        car_probability = result.team().rating().probability_finishing(race_distance_km=distance_success_km)
        driver_probability = result.driver().rating().probability_finishing(race_distance_km=distance_success_km)
        completed_probabilities = {
            'All': car_probability * driver_probability, 'Car': car_probability, 'Driver': driver_probability
        }
        full_identifier = '%s:%s' % (result.team().uuid(), result.driver().id())
        identifiers = {
            'All': full_identifier, 'Car': result.team().uuid(), 'Driver': result.driver().id()
        }
        for _ in range(self.get_oversample_rate(event.type(), event.id())):
            for mode in ['All', 'Car', 'Driver']:
                self._finish_pred_prob.get(mode).append(completed_probabilities.get(mode))
                self._finish_pred_true.get(mode).append(1)
                self.log_one_finish_probability(
                    event, mode, identifiers.get(mode), completed_probabilities.get(mode), 1)
                if self._debug_file is not None:
                    for km in range(math.floor(distance_success_km)):
                        print('Reliable\t%s\t%d\t1' % (mode, km), file=self._debug_file)
            # If they didn't finish the race, then either the car succeeded up until it crapped out and the driver
            # succeeded until they didn't, or vice versa.
            if result.dnf_category() == 'car':
                # The car didn't make it to the end.
                for mode in ['All', 'Car']:
                    self._finish_pred_prob.get(mode).append(full_probabilities.get(mode))
                    self._finish_pred_true.get(mode).append(0)
                    self.log_one_finish_probability(
                        event, mode, identifiers.get(mode), completed_probabilities.get(mode), 1)
                    if self._debug_file is not None:
                        print('Reliable\t%s\t%d\t0' % (mode, math.ceil(distance_success_km)), file=self._debug_file)
            elif result.dnf_category() == 'driver':
                # Blame the driver are the entire package for not making it to the end.
                for mode in ['All', 'Driver']:
                    self._finish_pred_prob.get(mode).append(full_probabilities.get(mode))
                    self._finish_pred_true.get(mode).append(0)
                    self.log_one_finish_probability(
                        event, mode, identifiers.get(mode), completed_probabilities.get(mode), 1)
                    if self._debug_file is not None:
                        print('Reliable\t%s\t%d\t0' % (mode, math.ceil(distance_success_km)), file=self._debug_file)

    def log_one_finish_probability(self, event, mode, identifier, probability, correct):
        if self._predict_file is None:
            return
        print('Finish\t%s\t%s\t%s\t%.6f\t%d' % (event.id(), mode, identifier, probability, correct),
              file=self._predict_file)

    def log_win_probabilities(self, event, predictions, driver_id, result):
        elo_before = predictions.drivers_before().get(result.driver())
        win_odds = predictions.win_probabilities().get(driver_id)
        won = 1 if result.end_position() == 1 else 0
        error = won - win_odds
        error = error ** 2
        for _ in range(self.get_oversample_rate(event.type(), event.id())):
            self._win_error_sum[event.type()] += error
            self._win_error_count[event.type()] += 1
            if self._predict_file is not None:
                print('WinOR\t%s\t%s\t%.1f\t%.6f\t%d' % (
                    event.id(), driver_id, elo_before, win_odds, won
                ),
                      file=self._predict_file)

    def log_podium_probabilities(self, event, predictions, driver_id, result):
        elo_before = predictions.drivers_before().get(result.driver())
        podium_odds = predictions.podium_probabilities().get(driver_id)
        podium = 1 if result.end_position() <= 3 else 0
        error = podium - podium_odds
        error = error ** 2
        for _ in range(self.get_oversample_rate(event.type(), event.id())):
            self._podium_pred_prob[event.type()].append(podium_odds)
            self._podium_pred_true[event.type()].append(podium)
            self._podium_error_sum[event.type()] += error
            self._podium_error_count[event.type()] += 1
            if self._predict_file is not None:
                print('PodiumOR\t%s\t%s\t%.1f\t%.6f\t%d' % (
                    event.id(), driver_id, elo_before, podium_odds, podium
                ),
                      file=self._predict_file)

    def log_summary_errors(self):
        for event_type in ['Q', 'R']:
            win_sum = self._win_error_sum[event_type]
            win_count = self._win_error_count[event_type]
            print('Error\tWin-%s\tTotal\t%.6f\t%7.1f\t%5d' % (event_type, win_sum / win_count, win_sum, win_count),
                  file=self._summary_file)
            podium_sum = self._podium_error_sum[event_type]
            podium_count = self._podium_error_count[event_type]
            print('Error\tPodium-%s\tTotal\t%.6f\t%7.1f\t%5d' % (
                event_type, podium_sum / podium_count, podium_sum, podium_count),
                  file=self._summary_file)
            precision, recall, thresholds = precision_recall_curve(self._podium_pred_true[event_type],
                                                                   self._podium_pred_prob[event_type])
            area_under_curve = auc(recall, precision)
            print('Error\tPodAUC-%s\tTotal\t%.6f\t--\t%5d' % (event_type, area_under_curve, podium_count),
                  file=self._summary_file)
        # For these we use ROC curves
        for mode in ['All', 'Car', 'Driver']:
            area_under_curve = roc_auc_score(self._finish_pred_true[mode], self._finish_pred_prob[mode])
            count = len(self._finish_pred_prob[mode])
            print('Error\tFin%sAUC\tTotal\t%.6f\t--\t%5d' % (mode, area_under_curve, count),
                  file=self._summary_file)
        for bucket, total in sorted(self._calibrate_num_total.items()):
            if bucket not in self._calibrate_num_correct:
                continue
            correct = self._calibrate_num_correct[bucket]
            pct_correct = 0
            if total:
                pct_correct = float(correct) / total
            print('Error\tCal\t%.3f\t%.3f\t%5d\t%5d' % (float(bucket) / 100, pct_correct, correct, total),
                  file=self._summary_file)
        for event_type, year_error_count in self._year_error_count.items():
            for year, count in year_error_count.items():
                if year not in self._year_error_sum[event_type]:
                    continue
                sse = self._year_error_sum[event_type][year]
                print('Error\tYear-%s\t%d\t%.6f\t%6.1f\t%5d' % (event_type, year, sse / count, sse, count),
                      file=self._summary_file)
        for event_type, decade_error_count in self._decade_error_count.items():
            sse_sum = 0
            count_sum = 0
            for decade, count in decade_error_count.items():
                if decade not in self._decade_error_sum[event_type]:
                    continue
                sse = self._decade_error_sum[event_type][decade]
                sse_sum += sse
                count_sum += count
                print('Error\tDecade-%s\t%d\t%.6f\t%7.1f\t%5d' % (event_type, decade, sse / count, sse, count),
                      file=self._summary_file)
            print('Error\tTotal-%s\tTotal\t%.6f\t%7.1f\t%5d' % (event_type, sse_sum / count_sum, sse_sum, count_sum),
                  file=self._summary_file)
        sse = self._total_error_sum
        count = self._total_error_count
        print('Error\tAllTotal\tTotal\t%.6f\t%7.1f\t%5d' % (sse / count, sse, count), file=self._summary_file)
