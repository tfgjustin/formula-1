import argparse
import datetime
import math
import numpy as np
import scipy.special as sc

from collections import defaultdict
from sklearn.metrics import auc, precision_recall_curve


def validate_factors(argument):
    parts = argument.split('_')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError('Invalid factor: %s' % argument)
    try:
        initial = int(parts[0])
    except ValueError as ve:
        raise argparse.ArgumentTypeError(
            'Factor %s has non-numeric initial value %s' % (argument, parts[0]))
    else:
        if initial < 0 or initial > 100:
            raise argparse.ArgumentTypeError(
                'Factor %s has invalid initial value %s' % (argument, parts[0]))

    try:
        year_step = int(parts[1])
    except ValueError as ve:
        raise argparse.ArgumentTypeError(
            'Factor %s has non-numeric year-step value %s' % (argument, parts[1]))
    else:
        if year_step < 0:
            raise argparse.ArgumentTypeError(
                'Factor %s has invalid year-step value %s' % (argument, parts[1]))
    try:
        step_value = int(parts[2])
    except ValueError as ve:
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


def select_reference_result(event, ratings, from_back):
    max_pfb = max(from_back.values())
    max_rating = max(
        [rating for driver_id, rating in ratings.items() if from_back[driver_id] == max_pfb]
    )
    reference_driver_ids = list([driver_id for driver_id, rating in ratings.items()
                                 if from_back[driver_id] == max_pfb and abs(rating - max_rating) < 1e-2])
    if not len(reference_driver_ids):
        print('ERROR: Found no reference driver IDs for %s' % event.id(), file=sys.stderr)
        return None
    reference_id = reference_driver_ids[0]
    reference_result = [result for result in event.results() if result.driver().id() == reference_id]
    if not reference_result:
        print('ERROR: Found no reference result for driver %s' % reference_id, file=sys.stderr)
        return None
    return reference_result[0]


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
        self._podium_error_sum = dict({'Q': 0.0, 'R': 0.0})
        self._podium_error_count = dict({'Q': 0, 'R': 0})
        self._podium_pred_prob = dict({'Q': list(), 'R': list()})
        self._podium_pred_true = dict({'Q': list(), 'R': list()})
        self._win_error_sum = dict({'Q': 0.0, 'R': 0.0})
        self._win_error_count = dict({'Q': 0, 'R': 0})
        self._calibrate_num_correct = defaultdict(int)
        self._calibrate_num_total = defaultdict(int)
        self.create_position_base_dict()
        self.create_team_share_dict()
        self._oversample_rates = dict(
            {'Q': dict({'195': 3, '196': 2}),
             'R': dict({'195': 7, '196': 6, '197': 3, '198': 3, '199': 3, '200': 2})
             })
        self._km_car_success = 0
        self._km_car_failure = 0

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
        print('RaceID\tDriverID\tPlaced\tNumDrivers\tDnfReason\tEloPre\tEloPost\tEloDiff\tEffectPre\tEffectPost',
              file=self._driver_rating_file)
        print('RaceID\tTeamUUID\tTeamID\tNumTeams\tEloPre\tEloPost\tEffectPre\tEffectPost', file=self._team_rating_file)
        for year in sorted(loader.seasons().keys()):
            self.run_one_year(year, loader.seasons()[year])
        self.print_errors()

    def run_one_year(self, year, season):
        """
        Run the model for one year.
        """
        events_dict = season.events()
        if self._logfile is not None:
            print('Running %d' % (year), file=self._logfile)
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
        drivers_before = dict()
        teams_before = dict()
        self.collect_ratings('Before', event, drivers_before, teams_before)
        k_factor_adjust = 1
        elo_denominator = self._args.elo_exponent_denominator_race
        if event.type() == 'Q':
            # If this is a qualifying session, adjust all K-factors by a constant multiplier so fewer points flow
            # between the drivers and teams. Also use a different denominator since the advantage is much more
            # pronounced in qualifying than in races (e.g., a 100 point Elo advantage in qualifying gives you a much
            # better chance of finishing near the front than a 100 point Elo advantage in a race).
            k_factor_adjust = self._args.qualifying_kfactor_multiplier
            elo_denominator = self._args.elo_exponent_denominator_qualifying
        # Predict the odds of each entrant either winning or finishing on the podium.
        win_probs = dict()
        podium_probs = dict()
        self.predict_winner(event, win_probs, podium_probs, k_factor_adjust, elo_denominator)
        event.start_updates()
        # Do the full pairwise comparison of each driver. In order to not calculate A vs B and B vs A (or A vs A) only
        # compare when the ID of A<B.
        for result_a in event.results():
            for result_b in event.results():
                if result_a.driver().id() >= result_b.driver().id():
                    continue
                self.compare_results(event, result_a, result_b, k_factor_adjust, elo_denominator)
        event.commit_updates()
        drivers_after = dict()
        teams_after = dict()
        self.collect_ratings('After', event, drivers_after, teams_after)
        self.print_legacy_info(event, drivers_before, drivers_after, teams_before, teams_after, win_probs, podium_probs)

    def collect_ratings(self, tag, event, drivers, teams):
        """
        Collect the per-driver and per-team ratings to dicts for easier access later one. Also optionally print out
        some debugging information.
        """
        teams.update({team: team.rating().rating() for team in event.teams()})
        drivers.update({driver: driver.rating().rating() for driver in event.drivers()})
        if self._debug_file is not None:
            print('    %s %s' % (tag, event.id()), file=self._debug_file)
            print('      Teams', file=self._debug_file)
            for team, rating in sorted(teams.items(), key=lambda item: item[1], reverse=True):
                print('        %7.2f %s' % (rating, team.id()), file=self._debug_file)
            print('      Drivers', file=self._debug_file)
            for driver, rating in sorted(drivers.items(), key=lambda item: item[1], reverse=True):
                print('        %7.2f %s' % (rating, driver.id()), file=self._debug_file)

    def predict_winner(self, event, win_probs, podium_probs, k_factor_adjust, elo_denominator):
        """
        Predict the probability of each entrant winning. Use the odds ratio.
        https://en.wikipedia.org/wiki/Odds_ratio
        """
        ratings = dict()
        from_back = dict()
        self.create_ratings(event, ratings, from_back)
        # Identify a reference entrant and compare each entrant to that person.
        reference_result = select_reference_result(event, ratings, from_back)
        if reference_result is None:
            return
        drivers = list()
        probabilities = np.ones([len(ratings)], dtype='float')
        idx = 0
        for result in event.results():
            drivers.append(result.driver().id())
            win_probability = self.compare_results(event, result, reference_result, k_factor_adjust, elo_denominator,
                                                   update_ratings=False, update_errors=False)
            if win_probability is None:
                continue
            probabilities[idx] = win_probability / (1 - win_probability)
            idx += 1
        probabilities /= np.sum(probabilities)
        for idx in range(0, len(ratings)):
            win_probs[drivers[idx]] = probabilities[idx]
        # Now that we have the odds of each one winning, predict the odds each one finishes in the top 3.
        self.predict_podium(event, win_probs, podium_probs)

    def create_ratings(self, event, ratings, from_back):
        max_laps = 0
        for result in event.results():
            rating, k_factor, pfb = self.create_merged_rating(event.season(), result)
            ratings[result.driver().id()] = rating
            from_back[result.driver().id()] = result.position_from_back()
            if result.laps() > max_laps:
                max_laps = result.laps()
        if event.type() == 'Q':
            return
        # Move this to its own function
        km_per_lap = 300. / max_laps
        km_car_success = 0
        km_car_failure = 0
        for result in event.results():
            if result.dnf_category() == 'driver':
                continue
            if result.laps() <= 1:
                continue
            km_car_success += km_per_lap * result.laps()
            if result.dnf_category() != '-':
                km_car_failure += 1
        self._km_car_failure *= 0.97
        self._km_car_success *= 0.97
        self._km_car_failure += km_car_failure
        self._km_car_success += km_car_success
        if self._summary_file is not None:
            num = self._km_car_success
            den = self._km_car_success + self._km_car_failure
            prob_success = num / den
            prob_failure = 1 - prob_success
            print('Reliable\t%s\t%.7f\t%.1f\t%.6f' % (
                event.id(), prob_success, 1 / prob_failure, math.pow(prob_success, 300)
            ),
                  file=self._summary_file)
            return

    def predict_podium(self, event, win_probs, podium_probs):
        """
        https://math.stackexchange.com/questions/625611/given-every-horses-chance-of-winning-a-race-what-is-the-probability-that-a-spe
        """
        second = defaultdict(float)
        third = defaultdict(float)
        for driver_id_a, win_prob_a in win_probs.items():
            for driver_id_b, win_prob_b in win_probs.items():
                if driver_id_a == driver_id_b:
                    continue
                prob_a_then_b = win_prob_a * (win_prob_b / (1 - win_prob_a))
                if self._debug_file is not None:
                    print('Part2nd\t%s\t%7.5f\t%7.5f\t%7.5f\t%7.5f\t%7.5f\t%s\t%s' % (
                        event.id(), win_prob_a, win_prob_b, prob_a_then_b, 1 - win_prob_b,
                        win_prob_a / (1 - win_prob_b), driver_id_a, driver_id_b),
                          file=self._debug_file)
                for driver_id_c, win_prob_c in win_probs.items():
                    if driver_id_c == driver_id_a or driver_id_c == driver_id_b:
                        continue
                    prob_a_then_b_then_c = prob_a_then_b * (win_prob_c / (1 - (win_prob_a + win_prob_b)))
                    if self._debug_file is not None:
                        print('Part3rd\t%s\t%8.6f\t%8.6f\t%8.6f\t%8.6f\t%8.6f\t%8.6f\t%s\t%s\t%s' % (
                            event.id(), win_prob_a, prob_a_then_b, win_prob_c,
                            1 - (win_prob_a + win_prob_b), win_prob_c / (1 - (win_prob_a + win_prob_b)),
                            prob_a_then_b_then_c, driver_id_a, driver_id_b, driver_id_c),
                              file=self._debug_file)
                    third[driver_id_c] += prob_a_then_b_then_c
                second[driver_id_b] += prob_a_then_b
        for driver_id, win_prob in win_probs.items():
            podium_probs[driver_id] = win_prob + second.get(driver_id, 0) + third.get(driver_id, 0)
            if self._debug_file is not None:
                print('AddPodium\t%s\t%7.5f\t%7.5f\t%7.5f\t%7.5f\t%s' % (
                    event.id(), podium_probs[driver_id], win_prob, second.get(driver_id, 0), third.get(driver_id, 0),
                    driver_id),
                      file=self._debug_file)

    def should_compare(self, rating_a, rating_b):
        return abs(rating_a - rating_b) <= self._args.elo_compare_window

    def create_merged_rating(self, season, result):
        car_factor = self._team_share_dict[season]
        rating = result.driver().rating().rating()
        rating *= (1 - car_factor)
        rating += (car_factor * result.team().rating().rating())
        k_factor = result.driver().rating().k_factor().factor()
        k_factor *= (1 - car_factor)
        k_factor += (car_factor * result.team().rating().k_factor().factor())
        if self._debug_file is not None:
            print('        R: %4d KF: %2d NR: %2d SP: %2d EP: %2d PFB: %2d DNFR: %6s D: %s T: %s' % (
                rating, k_factor, result.num_racers(), result.start_position(), result.end_position(),
                result.position_from_back(), result.dnf_category(), result.driver().id(), result.team().uuid()),
                  file=self._debug_file)
        return rating, k_factor, 1.0

    def start_position_advantage(self, season, from_back_a, from_back_b):
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

    def win_probability(self, r_a, r_b, denominator=None):
        """Standard logistic calculator, per
           https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
        """
        if denominator is None:
            denominator = self._args.elo_exponent_denominator_race
        q_a = 10 ** (r_a / denominator)
        q_b = 10 ** (r_b / denominator)
        return q_a / (q_a + q_b)

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

    def compare_results(self, event, result_a, result_b, k_factor_adjust, elo_denominator,
                        update_ratings=True, update_errors=True):
        rating_a, k_factor_a, pfb_a = self.create_merged_rating(event.season(), result_a)
        rating_b, k_factor_b, pfb_b = self.create_merged_rating(event.season(), result_b)
        if not k_factor_b or not k_factor_b:
            if self._debug_file is not None:
                print('      Skip: K', file=self._debug_file)
            return None
        k_factor = ((k_factor_a + k_factor_b) / 2) * k_factor_adjust
        rating_a += self.start_position_advantage(event.season(), pfb_a, pfb_b)
        win_prob_a = self.win_probability(rating_a, rating_b, elo_denominator)
        win_actual_a = 1 if result_a.end_position() < result_b.end_position() else 0
        use_drivers, use_team = allocate_results(result_a, result_b, self._debug_file)
        if use_drivers is None or use_team is None:
            if self._debug_file is not None:
                print('      Skip: Use', file=self._debug_file)
            return win_prob_a
        if update_errors:
            self.add_h2h_error(event, rating_a, rating_b, win_actual_a, win_prob_a)
            self.add_h2h_error(event, rating_b, rating_a, 1 - win_actual_a, 1 - win_prob_a)
        if not self.should_compare(rating_a, rating_b):
            if self._debug_file is not None:
                print('      Skip: SC', file=self._debug_file)
            return win_prob_a
        if not update_ratings:
            return win_prob_a
        delta_a = k_factor * (win_actual_a - win_prob_a)
        self.update_ratings(event.season(), result_a, use_drivers, use_team, delta_a)
        self.update_ratings(event.season(), result_b, use_drivers, use_team, -delta_a)
        return win_prob_a

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

    def print_errors(self):
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

    def print_legacy_info(self, event, drivers_before, drivers_after, teams_before, teams_after, win_probs, podium_probs):
        places = {result.driver(): result.end_position() for result in event.results()}
        dnfs = {result.driver(): result.dnf_category() for result in event.results()}
        num_drivers = len(drivers_after)
        for driver in sorted(drivers_after.keys(), key=lambda d: d.id()):
            if driver not in drivers_before or driver not in places or driver not in dnfs:
                continue
            before = drivers_before[driver]
            after = drivers_after[driver]
            placed = places[driver]
            dnf = dnfs[driver]
            before_effect = (before - self._args.driver_elo_initial) * (1- self._team_share_dict[event.season()])
            after_effect = (after - self._args.driver_elo_initial) * (1- self._team_share_dict[event.season()])
            print('S%s\t%s\t%d\t%d\t%s\t%.1f\t%.1f\t%6.1f\t%6.1f\t%6.1f' % (
                event.id(), driver.id(), placed, num_drivers, dnf, before, after, after - before,
                before_effect, after_effect),
                  file=self._driver_rating_file)
            if driver.id() in win_probs:
                win_odds = win_probs[driver.id()]
                won = 1 if placed == 1 else 0
                error = won - win_odds
                error = error ** 2
                for _ in range(self.get_oversample_rate(event.type(), event.id())):
                    self._win_error_sum[event.type()] += error
                    self._win_error_count[event.type()] += 1
                    if self._predict_file is not None:
                        print('WinOR\t%s\t%s\t%.1f\t%.4f\t%d' % (
                            event.id(), driver.id(), before, win_odds, won
                        ),
                              file=self._predict_file)
            if driver.id() in podium_probs:
                podium_odds = podium_probs[driver.id()]
                podium = 1 if placed <= 3 else 0
                error = podium - podium_odds
                error = error ** 2
                for _ in range(self.get_oversample_rate(event.type(), event.id())):
                    self._podium_pred_prob[event.type()].append(podium_odds)
                    self._podium_pred_true[event.type()].append(podium)
                    self._podium_error_sum[event.type()] += error
                    self._podium_error_count[event.type()] += 1
                    if self._predict_file is not None:
                        print('PodiumOR\t%s\t%s\t%.1f\t%.4f\t%d' % (
                            event.id(), driver.id(), before, podium_odds, podium
                        ),
                              file=self._predict_file)
        for team in sorted(teams_after.keys(), key=lambda t: t.id()):
            if team not in teams_before:
                continue
            before = teams_before[team]
            after = teams_after[team]
            before_effect = (before - self._args.team_elo_initial) * self._team_share_dict[event.season()]
            after_effect = (after - self._args.team_elo_initial) * self._team_share_dict[event.season()]
            print('S%s\t%s\t%s\t%d\t%.1f\t%.1f\t%6.1f\t%6.1f\t%6.1f' % (
                event.id(), team.uuid(), team.id(), len(teams_after), before, after, after - before,
                before_effect, after_effect
            ), file=self._team_rating_file)

