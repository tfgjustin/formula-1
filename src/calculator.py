import argparse
import datetime
import event as f1event
import functools
import math
import numpy as np

from fuzz import Fuzzer
from f1_logging import open_file_with_suffix
from predictions import EventPrediction, SimulatedEventPrediction
from ratings import CarReliability, DriverReliability, Reliability
from scipy.stats import skew
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


_ALL = 'All'
_CAR = 'Car'
_DRIVER = 'Driver'

_MODE_FULL = 'full'
_MODE_PARTIAL = 'partial'
_MODE_CLOSING = 'closing'


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
    # Always look at next year so we can predict one year ahead if necessary.
    next_year = datetime.datetime.now().year + 1
    num_seasons = next_year - 1950
    if year_step > 0:
        max_value = initial + (int(num_seasons / year_step) * step_value)
    else:
        max_value = initial
    if max_value >= 100:
        raise argparse.ArgumentTypeError(
            'Factor spec %s will result in a final factor of %d' % (argument, max_value))
    return argument


def was_performance_win(result_this, result_other, mode=_MODE_FULL):
    if mode != _MODE_PARTIAL:
        return result_this.dnf_category() == '-' and result_other.dnf_category() == '-'
    else:
        # Partial results
        return result_this.partial_position() is not None and result_other.partial_position() is not None


def assign_year_value_dict(spec, divisor, value_dict):
    parts = spec.split('_')
    initial = float(parts[0]) / divisor
    window = int(parts[1])
    step = float(parts[2]) / divisor
    # Always look at next year so we can predict one year ahead if necessary.
    next_year = datetime.datetime.now().year + 1
    count = 1
    value_dict[1950] = initial
    current_value = initial
    for year in range(1951, next_year + 1):
        if count >= window:
            current_value += step
            count = 1
        else:
            count += 1
        value_dict[year] = current_value


class Calculator(object):

    def __init__(self, args, base_filename):
        self._args = args
        if getattr(self._args, 'print_ratings', False):
            self._driver_rating_file = open_file_with_suffix(base_filename, '.driver_ratings')
            self._team_rating_file = open_file_with_suffix(base_filename, '.team_ratings')
            self._summary_file = open_file_with_suffix(base_filename, '.summary')
        else:
            self._driver_rating_file = None
            self._team_rating_file = None
            self._summary_file = None
        self._logfile = None
        if getattr(self._args, 'print_progress', False):
            self._logfile = open_file_with_suffix(base_filename, '.log')
        self._debug_file = None
        if getattr(self._args, 'print_debug', False):
            self._debug_file = open_file_with_suffix(base_filename, '.debug')
        self._predict_file = None
        if getattr(self._args, 'print_predictions', False):
            self._predict_file = open_file_with_suffix(base_filename, '.predict')
        self._simulation_log_file = None
        if getattr(self._args, 'print_future_simulations', False) or \
           getattr(self._args, 'print_simulations', False):
            self._simulation_log_file = open_file_with_suffix(base_filename, '.simulations')
        self._position_base_dict = dict()
        self.create_position_base_dict()
        self._team_share_dict = dict()
        self.create_team_share_dict()
        self._oversample_rates = dict(
            {f1event.QUALIFYING: dict({'195': 3, '196': 2}),
             f1event.RACE: dict({'195': 7, '196': 6, '197': 3, '198': 3, '199': 3, '200': 2})
             })
        self._base_car_reliability = CarReliability(
            default_decay_rate=self._args.team_reliability_decay,
            regress_percent=self._args.team_reliability_regress,
            regress_numerator=(self._args.team_reliability_lookback * Reliability.DEFAULT_KM_PER_RACE),
            wear_percent=self._args.wear_reliability_percent)
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
        self._fuzzer = Fuzzer(self._args, self._logfile)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        On exit of this class make sure we close all the opened files.
        """
        if self._driver_rating_file is not None:
            self._driver_rating_file.close()
        if self._team_rating_file is not None:
            self._team_rating_file.close()
        if self._summary_file is not None:
            self._summary_file.close()
        if self._logfile is not None:
            self._logfile.close()
        if self._debug_file is not None:
            self._debug_file.close()
        if self._predict_file is not None:
            self._predict_file.close()
        if self._simulation_log_file is not None:
            self._simulation_log_file.close()

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
        if event_type not in self._oversample_rates:
            return 1
        return self._oversample_rates[event_type].get(event_id[:3], 1)

    def run_all_ratings(self, loader):
        """
        Run the ratings for each year and print the output to the relevant files.
        """
        self.log_rating_headers()
        for year in sorted(loader.seasons().keys()):
            if year < self._args.start_year:
                continue
            self.run_one_year(year, loader.seasons()[year])
        self.log_summary_errors()

    def run_one_year(self, year, season):
        """
        Run the model for one year.
        """
        events = season.events()
        if self._logfile is not None:
            print('Running %d' % year, file=self._logfile)
        self._base_new_car_reliability.regress()
        for event_id in sorted(events.keys(), key=functools.cmp_to_key(f1event.compare_events)):
            event = events[event_id]
            if self.should_skip_event(event):
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
        elo_denominator, k_factor_adjust = self.get_elo_and_k_factor_parameters(event)
        predictions = EventPrediction(event, self._args, elo_denominator, k_factor_adjust,
                                      self._team_share_dict[event.season()], self._position_base_dict[event.season()],
                                      self._base_car_reliability, self._base_new_car_reliability,
                                      self._base_driver_reliability, self._debug_file,
                                      simulation_log_file=self._simulation_log_file)
        predictions.cache_ratings()
        # Starting the updates will also regress the start-of-year back to the mean
        predictions.start_updates()
        predictions.maybe_force_regress()
        # Predict the odds of each entrant either winning or finishing on the podium.
        predictions.predict_winner(getattr(self._args, 'print_predictions', False))
        # Do the full pairwise comparison of each driver. In order to not calculate A vs B and B vs A (or A vs A) only
        # compare when the ID of A<B.
        for result_a in event.results():
            for result_b in event.results():
                if result_a.driver().id() < result_b.driver().id():
                    self.compare_results(event, predictions, result_a, result_b)
        self.update_all_reliability(event)
        # When we commit updates we commit all state to the Driver and Team entities.
        predictions.commit_updates()
        # This will reset the weather, course, etc., reliability conditions.
        predictions.reset_condition_state()
        self.log_results(predictions)

    def simulate_future_events(self, loader, fuzz=None, sim_log_idx_offset=0, output_buffer=None):
        last_seen_event_id = sorted(loader.events().keys(), key=functools.cmp_to_key(f1event.compare_events))[-1]
        last_seen_event = loader.events()[last_seen_event_id]
        for year in sorted(loader.future_seasons().keys()):
            self.simulate_one_future_year(last_seen_event, year, loader, fuzz=fuzz,
                                          sim_log_idx_offset=sim_log_idx_offset, output_buffer=output_buffer)

    def simulate_one_future_year(self, last_seen_event, year, loader, fuzz=None, sim_log_idx_offset=0,
                                 output_buffer=None):
        events = loader.future_seasons()[year].events()
        drivers = loader.future_drivers()
        teams = loader.future_teams()
        if self._logfile is not None:
            print('Simulating future year %d' % year, file=self._logfile)
        if fuzz is None:
            self._fuzzer.generate_all_fuzz(year, events, drivers, teams)
        else:
            self._fuzzer.set_fuzz(fuzz)
        # List of outcomes, with each item specifying the ordering of finishers by driver ID in a previous event. This
        # lets us carryover starting positions from qualifying to sprint races to actual races.
        carryover_starting_positions = list()
        if last_seen_event.season() != year:
            # If the last seen real event was in a previous year, regress this back to the mean.
            self._base_new_car_reliability.regress()
        else:
            # The last seen event was in this year. See if it was a race (no need to carry anything over) or something
            # else (we do need to carryover).
            if last_seen_event.type() not in [f1event.SPRINT_RACE, f1event.RACE] and last_seen_event.has_results():
                # We do need to carryover the results.
                self.create_carryover_from_event(last_seen_event, carryover_starting_positions)
        for event_id in sorted(events.keys(), key=functools.cmp_to_key(f1event.compare_events)):
            event = events[event_id]
            if self.should_skip_event(event):
                if self._logfile is not None:
                    print('Skipping %s (%s)' % (event.id(), event.name()), file=self._logfile)
                continue
            self.simulate_one_future_event(event, carryover_starting_positions, loader,
                                           sim_log_idx_offset=sim_log_idx_offset, output_buffer=output_buffer)

    def simulate_one_future_event(self, event, carryover_starting_positions, loader, sim_log_idx_offset=0,
                                  output_buffer=None):
        if event.type() in [f1event.QUALIFYING, f1event.SPRINT_SHOOTOUT]:
            carryover_starting_positions.clear()
        if self._logfile is not None:
            print('  Future Event %s' % (event.id()), file=self._logfile)
        is_wet = self.create_is_wet_array(event)
        elo_denominator, k_factor_adjust = self.get_elo_and_k_factor_parameters_array(event, is_wet)
        predictions = SimulatedEventPrediction(
            event, self._args, elo_denominator, k_factor_adjust, self._team_share_dict[event.season()],
            self._position_base_dict[event.season()], self._base_car_reliability, self._base_new_car_reliability,
            self._base_driver_reliability, self._debug_file, is_wet, simulation_log_file=self._simulation_log_file,
            starting_positions=carryover_starting_positions)
        # Starting the updates will also regress the start-of-year back to the mean
        # The 'common' part will not apply the weather modifier, though.
        predictions.start_updates_common()
        predictions.maybe_force_regress()
        predictions.commit_updates()
        # Only simulate the results, don't actually update the Elo and reliability ratings.
        grid_penalties = loader.grid_penalties(event_id=event.id())
        predictions.only_simulate_outcomes(self._fuzzer.all_fuzz(), grid_penalties=grid_penalties,
                                           sim_log_idx_offset=sim_log_idx_offset, output_buffer=output_buffer)
        # Wait until after we've simulated the outcomes to remove the weather, street, etc., reliability conditions.
        predictions.reset_condition_state()

    def create_is_wet_array(self, event):
        if event.weather() != 'wet':
            return np.repeat(False, self._args.num_iterations)
        elif abs(event.weather_probability() - 1) < 1e-4:
            return np.repeat(True, self._args.num_iterations)
        wet_probabilities = np.random.uniform(0, 1, self._args.num_iterations)
        return wet_probabilities < event.weather_probability()

    @staticmethod
    def should_skip_event(event):
        # Skip all the Indianapolis races since they were largely disjoint
        # from the rest of Formula 1.
        return 'Indianapolis' in event.name() and event.season() <= 1960

    def get_elo_and_k_factor_parameters(self, event):
        return self.get_elo_and_k_factor_parameters_inner(event, event.weather())

    def get_elo_and_k_factor_parameters_array(self, event, is_wet_array):
        # If it's either going to be dry or it's guaranteed to be wet, just return all the samples the same.
        if event.weather() != 'wet' or abs(event.weather_probability() - 1) < 1e-4:
            elo, k_factor = self.get_elo_and_k_factor_parameters_inner(event, event.weather())
            return np.repeat(elo, len(is_wet_array)), np.repeat(k_factor, len(is_wet_array))
        # If we got here, then there are mixed conditions. Let's get both dry and wet ratings.
        dry_elo, dry_k_factor = self.get_elo_and_k_factor_parameters_inner(event, 'dry')
        wet_elo, wet_k_factor = self.get_elo_and_k_factor_parameters_inner(event, 'wet')
        elo = np.where(is_wet_array, wet_elo, dry_elo)
        k_factor = np.where(is_wet_array, wet_k_factor, dry_k_factor)
        return elo, k_factor

    def get_elo_and_k_factor_parameters_inner(self, event, weather):
        if event.type() in [f1event.QUALIFYING, f1event.SPRINT_SHOOTOUT]:
            # If this is a qualifying session, adjust all K-factors by a constant multiplier so fewer points flow
            # between the drivers and teams. Also use a different denominator since the advantage is much more
            # pronounced in qualifying than in races (e.g., a 100 point Elo advantage in qualifying gives you a much
            # better chance of finishing near the front than a 100 point Elo advantage in a race).
            k_factor_adjust = self._args.qualifying_kfactor_multiplier
            elo_denominator = self._args.elo_exponent_denominator_qualifying
        elif event.type() in [f1event.SPRINT_RACE, f1event.SPRINT_QUALIFYING]:
            # Sprint qualifying, new for 2021. Use the K-Factor adjustment and the average of the race and qualifying
            # Elo denominator.
            k_factor_adjust = self._args.qualifying_kfactor_multiplier
            elo_denominator = (self._args.elo_exponent_denominator_race +
                               self._args.elo_exponent_denominator_qualifying) / 2
        else:
            k_factor_adjust = self.race_distance_multiplier(event)
            elo_denominator = self._args.elo_exponent_denominator_race
        if weather == 'wet':
            if event.type() in [f1event.QUALIFYING, f1event.SPRINT_SHOOTOUT]:
                elo_denominator *= self._args.wet_multiplier_elo_denominator_qualifying
            else:
                elo_denominator *= self._args.wet_multiplier_elo_denominator_race
            k_factor_adjust *= self._args.wet_multiplier_k_factor
        return elo_denominator, k_factor_adjust

    @staticmethod
    def create_carryover_from_event(event, carryover_starting_positions):
        carryover_starting_positions.append(
            '|'.join([r.driver().id() for r in sorted(event.results(), key=lambda r: r.end_position())])
        )

    def update_all_reliability(self, event):
        if event.type() in [f1event.QUALIFYING, f1event.SPRINT_SHOOTOUT]:
            return
        driver_condition_multiplier_km = 1
        car_condition_multiplier_km = 1
        # TODO: Fix up these hard-coded values
        if 'Monaco' in event.name():
            driver_condition_multiplier_km = 0.99982
        if event.stage() == 1:
            car_condition_multiplier_km *= 0.99983
        if event.weather() == 'wet':
            driver_condition_multiplier_km *= self._args.reliability_km_multiplier_wet
        if event.is_street_course():
            driver_condition_multiplier_km *= self._args.reliability_km_multiplier_street
        # For races and sprint qualifying, keep going.
        driver_crash_laps = [
            result.laps() for result in event.results() if result.dnf_category() == 'driver' and result.laps() >= 1
        ]
        crash_laps = {lap: driver_crash_laps.count(lap) for lap in driver_crash_laps}
        for result in event.results():
            if result.laps() < 1:
                continue
            km_success = event.lap_distance_km() * result.laps()
            km_car_failure = 0
            km_driver_failure = 0
            if result.dnf_category() == 'car':
                km_car_failure = self._args.team_reliability_failure_constant
                km_car_failure *= (car_condition_multiplier_km ** km_success)
            elif result.dnf_category() == 'driver':
                km_driver_failure = self._args.driver_reliability_failure_constant / crash_laps[result.laps()]
                km_driver_failure *= (driver_condition_multiplier_km ** km_success)
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
        entrant_a = result_a.entrant()
        entrant_b = result_b.entrant()
        # First do the overall win probabilities
        win_actual_a = 1 if result_a.end_position() < result_b.end_position() else 0
        win_prob_a = predictions.get_win_probability(entrant_a, entrant_b)
        if win_prob_a is None:
            if self._debug_file is not None:
                print('      Skip: Win Prob is None', file=self._debug_file)
            return
        if result_a.laps() > 0 and result_b.laps() > 0:
            self.add_full_h2h_error(event, win_actual_a, win_prob_a)
            self.add_full_h2h_error(event, 1 - win_actual_a, 1 - win_prob_a)
        # Now do the performance (Elo-based) probabilities
        self.update_elo_ratings(event, predictions, entrant_a, entrant_b, result_a, result_b, win_actual_a,
                                mode=_MODE_PARTIAL)
        self.update_elo_ratings(event, predictions, entrant_a, entrant_b, result_a, result_b, win_actual_a,
                                mode=_MODE_CLOSING)
        self.update_elo_ratings(event, predictions, entrant_a, entrant_b, result_a, result_b, win_actual_a,
                                mode=_MODE_FULL)

    def update_elo_ratings(self, event, predictions, entrant_a, entrant_b, result_a, result_b, win_actual_a,
                           mode=_MODE_FULL):
        elo_win_prob_a, rating_a, rating_b = predictions.get_elo_win_probability(entrant_a, entrant_b)
        if elo_win_prob_a is None:
            if self._debug_file is not None:
                print('      Skip: Elo Prob is None', file=self._debug_file)
            return
        if not was_performance_win(result_a, result_b, mode=mode):
            if self._debug_file is not None:
                print('      Skip: At least one entrant DNF\'ed', file=self._debug_file)
            return
        if mode == _MODE_FULL:
            if elo_win_prob_a >= 0.5:
                self.add_elo_h2h_error(event, win_actual_a, elo_win_prob_a)
            else:
                self.add_elo_h2h_error(event, 1 - win_actual_a, 1 - elo_win_prob_a)
            return
        car_delta, driver_delta = predictions.get_elo_deltas(entrant_a, entrant_b, mode=mode)
        if car_delta is None or driver_delta is None:
            if self._debug_file is not None:
                print('      Skip: Use', file=self._debug_file)
            return
        if not self.should_compare(rating_a, rating_b, predictions.elo_denominator()):
            if self._debug_file is not None:
                print('      Skip: SC', file=self._debug_file)
            return
        self.update_ratings(result_a, car_delta, driver_delta)
        self.update_ratings(result_b, -car_delta, -driver_delta)

    def race_distance_multiplier(self, event):
        lap_distance_km = event.lap_distance_km()
        max_laps = max([result.laps() for result in event.results()], default=event.num_laps())
        if self._debug_file is not None:
            print('LAPS %s: Projected %3d Actual %3d' % (event.id(), event.num_laps(), max_laps), file=self._debug_file)
        if max_laps == event.num_laps():
            return 1.0
        actual_distance = max_laps * lap_distance_km
        if self._debug_file is not None:
            print('  Event %s was scheduled for %.1f km but only ran %.1f km' % (
                event.id(), event.total_distance_km(), actual_distance
                ), file=self._debug_file)
        return actual_distance / 300

    def should_compare(self, rating_a, rating_b, elo_denominator):
        return (abs(rating_a - rating_b) / elo_denominator) <= self._args.elo_compare_window

    def add_full_h2h_error(self, event, actual, prob):
        # Include races and sprint qualifying but not qualifying or sprint shootouts.
        if event.type() in [f1event.QUALIFYING, f1event.SPRINT_SHOOTOUT]:
            return
        # Only log the odds of the favorite so things like Skew make sense
        if prob < 0.5:
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

    def log_rating_headers(self):
        if self._driver_rating_file is not None:
            print(('RaceID\tDriverID\tPlaced\tNumDrivers\tDnfReason\tEloPre\tEloPost\tEloDiff\tEffectPre\tEffectPost'
                   + '\tKFEventsPre\tKFEventsPost\tKmSuccessPre\tKmSuccessPost\tKmFailurePre\tKmFailurePost'
                   + '\tWearSuccessPre\tWearSuccessPost\tWearFailurePre\tWearFailurePost\tSuccessPre\tSuccessPost'
                   ),
                  file=self._driver_rating_file)
        if self._team_rating_file is not None:
            print(('RaceID\tTeamUUID\tTeamID\tNumTeams\tEloPre\tEloPost\tEloDiff\tEffectPre\tEffectPost'
                   + '\tKFEventsPre\tKFEventsPost\tKmSuccessPre\tKmSuccessPost\tKmFailurePre\tKmFailurePost'
                   + '\tWearSuccessPre\tWearSuccessPost\tWearFailurePre\tWearFailurePost\tSuccessPre\tSuccessPost'
                   ),
                  file=self._team_rating_file)

    def log_results(self, predictions):
        event = predictions.event()
        for team in sorted(event.teams(), key=lambda t: t.id()):
            self.log_team_results(event, predictions, team)
        self.log_average_team(event, predictions, self._base_car_reliability, 'TeamBase')
        self.log_average_team(event, predictions, self._base_new_car_reliability, 'TeamNew')
        driver_results = {result.driver().id(): result for result in event.results()}
        num_drivers = len(driver_results)
        # We use the driver ID as the key so the output in the logs will be sorted.
        for driver_id, result in sorted(driver_results.items()):
            self.log_driver_results(event, predictions, num_drivers, result)
            self.log_finish_probabilities(event, predictions, driver_id, result)
            if getattr(self._args, 'print_predictions', False):
                self.log_win_probabilities(event, predictions, driver_id, result)
                self.log_podium_probabilities(event, predictions, driver_id, result)

    def log_team_results(self, event, predictions, team):
        if self._team_rating_file is None:
            return
        rating_before = predictions.team_before(team.id()).rating()
        rating_after = team.rating()
        # We use this instead of after-before because there could've been regression to the mean throwing off the delta.
        elo_diff = rating_after.elo_delta()
        before_effect = (rating_before.elo() - self._args.team_elo_initial) * self._team_share_dict[event.season()]
        after_effect = (rating_after.elo() - self._args.team_elo_initial) * self._team_share_dict[event.season()]
        reliability_before = rating_before.reliability()
        if reliability_before is None:
            reliability_before = Reliability()
        reliability_after = rating_after.reliability()
        if reliability_after is None:
            reliability_after = Reliability()
        print(('S%s\t%s\t%s\t%d\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%5.1f\t%5.1f\t%8.2f\t%8.2f'
               + '\t%8.2f\t%8.2f\t%8.4f\t%8.4f\t%8.4f\t%8.4f\t%.6f\t%.6f') % (
            event.id(), team.uuid(), team.id(), len(event.teams()), rating_before.elo(), rating_after.elo(), elo_diff,
            before_effect, after_effect, rating_before.k_factor().num_events(), rating_after.k_factor().num_events(),
            reliability_before.km_success(), reliability_after.km_success(),
            reliability_before.km_failure(), reliability_after.km_failure(),
            reliability_before.wear_success(), reliability_after.wear_success(),
            reliability_before.wear_failure(), reliability_after.wear_failure(),
            rating_before.probability_finishing(), rating_after.probability_finishing()
        ),
              file=self._team_rating_file)

    def log_average_team(self, event, predictions, base_reliability, base_name):
        if self._team_rating_file is None:
            return
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
               + '\t%8.2f\t%8.2f\t%8.4f\t%8.4f\t%8.4f\t%8.4f\t%.6f\t%.6f') % (
            event.id(), base_name, base_name, len(matching_teams), rating_before, rating_after, elo_diff,
            before_effect, after_effect, 0, 0,
            0, base_reliability.km_success(),
            0, base_reliability.km_failure(),
            0, base_reliability.wear_success(),
            0, base_reliability.wear_failure(),
            0, base_reliability.probability_finishing()
        ),
              file=self._team_rating_file)

    def log_driver_results(self, event, predictions, num_drivers, result):
        if self._driver_rating_file is None:
            return
        driver = result.driver()
        placed = result.end_position()
        dnf = result.dnf_category()
        rating_before = predictions.driver_before(result.driver().id()).rating()
        rating_after = result.driver().rating()
        # We use this instead of after-before because there could've been regression to the mean throwing off the delta.
        elo_diff = rating_after.elo_delta()
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
               + '\t%8.2f\t%8.2f\t%8.4f\t%8.4f\t%8.4f\t%8.4f\t%.6f\t%.6f') % (
            event.id(), driver.id(), placed, num_drivers, dnf, rating_before.elo(), rating_after.elo(), elo_diff,
            before_effect, after_effect, rating_before.k_factor().num_events(), rating_after.k_factor().num_events(),
            reliability_before.km_success(), reliability_after.km_success(),
            reliability_before.km_failure(), reliability_after.km_failure(),
            reliability_before.wear_success(), reliability_after.wear_success(),
            reliability_before.wear_failure(), reliability_after.wear_failure(),
            rating_before.probability_finishing(), rating_after.probability_finishing()
        ),
              file=self._driver_rating_file)

    def log_finish_probabilities(self, event, predictions, driver_id, result):
        # Log this for sprint qualifying and races but not regular qualifying or sprint shootouts
        if event.type() in [f1event.QUALIFYING, f1event.SPRINT_SHOOTOUT]:
            return
        # Identifiers for the entrant, team (car), and driver
        identifiers = {
            _ALL: result.entrant().id(), _CAR: result.team().uuid(), _DRIVER: result.driver().id()
        }
        # The probabilities that the average entrant will finish and that this entrant will finish
        # NOTE: The naive full probability from the predictions does NOT take into account starting position.
        probability_full_naive = predictions.naive_finish_probabilities()
        probability_full_naive_all = probability_full_naive[_ALL] * result.entrant().probability_survive_opening()
        probability_full_entrant = predictions.finish_probabilities().get(driver_id)
        # Regardless of whether they finished the race or not this is the probability that they got as far as they
        # did in the race.
        distance_success_km = result.laps() * event.lap_distance_km()
        probability_completed_entrant_car = result.team().rating().probability_finishing(
            race_distance_km=distance_success_km)
        probability_completed_entrant_driver = result.driver().rating().probability_finishing(
            race_distance_km=distance_success_km)
        # NOTE: We don't just multiply these two together because there's the opening lap survival odds.
        probability_completed_entrant_all = result.entrant().probability_complete_n_laps(event.num_laps())
        # The naive (baseline) odds that they got as far as they did
        team_num_events = result.team().rating().k_factor().num_events()
        if team_num_events >= self._args.team_reliability_new_events:
            probability_completed_naive_car = self._base_car_reliability.probability_finishing(
                race_distance_km=distance_success_km)
        else:
            probability_completed_naive_car = self._base_new_car_reliability.probability_finishing(
                race_distance_km=distance_success_km)
        probability_completed_naive_driver = self._base_driver_reliability.probability_finishing(
            race_distance_km=distance_success_km)
        probability_completed_naive_all = probability_completed_naive_driver * probability_completed_naive_car
        # The only entrant-specific probability we use is for the starting grid
        probability_completed_naive_all *= result.entrant().probability_survive_opening()
        completed_error_arrays = {
            _ALL: [event.id(), probability_completed_naive_all, probability_completed_entrant_all, 1],
            _CAR: [event.id(), probability_completed_naive_car, probability_completed_entrant_car, 1],
            _DRIVER: [event.id(), probability_completed_naive_driver, probability_completed_entrant_driver, 1]
        }
        # These are in case the entrant fails to get the full distance
        failed_error_arrays = {
            _ALL: [event.id(), probability_full_naive_all, probability_full_entrant[_ALL], 0],
            _CAR: [event.id(), probability_full_naive[_CAR], probability_full_entrant[_CAR], 0],
            _DRIVER: [event.id(), probability_full_naive[_DRIVER], probability_full_entrant[_DRIVER], 0]
        }
        for _ in range(self.get_oversample_rate(event.type(), event.id())):
            for mode in [_ALL, _CAR, _DRIVER]:
                self._finish_odds_log[mode].append(completed_error_arrays[mode])
                self.log_one_finish_probability(
                    event, mode, identifiers.get(mode), completed_error_arrays[mode][-2], 1)
            # If they didn't finish the race, then either the car succeeded up until it crapped out and the driver
            # succeeded until they didn't, or vice versa.
            failed_modes = []
            if result.laps() <= 1 and result.dnf_category() != '-':
                # If they crashed out or the car failed on the first or second lap, only tag that
                # against the overall reliability. The car and driver reliability should only take
                # into account post-infancy failure.
                failed_modes = [_ALL]
            elif result.dnf_category() == 'car':
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
        if self._predict_file is None:
            return
        win_odds = predictions.win_probabilities().get(driver_id)
        if win_odds is None:
            return
        won = 1 if result.end_position() == 1 else 0
        naive_odds = 1.0 / len(event.drivers())
        error_array = [event.id(), naive_odds, win_odds, won]
        for _ in range(self.get_oversample_rate(event.type(), event.id())):
            self._win_odds_log.append(error_array)
            print('WinOR\t%s\t%s\t%.6f\t%.6f\t%d' % (
                event.id(), driver_id, naive_odds, win_odds, won
            ),
                  file=self._predict_file)

    def log_podium_probabilities(self, event, predictions, driver_id, result):
        if self._predict_file is None:
            return
        podium_odds = predictions.podium_probabilities().get(driver_id)
        if podium_odds is None:
            return
        podium = 1 if result.end_position() <= 3 else 0
        naive_odds = 1.0 / len(event.drivers())
        error_array = [event.id(), naive_odds, podium_odds, podium]
        for _ in range(self.get_oversample_rate(event.type(), event.id())):
            self._podium_odds_log.append(error_array)
            print('PodiumOR\t%s\t%s\t%.6f\t%.6f\t%d' % (
                event.id(), driver_id, naive_odds, podium_odds, podium
            ),
                  file=self._predict_file)

    def log_summary_errors(self):
        start_decade = int(math.floor(self._args.start_year / 10))
        for decade in range(start_decade, 203):
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
        for event_type in [f1event.QUALIFYING, f1event.RACE]:
            matching_errors = self.get_matching_errors(self._elo_h2h_log, decade, event_type)
            self.log_one_error_log('EloH2H', matching_errors, decade, event_type)
            self.log_one_pr_auc('EloH2H', matching_errors, decade, event_type)

    def log_win_summary(self, decade=''):
        if not getattr(self._args, 'print_predictions', False):
            return
        for event_type in [f1event.QUALIFYING, f1event.RACE]:
            matching_errors = self.get_matching_errors(self._win_odds_log, decade, event_type)
            self.log_one_error_log('Win', matching_errors, decade, event_type)
            self.log_one_pr_auc('Win', matching_errors, decade, event_type)

    def log_podium_summary(self, decade=''):
        if not getattr(self._args, 'print_predictions', False):
            return
        for event_type in [f1event.QUALIFYING, f1event.RACE]:
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
        if self._summary_file is None:
            return
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
        if self._summary_file is None:
            return
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
        if self._summary_file is None:
            return
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
