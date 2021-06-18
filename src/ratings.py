import math
import sys

import pandas as pd

from lifelines import CoxPHFitter


def event_to_year(event_id):
    return int(event_id[0:4])


def year_gap(event_id, last_event_id):
    last_year = event_to_year(last_event_id)
    this_year = event_to_year(event_id)
    return this_year - last_year


class KFactor(object):
    INVALID = -1
    _MAX_FACTOR = 24
    _MIN_FACTOR = 12

    def __init__(self, regress_rate=0.0):
        self._num_events = 0
        self._regress_rate = regress_rate

    def factor(self):
        if not self._num_events:
            return 0
        raw_factor = 800. / self._num_events
        if raw_factor > self._MAX_FACTOR:
            return self._MAX_FACTOR
        elif raw_factor < self._MIN_FACTOR:
            return self._MIN_FACTOR
        else:
            return raw_factor

    def num_events(self):
        return self._num_events

    def increment_events(self, count=1):
        self._num_events += count

    def regress(self, gap):
        self._num_events *= ((1 - self._regress_rate) ** gap)


class Reliability(object):
    DEFAULT_PROBABILITY = 0.7
    DEFAULT_KM_PER_RACE = 305.0

    def __init__(self, default_decay_rate=0.98, other=None, regress_numerator=None, regress_percent=0.02):
        self._template = None
        if other is None:
            self._km_success = 0
            self._km_failure = 0
            self._default_decay_rate = default_decay_rate
            self._regress_numerator = regress_numerator
            self._regress_percent = regress_percent
            if self._regress_numerator is None:
                self._regress_numerator = 2 * self.DEFAULT_KM_PER_RACE
        else:
            self._default_decay_rate = other.decay_rate()
            self._km_success = other.km_success()
            self._km_failure = other.km_failure()
            self._regress_numerator = other.regress_numerator()
            self._regress_percent = other.regress_percent()
            self.regress()
            self._template = other

    def regress(self):
        if not self._km_success:
            return
        ratio = self._regress_numerator / self._km_success
        if ratio < 1.0:
            self._km_success *= ratio
            self._km_failure *= ratio
        self.regress_to_template()

    def regress_to_template(self):
        if self._template is None:
            return
        template_km_success = self._template.km_success()
        template_km_failure = self._template.km_failure()
        template_total = template_km_failure + template_km_success
        if not template_total:
            return
        ratio = (self._km_success + self._km_failure) / template_km_success
        template_km_success *= (ratio * self._regress_percent)
        self._km_success *= (1 - self._regress_percent)
        self._km_success += template_km_success
        template_km_failure *= (ratio * self._regress_percent)
        self._km_failure *= (1 - self._regress_percent)
        self._km_failure += template_km_failure

    def start_update(self):
        self.decay()

    def update(self, km_success, km_failure):
        self._km_failure += km_failure
        self._km_success += km_success

    def commit_update(self):
        # A no-op for right now
        return

    def decay(self):
        decay_rate = self.decay_rate()
        self._km_failure *= decay_rate
        self._km_success *= decay_rate

    def probability_finishing(self, race_distance_km=DEFAULT_KM_PER_RACE):
        denominator = self._km_failure + self._km_success
        if not denominator:
            return self.DEFAULT_PROBABILITY
        per_km_success_rate = self._km_success / denominator
        return math.pow(per_km_success_rate, race_distance_km)

    def km_success(self):
        if self._km_success is not None:
            return self._km_success
        return 0

    def km_failure(self):
        if self._km_failure is not None:
            return self._km_failure
        return 0

    def decay_rate(self):
        return self._default_decay_rate

    def regress_numerator(self):
        return self._regress_numerator

    def regress_percent(self):
        return self._regress_percent

    def template(self):
        return self._template

    def set_template(self, other):
        self._template = other

    @staticmethod
    def opening_lap_car_model():
        return CarReliability._opening_lap_reliability

    @staticmethod
    def opening_lap_driver_model():
        return DriverReliability._opening_lap_reliability


class CarReliability(Reliability):
    _DEFAULT_SCALE = 100
    _MINIMUM_SCALE = 10
    _DEFAULT_DECAY = 0.95
    _opening_lap_reliability = CoxPHFitter()
    _opening_lap_data = list()
    _opening_lap_size = list()

    def __init__(self, default_decay_rate=0.98, regress_numerator=(64 * Reliability.DEFAULT_KM_PER_RACE),
                 regress_percent=0.03):
        super().__init__(default_decay_rate=default_decay_rate, regress_numerator=regress_numerator,
                         regress_percent=regress_percent)

    @staticmethod
    def opening_lap_data():
        return CarReliability._opening_lap_data

    @staticmethod
    def add_observation(base_probability, laps_completed, did_fail):
        failure_probability = 1 - base_probability
        if failure_probability < 1e-4:
            failure_probability = 1e-4
        data = [math.log10(failure_probability), laps_completed, did_fail]
        CarReliability._opening_lap_data.append(data)
        multiplier = 1
        if did_fail:
            multiplier = 4
        CarReliability._opening_lap_size.append(CarReliability._DEFAULT_SCALE * multiplier)

    @staticmethod
    def regress_opening_lap():
        idx_to_remove = list()
        num_samples_before = len(CarReliability._opening_lap_size)
        for idx in range(num_samples_before):
            new_size = CarReliability._opening_lap_size[idx] * CarReliability._DEFAULT_DECAY
            if math.floor(new_size) >= CarReliability._MINIMUM_SCALE:
                CarReliability._opening_lap_size[idx] = new_size
            else:
                idx_to_remove.append(idx)
        print('CarReliability SamplesBefore: %d SamplesAfter: %d' % (
            num_samples_before, num_samples_before - len(idx_to_remove)))
        for idx in sorted(idx_to_remove, reverse=True):
            del CarReliability._opening_lap_data[idx]
            del CarReliability._opening_lap_size[idx]

    @staticmethod
    def fit_opening_lap(event_id):
        df = pd.DataFrame(data=CarReliability._opening_lap_data,
                          columns=['base_probability', 'laps_completed', 'did_crash'])
        num_unique = len(pd.unique(df['base_probability']))
        if num_unique < 20:
            return
        df['weights'] = CarReliability._opening_lap_size
        print(df.info())
        CarReliability._opening_lap_reliability.fit(df, duration_col='laps_completed', event_col='did_crash',
                                                    weights_col='weights', robust=True)
        print('CR %s' % event_id)
        print('CR %s' % event_id, file=sys.stderr)
        print(CarReliability._opening_lap_reliability.baseline_hazard_)
        print(CarReliability._opening_lap_reliability.params_)
        CarReliability._opening_lap_reliability.print_summary()


class DriverReliability(Reliability):
    _DEFAULT_SCALE = 100
    _MINIMUM_SCALE = 10
    _DEFAULT_DECAY = 0.975
    _opening_lap_reliability = CoxPHFitter()
    _opening_lap_data = list()
    _opening_lap_size = list()

    def __init__(self, default_decay_rate=0.98, regress_numerator=(64 * Reliability.DEFAULT_KM_PER_RACE),
                 regress_percent=0.01):
        super().__init__(default_decay_rate=default_decay_rate, regress_numerator=regress_numerator,
                         regress_percent=regress_percent)

    @staticmethod
    def opening_lap_data():
        return DriverReliability._opening_lap_data

    @staticmethod
    def add_observation(base_probability, start_position, laps_completed, did_fail):
        failure_probability = 1 - base_probability
        if failure_probability < 1e-4:
            failure_probability = 1e-4
        data = [math.log10(failure_probability), start_position, laps_completed, did_fail]
        DriverReliability._opening_lap_data.append(data)
        multiplier = 1
        if did_fail:
            multiplier = 9
        DriverReliability._opening_lap_size.append(DriverReliability._DEFAULT_SCALE * multiplier)

    @staticmethod
    def regress_opening_lap():
        idx_to_remove = list()
        num_samples_before = len(DriverReliability._opening_lap_size)
        for idx in range(num_samples_before):
            new_size = DriverReliability._opening_lap_size[idx] * DriverReliability._DEFAULT_DECAY
            if math.floor(new_size) >= DriverReliability._MINIMUM_SCALE:
                DriverReliability._opening_lap_size[idx] = new_size
            else:
                idx_to_remove.append(idx)
        print('DriverReliability SamplesBefore: %d SamplesAfter: %d' % (
            num_samples_before, num_samples_before - len(idx_to_remove)))
        for idx in sorted(idx_to_remove, reverse=True):
            del DriverReliability._opening_lap_data[idx]
            del DriverReliability._opening_lap_size[idx]

    @staticmethod
    def fit_opening_lap(event_id):
        df = pd.DataFrame(data=DriverReliability._opening_lap_data,
                          columns=['base_probability', 'start_position', 'laps_completed', 'did_crash'])
        num_unique = len(pd.unique(df['base_probability']))
        if num_unique < 20:
            return
        df['weights'] = DriverReliability._opening_lap_size
        print(df.info())
        DriverReliability._opening_lap_reliability.fit(df, duration_col='laps_completed', event_col='did_crash',
                                                       weights_col='weights', robust=True)
        print('DR %s' % event_id)
        print('DR %s' % event_id, file=sys.stderr)
        print(DriverReliability._opening_lap_reliability.baseline_hazard_)
        print(DriverReliability._opening_lap_reliability.params_)
        DriverReliability._opening_lap_reliability.print_summary()


class EloRating(object):

    def __init__(self, init_rating, regress_rate=0.0, k_factor_regress_rate=0.0):
        self._default_rating = init_rating
        self._elo_rating = init_rating
        self._k_factor = KFactor(regress_rate=k_factor_regress_rate)
        self._regress_rate = regress_rate
        self._reliability = None
        self._non_alias_callers = list()
        self._alias_callers = list()
        self._deferred_complete = False
        self._temp_elo_rating = None
        self._last_event_id = None
        self._current_event_id = None
        self._commit_complete = False

    def start_update(self, event_id, caller_id, is_alias=False, base_reliability=None):
        self._commit_complete = False
        if is_alias:
            self._alias_callers.append(caller_id)
        else:
            self._non_alias_callers.append(caller_id)
        if self._current_event_id is None:
            self._current_event_id = event_id
        elif self._current_event_id != event_id:
            print('ERROR: conflicting event IDs')
        if self._reliability is None and base_reliability is not None:
            self._reliability = Reliability(other=base_reliability)

    def deferred_start_update(self):
        if self._deferred_complete:
            return
        if len(self._non_alias_callers) > 1:
            print('ERROR: Multiple canonical callers for %s: [%s]' % (
                self._current_event_id, ', '.join(self._non_alias_callers)
            ))
        if self._temp_elo_rating is not None:
            print('ERROR: start_update was called on this rating multiple times')
        self.regress(self._current_event_id)
        self._last_event_id = self._current_event_id
        self._temp_elo_rating = self._elo_rating
        # The reliability has to have its update happen after we regress in case we need to regress it, too.
        if self._reliability is not None:
            self._reliability.start_update()
        self._deferred_complete = True

    def update(self, delta):
        self.deferred_start_update()
        self._temp_elo_rating += delta

    def update_reliability(self, km_success, km_failure):
        self.deferred_start_update()
        self._reliability.update(km_success, km_failure)

    def probability_finishing(self, race_distance_km=Reliability.DEFAULT_KM_PER_RACE):
        if self._reliability is None:
            return Reliability.DEFAULT_PROBABILITY
        return self._reliability.probability_finishing(race_distance_km=race_distance_km)

    def commit_update(self):
        if self._commit_complete:
            return
        # We need this first conditional if a driver or team never has an update called,
        # and therefore never has the deferred update run.
        if self._temp_elo_rating is None:
            self.deferred_start_update()
        self._elo_rating = self._temp_elo_rating
        self._temp_elo_rating = None
        self._k_factor.increment_events()
        self._deferred_complete = False
        self._current_event_id = None
        self.reset_lists()
        self._commit_complete = True
        if self._reliability is not None:
            self._reliability.commit_update()

    def elo(self):
        return self._elo_rating

    def k_factor(self):
        return self._k_factor

    def reliability(self):
        return self._reliability

    def reset_lists(self):
        self._alias_callers.clear()
        self._non_alias_callers.clear()

    def regress(self, event_id):
        if self._last_event_id is None:
            return
        if self._last_event_id > event_id:
            print('ERROR: This event before last event: %s > %s' % (
                self._last_event_id, event_id))
        gap = year_gap(event_id, self._last_event_id)
        if not gap:
            return
        if self._reliability is not None:
            self._reliability.regress()
        self.regress_internal(gap)
        self._k_factor.regress(gap)

    def regress_internal(self, gap):
        regress_factor = (1 - self._regress_rate) ** gap
        #        print('#Years: %2d BaseRate: %.3f ThisRate: %.3f %s [%s] [%s]' % (
        #              this_year - last_year, regress_rate, regress_factor, self._current_event_id,
        #              ', '.join(self._non_alias_callers), ', '.join(self._alias_callers)
        #        ))
        self._elo_rating *= regress_factor
        self._elo_rating += ((1 - regress_factor) * self._default_rating)
