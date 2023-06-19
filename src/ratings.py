import event as f1event
import math
import sys

from collections import deque


def triangle(number):
    return (number * (number + 1)) / 2


def event_to_year(event_id):
    return int(event_id[0:4])


def year_gap(event_id, last_event_id):
    last_year = event_to_year(last_event_id)
    this_year = event_to_year(event_id)
    return this_year - last_year


class KFactor(object):
    INVALID = -1
    _MAX_FACTOR = 28
    _MIN_FACTOR = 12

    def __init__(self, regress_rate=0.0, num_events=0):
        self._num_events = num_events
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

    def __init__(self, default_decay_rate=0.98, other=None, regress_numerator=None, regress_percent=0.02,
                 km_success=0, km_failure=0, wear_success=0, wear_failure=0, wear_percent=0):
        self._template = None
        if other is None:
            self._km_success = km_success
            self._km_failure = km_failure
            self._wear_success = wear_success
            self._wear_failure = wear_failure
            self._wear_percent = wear_percent
            self._default_decay_rate = default_decay_rate
            self._regress_numerator = regress_numerator
            self._regress_percent = regress_percent
            if self._regress_numerator is None:
                self._regress_numerator = 2 * self.DEFAULT_KM_PER_RACE
        else:
            self._default_decay_rate = other.decay_rate()
            self._km_success = other.km_success()
            self._km_failure = other.km_failure()
            self._wear_success = other.wear_success()
            self._wear_failure = other.wear_failure()
            self._wear_percent = other.wear_percent()
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
            self._wear_success *= ratio
            self._wear_failure *= ratio
        self.regress_to_template()

    def regress_to_template(self):
        if self._template is None:
            return
        if not self._template.km_failure() and not self._template.km_success():
            return
        ratio = (self._km_success + self._km_failure) / self._template.km_success()
        for km_or_wear in ['km', 'wear']:
            for outcome in ['success', 'failure']:
                self.internal_regress_one_variable(ratio, km_or_wear, outcome)

    def internal_regress_one_variable(self, ratio, km_or_wear, outcome):
        fn_name = '%s_%s' % (km_or_wear, outcome)
        attr_name = '_%s' % fn_name
        template_fn = getattr(self._template, fn_name)
        if template_fn is None:
            return
        self_value = getattr(self, attr_name)
        if self_value is None:
            return
        self_value *= (1 - self._regress_percent)
        self_value += (template_fn() * ratio * self._regress_percent)
        setattr(self, attr_name, self_value)

    def start_update(self):
        self.decay()

    def update(self, km_success, km_failure, wear_success=None, wear_failure=None):
        self._km_failure += km_failure
        self._km_success += km_success
        if wear_success is None:
            self._wear_success += triangle(km_success)
        else:
            self._wear_success += wear_success
        if wear_failure is None:
            self._wear_failure += km_failure
        else:
            self._wear_failure += wear_failure

    def commit_update(self):
        # A no-op for right now
        return

    def decay(self):
        decay_rate = self.decay_rate()
        self._km_failure *= decay_rate
        self._km_success *= decay_rate
        self._wear_failure *= decay_rate
        self._wear_success *= decay_rate

    def probability_finishing(self, start_km=0, race_distance_km=DEFAULT_KM_PER_RACE):
        km_denominator = self._km_failure + self._km_success
        if not km_denominator:
            return self.DEFAULT_PROBABILITY
        wear_denominator = self._wear_failure + self._wear_success
        if not wear_denominator:
            print('%8.2f\t%8.2f' % (self._km_failure, self._km_success))
            sys.exit(1)
        per_km_failure_rate = self._km_failure / km_denominator
        per_km_wear_rate = self._wear_failure / wear_denominator
        success_probability = 1
        for km in range(math.floor(start_km), math.floor(race_distance_km)):
            success_probability *= self.probability_success_at(km, per_km_failure_rate, per_km_wear_rate)
        return success_probability

    def probability_success_at(self, km, base_fail_per_km, wear_per_km):
        return 1 - (base_fail_per_km + (self._wear_percent * wear_per_km * km))

    def wear_success(self):
        if self._wear_success is not None:
            return self._wear_success
        return 0

    def wear_failure(self):
        if self._wear_failure is not None:
            return self._wear_failure
        return 0

    def wear_percent(self):
        if self._wear_percent is not None:
            return self._wear_percent
        return 0

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

    def set_decay_rate(self, decay_rate):
        self._default_decay_rate = decay_rate

    def set_template(self, other):
        self._template = other


class CarReliability(Reliability):

    def __init__(self, default_decay_rate=0.995, regress_numerator=(48 * Reliability.DEFAULT_KM_PER_RACE),
                 regress_percent=0.03, km_success=0, km_failure=0, wear_success=0, wear_failure=0, wear_percent=0.638):
        super().__init__(default_decay_rate=default_decay_rate, regress_numerator=regress_numerator,
                         regress_percent=regress_percent, km_success=km_success, km_failure=km_failure,
                         wear_success=wear_success, wear_failure=wear_failure, wear_percent=wear_percent)


class DriverReliability(Reliability):

    def __init__(self, default_decay_rate=0.98, regress_numerator=(64 * Reliability.DEFAULT_KM_PER_RACE),
                 regress_percent=0.01, km_success=0, km_failure=0, wear_success=0, wear_failure=0):
        super().__init__(default_decay_rate=default_decay_rate, regress_numerator=regress_numerator,
                         regress_percent=regress_percent, km_success=km_success, km_failure=km_failure,
                         wear_success=wear_success, wear_failure=wear_failure)


class EloRating(object):

    def __init__(self, init_rating, regress_rate=0.0, k_factor_regress_rate=0.0, k_factor=None,
                 reliability=None, last_event_id=None, lookback_length=10):
        self._default_rating = init_rating
        self._elo_rating = init_rating
        if k_factor is None:
            self._k_factor = KFactor(regress_rate=k_factor_regress_rate)
        else:
            self._k_factor = k_factor
        self._regress_rate = regress_rate
        self._reliability = reliability
        self._last_event_id = last_event_id
        self._non_alias_callers = list()
        self._alias_callers = list()
        self._deferred_complete = False
        self._temp_elo_rating = None
        self._current_event_id = None
        self._commit_complete = False
        self._lookback_length = lookback_length
        self._current_delta = None
        self._recent_lookback = {
            f1event.QUALIFYING: deque(list(), self._lookback_length),
            f1event.RACE: deque(list(), self._lookback_length)
        }

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
        self._current_delta = 0
        self._deferred_complete = True

    def update(self, delta):
        self.deferred_start_update()
        self._temp_elo_rating += delta
        self._current_delta += delta

    def update_reliability(self, km_success, km_failure):
        self.deferred_start_update()
        self._reliability.update(km_success, km_failure)

    def probability_finishing(self, start_km=0, race_distance_km=Reliability.DEFAULT_KM_PER_RACE):
        if self._reliability is None:
            return Reliability.DEFAULT_PROBABILITY
        return self._reliability.probability_finishing(start_km=start_km, race_distance_km=race_distance_km)

    def commit_update(self):
        if self._commit_complete:
            return
        # We need this first conditional if a driver or team never has an update called,
        # and therefore never has the deferred update run.
        if self._temp_elo_rating is None:
            self.deferred_start_update()
        self._elo_rating = self._temp_elo_rating
        # This has to be called before we reset the event ID and the current delta
        self.update_lookback()
        self._temp_elo_rating = None
        self._k_factor.increment_events()
        self._deferred_complete = False
        self._current_event_id = None
        self.reset_lists()
        if self._reliability is not None:
            self._reliability.commit_update()
        self._commit_complete = True

    def update_lookback(self, event_id=None, delta=None):
        lookback_event_id = event_id if event_id is not None else self._current_event_id
        lookback_delta = delta if delta is not None else self._current_delta
        if lookback_event_id is None:
            return
        # Grab the last two letters of the event since that's the event type.
        lookback_queue = self._recent_lookback.get(lookback_event_id[-2:])
        if lookback_queue is None:
            return
        lookback_queue.append(lookback_delta)

    def set_reliability(self, reliability):
        self._reliability = reliability

    def elo(self):
        return self._elo_rating

    def k_factor(self):
        return self._k_factor

    def reliability(self):
        return self._reliability

    def elo_delta(self):
        return self._current_delta

    def lookback_deltas(self, event_type):
        if event_type is None:
            return None
        return self._recent_lookback.get(event_type)

    def reset_lists(self):
        self._alias_callers.clear()
        self._non_alias_callers.clear()

    def regress(self, event_id):
        if self._last_event_id is None:
            return
        gap = year_gap(event_id, self._last_event_id)
        if not gap:
            return
        if self._reliability is not None:
            self._reliability.regress()
        self.regress_internal(gap)
        self._k_factor.regress(gap)
        self.reset_lookback()

    def regress_internal(self, gap):
        regress_factor = (1 - self._regress_rate) ** gap
        #        print('#Years: %2d BaseRate: %.3f ThisRate: %.3f %s [%s] [%s]' % (
        #              this_year - last_year, regress_rate, regress_factor, self._current_event_id,
        #              ', '.join(self._non_alias_callers), ', '.join(self._alias_callers)
        #        ))
        self._elo_rating *= regress_factor
        self._elo_rating += ((1 - regress_factor) * self._default_rating)

    def reset_lookback(self):
        for lookback in self._recent_lookback.values():
            lookback.clear()
