import copy
import math


def event_to_year(event_id):
    return int(event_id[0:4])


def year_gap(event_id, last_event_id):
    last_year = event_to_year(last_event_id)
    this_year = event_to_year(event_id)
    return this_year - last_year


class KFactor(object):
    _MAX_FACTOR = 24
    _MIN_FACTOR = 12

    def __init__(self, regress_rate=0.0):
        self._num_events = 0
        self._regress_rate = regress_rate

    def factor(self):
        if not self._num_events:
            # Always skip the first event
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
    _KM_PER_RACE = 300.0

    def __init__(self, decay_rate=0.95):
        self._km_success = 0
        self._km_failure = 0
        self._decay_rate = decay_rate

    def update(self, km_success, km_failure):
        self.decay()
        self._km_failure += km_failure
        self._km_success += km_success

    def decay(self):
        self._km_failure *= self._decay_rate
        self._km_success *= self._decay_rate

    def probability_finishing(self):
        per_km_failure_rate = self._km_failure / (self._km_failure + self._km_success)
        return math.pow(per_km_failure_rate, self._KM_PER_RACE)


class EloRating(object):

    def __init__(self, init_rating, regress_rate=0.0, k_factor_regress_rate=0.0):
        self._default_rating = init_rating
        self._rating = init_rating
        self._k_factor = KFactor(regress_rate=k_factor_regress_rate)
        self._regress_rate = regress_rate
        self._reliability = None
        self._non_alias_callers = list()
        self._alias_callers = list()
        self._deferred_complete = False
        self._temp_rating = None
        self._last_event_id = None
        self._current_event_id = None
        self._commit_complete = False

    def start_update(self, event_id, caller_id, is_alias=False, base_reliability=base_reliability):
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
            self._reliability = copy.copy(base_reliability)

    def deferred_start_update(self):
        if self._deferred_complete:
            return
        if len(self._non_alias_callers) > 1:
            print('ERROR: Multiple canonical callers for %s: [%s]' % (
                self._current_event_id, ', '.join(self._non_alias_callers)
            ))
        if self._temp_rating is not None:
            print('ERROR: start_update was called on this rating multiple times')
        self.regress(self._current_event_id)
        self._last_event_id = self._current_event_id
        self._temp_rating = self._rating
        self._deferred_complete = True

    def update(self, delta):
        self.deferred_start_update()
        self._temp_rating += delta

    def commit_update(self):
        if self._commit_complete:
            return
        # We need this first conditional if a driver or team never has an update called,
        # and therefore never has the deferred update run.
        if self._temp_rating is None:
            self.deferred_start_update()
        self._rating = self._temp_rating
        self._temp_rating = None
        self._k_factor.increment_events()
        self._deferred_complete = False
        self._current_event_id = None
        self.reset_lists()
        self._commit_complete = True

    def rating(self):
        return self._rating

    def k_factor(self):
        return self._k_factor

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
        self.regress_internal(gap)
        self._k_factor.regress(gap)

    def regress_internal(self, gap):
        regress_factor = (1 - self._regress_rate) ** gap
        #        print('#Years: %2d BaseRate: %.3f ThisRate: %.3f %s [%s] [%s]' % (
        #              this_year - last_year, regress_rate, regress_factor, self._current_event_id,
        #              ', '.join(self._non_alias_callers), ', '.join(self._alias_callers)
        #        ))
        self._rating *= regress_factor
        self._rating += ((1 - regress_factor) * self._default_rating)
