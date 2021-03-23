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
    DEFAULT_PROBABILITY = 0.7
    DEFAULT_KM_PER_RACE = 305.0

    def __init__(self, decay_rate=0.97, other=None):
        if other is None:
            self._km_success = 0
            self._km_failure = 0
            self._decay_rate = decay_rate
        else:
            self._decay_rate = other.decay_rate()
            self._km_success = other.km_success()
            self._km_failure = other.km_failure()
            # Since the template we're going to use is taken using 20-30 other drivers, the number of KM of success and
            # failure will be much larger than for one driver. Normalize the number of success KM down to the maximum
            # number of KM one driver can generate. Since the limit of the sum of a geometric sequence with a ratio of
            # $R is 1/(1-$R) we can figure out the maximum value for success.
            max_success_km = self.DEFAULT_KM_PER_RACE / (1 - self._decay_rate)
            ratio = self._km_success / max_success_km
            if ratio < 1.0:
                self._km_success *= ratio
                self._km_failure *= ratio

    def start_update(self):
        self.decay()

    def update(self, km_success, km_failure):
        self._km_failure += km_failure
        self._km_success += km_success

    def commit_update(self):
        # This is a no-op for right now.
        return

    def decay(self):
        self._km_failure *= self._decay_rate
        self._km_success *= self._decay_rate

    def probability_finishing(self, race_distance_km=DEFAULT_KM_PER_RACE):
        denominator = self._km_failure + self._km_success
        if not denominator:
            return self.DEFAULT_PROBABILITY
        per_km_success_rate = self._km_success / denominator
        return math.pow(per_km_success_rate, race_distance_km)

    def km_success(self):
        return self._km_success

    def km_failure(self):
        return self._km_failure

    def decay_rate(self):
        return self._decay_rate


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
        if self._temp_rating is not None:
            print('ERROR: start_update was called on this rating multiple times')
        if self._reliability is not None:
            self._reliability.start_update()
        self.regress(self._current_event_id)
        self._last_event_id = self._current_event_id
        self._temp_rating = self._rating
        self._deferred_complete = True

    def update(self, delta):
        self.deferred_start_update()
        self._temp_rating += delta

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
        if self._temp_rating is None:
            self.deferred_start_update()
        self._rating = self._temp_rating
        self._temp_rating = None
        self._k_factor.increment_events()
        self._deferred_complete = False
        self._current_event_id = None
        self.reset_lists()
        self._commit_complete = True
        if self._reliability is not None:
            self._reliability.commit_update()

    def rating(self):
        return self._rating

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
