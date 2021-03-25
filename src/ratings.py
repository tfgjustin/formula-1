import copy
import math
import numpy as np

from lifelines import KaplanMeierFitter


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
    _MAX_DECAY_RATE = 0.965
    _MIN_DECAY_RATE = 0.99
    _DEFAULT_OBSERVATION_SCALE = 1000
    DEFAULT_PROBABILITY = 0.7
    DEFAULT_KM_PER_RACE = 305.0

    class Observation(object):

        def __init__(self, length, failure, decay_rate=0.99, scale=None):
            if scale is None:
                scale = Reliability._DEFAULT_OBSERVATION_SCALE
            self._duration = np.full(scale, length, dtype=np.float)
            self._failure = np.full(scale, failure, dtype=np.bool)
            self._current_length = scale
            self._decay_rate = decay_rate

        def duration(self):
            return self._duration[:self._current_length]

        def failure(self):
            return self._failure[:self._current_length]

        def decay(self, decay_rate=None):
            if decay_rate is None:
                decay_rate = self._decay_rate
            self._current_length = math.floor(self._current_length * decay_rate)

        def current_length(self):
            return self._current_length

        def decay_rate(self):
            return self._decay_rate

    def __init__(self, default_decay_rate=0.98, other=None, regress_numerator=None, regress_percent=0.02):
        self._model = KaplanMeierFitter()
        self._template = None
        if other is None:
            self._km_success = 0
            self._km_failure = 0
            self._default_decay_rate = default_decay_rate
            self._regress_numerator = regress_numerator
            self._regress_percent = regress_percent
            self._observations = list()
            if self._regress_numerator is None:
                self._regress_numerator = 2 * self.DEFAULT_KM_PER_RACE
        else:
            self._default_decay_rate = other.decay_rate()
            self._km_success = other.km_success()
            self._km_failure = other.km_failure()
            self._regress_numerator = other.regress_numerator()
            self._regress_percent = other.regress_percent()
            self._observations = other.observations()
            self.regress()
            self._template = other

    def regress(self):
        if not self._km_success:
            return
        self.regress_observations()
        ratio = self._regress_numerator / self._km_success
        if ratio < 1.0:
            self._km_success *= ratio
            self._km_failure *= ratio
        self.regress_to_template()

    def regress_observations(self):
        self.cap_observations(self._regress_numerator / self.DEFAULT_KM_PER_RACE)
        if self._template is None:
            return
        self.decay_observations(decay_rate=(1 - self._regress_percent))
        template_copy = copy.deepcopy(self._template)
        template_copy.cap_observations(self._regress_numerator / self.DEFAULT_KM_PER_RACE)
        template_copy.decay_observations(decay_rate=self._regress_percent)
        self._observations.extend(template_copy.observations())

    def cap_observations(self, max_observations):
        scaled_max_observations = max_observations * self._DEFAULT_OBSERVATION_SCALE
        total_observations = self.total_observations()
        if not total_observations or scaled_max_observations > total_observations:
            return
        decay = scaled_max_observations / total_observations
        self.decay_observations(decay_rate=decay)

    def update_model(self):
        if not self._observations:
            return
        durations = np.array(self._observations[0].duration())
        failure = np.array(self._observations[0].failure())
        for observation in self._observations[1:]:
            durations = np.append(durations, observation.duration())
            failure = np.append(failure, observation.failure())
        self._model.fit(durations, failure)

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
        self._observations.append(self.Observation(km_success, km_failure > 0, decay_rate=self._default_decay_rate))

    def commit_update(self):
        # Update the survival models
        self.update_model()

    def decay(self):
        self.decay_observations()
        decay_rate = self.decay_rate()
        self._km_failure *= decay_rate
        self._km_success *= decay_rate

    def decay_observations(self, decay_rate=None):
        [observation.decay(decay_rate=decay_rate) for observation in self._observations]
        self._observations = list(filter(lambda obs: obs.current_length() > 10, self._observations))

    def probability_finishing_model(self, race_distance_km=DEFAULT_KM_PER_RACE):
        if not self._model:
            return self.DEFAULT_PROBABILITY
        if not hasattr(self._model, 'survival_function_'):
            return self.DEFAULT_PROBABILITY
        return self._model.predict(math.ceil(race_distance_km))

    def probability_finishing(self, race_distance_km=DEFAULT_KM_PER_RACE):
        # return self.probability_finishing_adhoc(distance_km)
        return self.probability_finishing_model(race_distance_km=race_distance_km)

    def probability_finishing_adhoc(self, race_distance_km=DEFAULT_KM_PER_RACE):
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
        return self._default_decay_rate

    def regress_numerator(self):
        return self._regress_numerator

    def regress_percent(self):
        return self._regress_percent

    def observations(self):
        return self._observations

    def total_observations(self):
        return sum([obs.current_length() for obs in self._observations])


class CarReliability(Reliability):

    def __init__(self, default_decay_rate=0.98, regress_numerator=(64 * Reliability.DEFAULT_KM_PER_RACE),
                 regress_percent=0.03):
        super().__init__(default_decay_rate=default_decay_rate, regress_numerator=regress_numerator,
                         regress_percent=regress_percent)


class DriverReliability(Reliability):

    def __init__(self, default_decay_rate=0.98, regress_numerator=(64 * Reliability.DEFAULT_KM_PER_RACE),
                 regress_percent=0.01):
        super().__init__(default_decay_rate=default_decay_rate, regress_numerator=regress_numerator,
                         regress_percent=regress_percent)


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
        self.regress(self._current_event_id)
        self._last_event_id = self._current_event_id
        self._temp_rating = self._rating
        # The reliability has to have its update happen after we regress in case we need to regress it, too.
        if self._reliability is not None:
            self._reliability.start_update()
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
        self._rating *= regress_factor
        self._rating += ((1 - regress_factor) * self._default_rating)
