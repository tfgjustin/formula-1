RACE = 'RA'
RACE_OPENING = 'RO'
SPRINT_QUALIFYING = 'SQ'
QUALIFYING = 'QU'
SPRINT_RACE = 'SR'
SPRINT_SHOOTOUT = 'SS'
_EVENT_TYPE_TO_VALUE = {
    'RA': 90,   # Race: Always the last event of the weekend
    'RO': 89,   # Opening lap of a race: a pseudo-event which is part of a race but never surfaced externally.
    'SQ': 70,   # Sprint qualifying: A sprint (100km event) which sets the grid for a race
    'QU': 50,   # Qualifying: A pre-race/sprint event which sets the grid for either a race or sprint qualifying
    'SR': 30,   # Sprint race: A sprint which is itself a race
    'SS': 10    # Sprint shootout: A shortened qualifying event which sets the grid for a sprint race
}


def event_tag_to_value(event_tag):
    assert event_tag in _EVENT_TYPE_TO_VALUE, 'Invalid event tag: %s' % event_tag
    return _EVENT_TYPE_TO_VALUE.get(event_tag)


def compare_events(e1, e2):
    left = e1.split('-')
    right = e2.split('-')
    left_tag_value = event_tag_to_value(left[2])
    right_tag_value = event_tag_to_value(right[2])
    if left[0] < right[0]:
        return -1
    elif left[0] > right[0]:
        return 1
    if left[1] < right[1]:
        return -1
    elif left[1] > right[1]:
        return 1
    if left_tag_value < right_tag_value:
        return -1
    elif left_tag_value > right_tag_value:
        return 1
    else:
        return 0


class Event(object):

    def __init__(self, event_id, name, season, stage, date, event_type, num_laps, lap_distance_km, is_street_course,
                 weather, weather_probability=1.0):
        self._id = event_id
        self._name = name
        self._season = int(season)
        self._stage = int(stage)
        self._date = date
        self._type = event_type
        self._num_laps = int(num_laps)
        self._lap_distance_km = float(lap_distance_km)
        self._entrants = list()
        self._results = list()
        self._drivers = set()
        self._teams = set()
        self._is_street_course = is_street_course
        self._weather = weather
        self._weather_probability = weather_probability

    def add_entrant(self, entrant):
        if entrant.event().id() == self._id:
            self._entrants.append(entrant)
            self._drivers.add(entrant.driver())
            if entrant.team() is not None:
                self._teams.add(entrant.team())
            if entrant.has_result():
                self._results.append(entrant.result())

    def num_entrants(self):
        return len(self._results)

    def entrants(self):
        return self._entrants

    def has_results(self):
        return len(self._results) > 0

    def results(self):
        return self._results

    def drivers(self):
        return self._drivers

    def teams(self):
        return self._teams

    def id(self):
        return self._id

    def name(self):
        return self._name

    def season(self):
        return self._season

    def stage(self):
        return self._stage

    def date(self):
        return self._date

    def type(self):
        return self._type

    def is_street_course(self):
        return self._is_street_course

    def weather(self):
        return self._weather

    def weather_probability(self):
        return self._weather_probability

    def num_laps(self):
        return self._num_laps

    def lap_distance_km(self):
        return self._lap_distance_km

    def total_distance_km(self):
        return self._num_laps * self._lap_distance_km


class Qualifying(Event):
    def __init__(self, event_id, name, season, stage, date, num_laps, lap_distance_km, is_street_course, weather,
                 weather_probability=1.0):
        super().__init__(event_id, name, season, stage, date, QUALIFYING, num_laps, lap_distance_km, is_street_course,
                         weather, weather_probability=weather_probability)


class SprintShootout(Event):
    def __init__(self, event_id, name, season, stage, date, num_laps, lap_distance_km, is_street_course, weather,
                 weather_probability=1.0):
        super().__init__(event_id, name, season, stage, date, SPRINT_SHOOTOUT, num_laps, lap_distance_km,
                         is_street_course, weather, weather_probability=weather_probability)


class SprintQualifying(Event):
    def __init__(self, event_id, name, season, stage, date, num_laps, lap_distance_km, is_street_course, weather,
                 weather_probability=1.0):
        super().__init__(event_id, name, season, stage, date, SPRINT_QUALIFYING, num_laps, lap_distance_km,
                         is_street_course, weather, weather_probability=weather_probability)


class SprintRace(Event):
    def __init__(self, event_id, name, season, stage, date, num_laps, lap_distance_km, is_street_course, weather,
                 weather_probability=1.0):
        super().__init__(event_id, name, season, stage, date, SPRINT_RACE, num_laps, lap_distance_km, is_street_course,
                         weather, weather_probability=weather_probability)


class Race(Event):
    def __init__(self, event_id, name, season, stage, date, num_laps, lap_distance_km, is_street_course, weather,
                 weather_probability=1.0):
        super().__init__(event_id, name, season, stage, date, RACE, num_laps, lap_distance_km, is_street_course,
                         weather, weather_probability=weather_probability)


class EventFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def create_events(event_id, name, season, stage, date, num_laps, lap_distance_km, is_street_course, weather,
                      weather_probability=1.0):
        events = {}
        event_type = event_id[-2:]
        # breakpoint()
        if event_type == QUALIFYING:
            event = Qualifying(event_id, name, season, stage, date, num_laps, lap_distance_km, is_street_course,
                               weather, weather_probability=weather_probability)
            events[event_id] = event
        elif event_type == SPRINT_SHOOTOUT:
            event = SprintShootout(event_id, name, season, stage, date, num_laps, lap_distance_km, is_street_course,
                                   weather, weather_probability=weather_probability)
            events[event_id] = event
        elif event_type == SPRINT_QUALIFYING:
            event = SprintQualifying(event_id, name, season, stage, date, num_laps, lap_distance_km, is_street_course,
                                     weather, weather_probability=weather_probability)
            events[event_id] = event
        elif event_type == SPRINT_RACE:
            event = SprintRace(event_id, name, season, stage, date, num_laps,  lap_distance_km, is_street_course,
                               weather, weather_probability=weather_probability)
            events[event_id] = event
        elif event_type == RACE:
            event = Race(event_id, name, season, stage, date, num_laps, lap_distance_km, is_street_course, weather,
                         weather_probability=weather_probability)
            events[event_id] = event
        return events
