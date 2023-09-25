RACE = 'RA'
SPRINT_QUALIFYING = 'SQ'
QUALIFYING = 'QU'
SPRINT_RACE = 'SR'
SPRINT_SHOOTOUT = 'SS'
_EVENT_TYPE_TO_VALUE = {
    RACE:              90,   # Race: Always the last event of the weekend
    SPRINT_QUALIFYING: 70,   # Sprint qualifying: A sprint (100km event) which sets the grid for a race
    QUALIFYING:        50,   # Qualifying: A pre-race/sprint event which sets the grid for a race or sprint qualifying
    SPRINT_RACE:       30,   # Sprint race: A sprint which is itself a race
    SPRINT_SHOOTOUT:   10    # Sprint shootout: A shortened qualifying event which sets the grid for a sprint race
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


class Phase(object):
    def __init__(self, event, phase_id, sequence_number, num_laps, lap_distance_km):
        self._event = event
        self._phase_id = phase_id
        self._sequence_number = sequence_number
        self._num_laps = int(num_laps)
        self._lap_distance_km = float(lap_distance_km)
        self._entrants = list()
        self._results = list()
        self._results = list()
        self._drivers = set()
        self._teams = set()

    def add_entrant(self, entrant):
        if entrant.event().id() == self._event.id():
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
        return f'{self._event.id()}-{self._phase_id}'

    def num_laps(self):
        return self._num_laps

    def lap_distance_km(self):
        return self._lap_distance_km

    def total_distance_km(self):
        return self._num_laps * self._lap_distance_km


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
        self._phases = list()

    def add_entrant(self, entrant):
        if entrant.event().id() == self._id:
            self._entrants.append(entrant)
            self._drivers.add(entrant.driver())
            if entrant.team() is not None:
                self._teams.add(entrant.team())
            if entrant.has_result():
                self._results.append(entrant.result())

    def add_phase(self, phase):
        self._phases.append(phase)

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

    def phases(self):
        return self._phases


class Qualifying(Event):
    def __init__(self, event_id, name, season, stage, date, num_laps, lap_distance_km, is_street_course, weather,
                 weather_probability=1.0):
        super().__init__(event_id, name, season, stage, date, QUALIFYING, num_laps, lap_distance_km, is_street_course,
                         weather, weather_probability=weather_probability)
        # TODO: Add phases for qualifying
        # https://racingnews365.com/every-formula-1-qualifying-format-ever
        # 1950-1995: Two-day format; fastest time from across two days
        # 1996-2002: One hour shootout; 12 laps to set the fastest time
        # 2003-2004: One lap on each of two days; Friday set order for Saturday, and Saturday for the race
        # 2005 (first 6 rounds): Low fuel Saturday, race fuel Sunday, add the two times
        # 2005 (remaining rounds): Back to 2003-2004 format
        # 2006-2015: Current Q1/Q2/Q3 format
        # 2016 (first 2 rounds): Weird knockout qualifying
        # 2016 (round 3) to present: Current Q1/Q2/Q3 format


class SprintShootout(Event):
    def __init__(self, event_id, name, season, stage, date, num_laps, lap_distance_km, is_street_course, weather,
                 weather_probability=1.0):
        super().__init__(event_id, name, season, stage, date, SPRINT_SHOOTOUT, num_laps, lap_distance_km,
                         is_street_course, weather, weather_probability=weather_probability)
        # TODO: Add phases for Q1/Q2/Q3 qualifying


class SprintQualifying(Event):
    def __init__(self, event_id, name, season, stage, date, num_laps, lap_distance_km, is_street_course, weather,
                 weather_probability=1.0):
        super().__init__(event_id, name, season, stage, date, SPRINT_QUALIFYING, num_laps, lap_distance_km,
                         is_street_course, weather, weather_probability=weather_probability)
        # TODO: Add phases for opening lap, and full race distance


class SprintRace(Event):
    def __init__(self, event_id, name, season, stage, date, num_laps, lap_distance_km, is_street_course, weather,
                 weather_probability=1.0):
        super().__init__(event_id, name, season, stage, date, SPRINT_RACE, num_laps, lap_distance_km, is_street_course,
                         weather, weather_probability=weather_probability)
        # TODO: Add phases for opening lap, and full race distance


class Race(Event):
    def __init__(self, event_id, name, season, stage, date, num_laps, lap_distance_km, is_street_course, weather,
                 weather_probability=1.0):
        super().__init__(event_id, name, season, stage, date, RACE, num_laps, lap_distance_km, is_street_course,
                         weather, weather_probability=weather_probability)
        # TODO: Add phases for opening lap, partial distance, and full race distance


class EventFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def create_events(event_id, name, season, stage, date, num_laps, lap_distance_km, is_street_course, weather,
                      weather_probability=1.0):
        events = {}
        event_type = event_id[-2:]
        if event_type == QUALIFYING:
            events[event_id] = Qualifying(event_id, name, season, stage, date, num_laps, lap_distance_km,
                                          is_street_course, weather, weather_probability=weather_probability)
        elif event_type == SPRINT_SHOOTOUT:
            events[event_id] = SprintShootout(event_id, name, season, stage, date, num_laps, lap_distance_km,
                                              is_street_course, weather, weather_probability=weather_probability)
        elif event_type == SPRINT_QUALIFYING:
            events[event_id] = SprintQualifying(event_id, name, season, stage, date, num_laps, lap_distance_km,
                                                is_street_course, weather, weather_probability=weather_probability)
        elif event_type == SPRINT_RACE:
            events[event_id] = SprintRace(event_id, name, season, stage, date, num_laps,  lap_distance_km,
                                          is_street_course, weather, weather_probability=weather_probability)
        elif event_type == RACE:
            events[event_id] = Race(event_id, name, season, stage, date, num_laps, lap_distance_km, is_street_course,
                                    weather, weather_probability=weather_probability)
        return events
