class Season(object):
    def __init__(self, season):
        self._season = int(season)
        self._events = dict()
        self._drivers = set()
        self._teams = set()

    def add_event(self, event):
        if event.season() == self._season:
            self._events[event.id()] = event

    def year(self):
        return self._season

    def events(self):
        return self._events

    def identify_participants(self):
        for event in self._events.values():
            self._drivers.update(event.drivers())
            self._teams.update(event.teams())

