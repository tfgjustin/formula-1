import logging
import math

from collections import defaultdict


class F1Topic(object):
    def __init__(self, run_id=None):
        self.run_id = run_id

    def topic_key(self):
        assert 'topic_key() function is not implemented'

    def topic_subkey(self):
        return '(n/a)'


class SimulationRun(F1Topic):
    def __init__(self, run_id=None, total_num_sims=None, status='pending'):
        super(SimulationRun, self).__init__(run_id=run_id)
        self.total_num_sims = total_num_sims
        self.status = status
        self.fuzz_block_ids = defaultdict(list)
        self.sim_block_ids = defaultdict(list)
        self.tag_block_ids = defaultdict(list)

    def __str__(self):
        return 'SimulationRun: RunId: %s;#Sims: %d;Status: %s' % (self.run_id, self.total_num_sims, self.status)

    def topic_key(self):
        k = 'run_id=%s' % self.run_id
        return k.encode('utf-8')

    def merge(self, other_simulation_run):
        if self.run_id != other_simulation_run.run_id:
            return
        self.fuzz_block_ids.update(other_simulation_run.fuzz_block_ids)
        self.sim_block_ids.update(other_simulation_run.sim_block_ids)
        self.tag_block_ids.update(other_simulation_run.tag_block_ids)


class SimulationInput(F1Topic):
    def __init__(self, run_id=None, data_loader=None):
        super(SimulationInput, self).__init__(run_id=run_id)
        self.data_loader = data_loader

    def topic_key(self):
        k = 'run_id=%s' % self.run_id
        return k.encode('utf-8')


class FuzzRequestBlock(F1Topic):
    def __init__(self, run_id=None, start_idx=None, block_size=None):
        super(FuzzRequestBlock, self).__init__(run_id=run_id)
        self.start_idx = start_idx
        self.block_size = block_size

    def topic_key(self):
        k = 'run_id=%s;start_idx=%d' % (self.run_id, self.start_idx)
        return k.encode('utf-8')

    def topic_subkey(self):
        return 'block_size=%d' % self.block_size


class SimulationFuzzBlock(F1Topic):
    def __init__(self, run_id=None, start_idx=None, block_size=None, fuzz=None):
        super(SimulationFuzzBlock, self).__init__(run_id=run_id)
        self.start_idx = start_idx
        self.block_size = block_size
        # The output fuzz from the all_fuzz of the Fuzz object
        self.fuzz = fuzz

    def topic_key(self):
        k = 'run_id=%s;start_idx=%d' % (self.run_id, self.start_idx)
        return k.encode('utf-8')

    def topic_subkey(self):
        return 'block_size=%d' % self.block_size


class SimulatedEntity(object):
    def __init__(self, driver_name=None, team_id=None, start_position=None, end_position=None, num_laps=None,
                 full_input=None):
        self.driver_name = driver_name
        self.team_id = team_id
        self.start_position = start_position
        self.end_position = end_position
        self.num_laps = num_laps
        if full_input is not None:
            self.from_string(full_input)

    def __str__(self):
        return ':'.join([str(self.end_position), self.driver_name, self.team_id, str(self.start_position),
                         str(self.num_laps)])

    def __repr__(self):
        return self.__str__()

    def from_string(self, input_string):
        parts = input_string.split(':')
        assert len(parts) == 5, 'Invalid entity input string: "%s"' % input_string
        self.driver_name = parts[1]
        self.team_id = parts[2]
        self.start_position = int(parts[3])
        self.num_laps = int(parts[4])
        self.end_position = int(parts[0].replace('P0', '').replace('P', ''))

    def to_string(self):
        return 'P%02d:%s:%s:%d:%d' % (
            self.end_position, self.driver_name, self.team_id, self.start_position, self.num_laps
        )


class SimulationOutput(object):
    def __init__(self, event_id=None, sim_id=None, full_input=None):
        self.event_id = event_id
        self.sim_id = sim_id
        # End position to entity
        self.entities = dict()
        if full_input is not None:
            self.from_string(full_input)

    def __str__(self):
        s = 'SimulationOutput(event_id=%s, sim_id=%s, num_entities=%d)' % (
            self.event_id, self.sim_id, len(self.entities)
        )
        return s

    def from_string(self, input_string):
        parts = input_string.split(',')
        assert len(parts) == 3, 'Invalid simulation input string: "%s"' % input_string
        self.event_id = parts[0]
        self.sim_id = parts[1]
        entities = [SimulatedEntity(full_input=s) for s in parts[2].split('|')]
        self.entities = {e.end_position: e for e in entities}

    def to_string(self):
        return '%s,%s,%s' % (
            self.event_id, self.sim_id, '|'.join(
                [e[1].to_string() for e in sorted(self.entities.items())]
            )
        )


class SimulationOutputBlock(F1Topic):
    def __init__(self, run_id=None, start_idx=None, end_idx=None):
        super(SimulationOutputBlock, self).__init__(run_id=run_id)
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.simulations = list()

    def topic_key(self):
        k = 'run_id=%s;start_idx=%d' % (self.run_id, self.start_idx)
        return k.encode('utf-8')

    def topic_subkey(self):
        return 'end_idx=%d' % self.end_idx

    def append(self, input_string):
        # TODO: Validate that the simulation ID is within the ID range
        self.simulations.append(SimulationOutput(self.run_id, full_input=input_string))

    def extend(self, input_strings):
        for input_string in input_strings:
            self.append(input_string)

    def num_simulations(self):
        return len(self.simulations)


class TagOutputShard(F1Topic):
    def __init__(self, run_id=None, entity_id=None, event_id=None, start_idx=None, end_idx=None):
        super(TagOutputShard, self).__init__(run_id=run_id)
        self.entity_id = entity_id
        self.event_id = event_id
        self.start_idx = start_idx
        self.end_idx = end_idx
        # [tag_id] = [list of sim IDs]
        self.tags = defaultdict(set)

    def topic_key(self):
        k = 'run_id=%s;entity_id=%s;event_id=%s' % (self.run_id, self.entity_id, self.event_id)
        return k.encode('utf-8')

    def topic_subkey(self):
        return 'end_idx=%d' % self.end_idx

    def add_tag_occurrence(self, tag_id, sim_id):
        self.tags[tag_id].add(sim_id)


class TaggerOutput(object):
    def __init__(self, run_id=None, start_idx=None, end_idx=None):
        self.run_id = run_id
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.tags = defaultdict(dict)

    def add_tag(self, entity_id, event_id, tag_id, sim_id):
        tag_output_shard = self.tags.get(entity_id, {}).get(event_id)
        if tag_output_shard is None:
            tag_output_shard = TagOutputShard(run_id=self.run_id, entity_id=entity_id, event_id=event_id,
                                              start_idx=self.start_idx, end_idx=self.end_idx)
        tag_output_shard.add_tag_occurrence(tag_id, sim_id)
        self.tags[entity_id][event_id] = tag_output_shard


def search_block_available(sorted_start_indexes, blocks_dict, start_sim_idx, end_sim_idx):
    if not sorted_start_indexes:
        return True
    if len(sorted_start_indexes) == 1:
        block = blocks_dict[sorted_start_indexes[0]]
        return end_sim_idx < block.start_idx or start_sim_idx > block.end_idx
    mid_idx = math.floor(len(sorted_start_indexes) / 2)
    mid_block = blocks_dict[sorted_start_indexes[mid_idx]]
    if end_sim_idx < mid_block.start_idx:
        return search_block_available(sorted_start_indexes[:mid_idx], blocks_dict, start_sim_idx, end_sim_idx)
    elif start_sim_idx > mid_block.end_idx:
        return search_block_available(sorted_start_indexes[mid_idx + 1:], blocks_dict, start_sim_idx, end_sim_idx)
    else:
        # Overlapping block. We should not be here.
        return False


class TagOutputAggregator(object):
    def __init__(self, run_id=None, entity_id=None, event_id=None, total_num_sims=None):
        self.run_id = run_id
        self.entity_id = entity_id
        self.event_id = event_id
        self.total_num_sims = total_num_sims
        self.sims_received = 0
        self.blocks = dict()
        self.is_complete = False

    def add_tag_output_shard(self, tag_output_shard):
        if not self.tag_output_shard_matches(tag_output_shard):
            logging.warning('  Does not match this aggregator')
            return
        if not self.is_block_available(tag_output_shard):
            logging.warning('  Block is duplicate or askew')
            return
        self.blocks[tag_output_shard.start_idx] = tag_output_shard
        self.sims_received += (tag_output_shard.end_idx - tag_output_shard.start_idx + 1)
        self.is_complete = self.total_num_sims == self.sims_received
        logging.debug('  Received %6d/%6d (complete=%d)' % (self.sims_received, self.total_num_sims, self.is_complete))

    def tag_shards(self):
        return self.blocks.values()

    def tag_output_shard_matches(self, tag_output_shard):
        return self.run_id == tag_output_shard.run_id and self.entity_id == tag_output_shard.entity_id and \
                self.event_id == tag_output_shard.event_id

    def is_block_available(self, entity_tag_output):
        start_idx = entity_tag_output.start_idx
        end_idx = entity_tag_output.end_idx
        if start_idx in self.blocks:
            if self.blocks[start_idx].end_idx == end_idx:
                # Duplicate block
                logging.warning('Duplicate block:', entity_tag_output.topic_key())
                return False
        return search_block_available(sorted(self.blocks.keys()), self.blocks, start_idx, end_idx)
