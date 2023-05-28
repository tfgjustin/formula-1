import kafka_topic_names
import sys

from collections import defaultdict
from f1_logging import init_logging
from kafka_producers import F1TopicProducer
from kafka_topics import SimulationOutputBlock, SimulationRun
from time import time


def usage(program_name):
    print('Usage: %s <simulation_csv> <block_size>' % program_name)


def load_sims(filename):
    raw_sims = defaultdict(list)
    with open(filename, 'r') as infile:
        for line in infile.readlines():
            line = line.strip()
            parts = line.split(',')
            assert len(parts) == 3, 'Invalid sim line: "%s"' % line
            raw_sims[int(parts[1])].append(line)
    return raw_sims


def validate_one_block(sim_ids, current_idx, idx_range):
    start_sim_id = sim_ids[current_idx]
    end_sim_id = sim_ids[current_idx + idx_range]
    assert (end_sim_id - start_sim_id) == idx_range, 'Invalid index range'


def validate_sim_blocks(sim_ids, block_size):
    idx_range = block_size - 1
    current_idx = 0
    while (current_idx + block_size) <= len(sim_ids):
        validate_one_block(sim_ids, current_idx, idx_range)
        current_idx += block_size
    if current_idx == len(sim_ids):
        return
    # There may be a final set of sims which are less than the block size
    validate_one_block(sim_ids, current_idx, len(sim_ids) - current_idx - 1)


def chunk_and_send_one_block(run_id, raw_sims, sim_ids, current_idx, block_size, producer):
    start_sim_id = sim_ids[current_idx]
    end_sim_id = sim_ids[current_idx + block_size - 1]
    output_block = SimulationOutputBlock(run_id=run_id, start_idx=start_sim_id, end_idx=end_sim_id)
    for sim_id in range(start_sim_id, end_sim_id + 1):
        output_block.extend(raw_sims[sim_id])
    producer.send_message(output_block)


def chunk_and_send(run_id, raw_sims, block_size, producer):
    sim_ids = sorted(raw_sims.keys())
    current_idx = 0
    while (current_idx + block_size) <= len(sim_ids):
        chunk_and_send_one_block(run_id, raw_sims, sim_ids, current_idx, block_size, producer)
        current_idx += block_size
    if current_idx == len(sim_ids):
        return
    # One last chunk to send
    chunk_and_send_one_block(run_id, raw_sims, sim_ids, current_idx, len(sim_ids) - current_idx, producer)


def write_sims_to_producer(filename, block_size, sim_run_producer, sim_output_producer):
    run_id = int(time())
    raw_sims = load_sims(filename)
    validate_sim_blocks(sorted(raw_sims.keys()), block_size)
    sim_run_producer.send_message(SimulationRun(run_id=run_id, total_num_sims=len(raw_sims)))
    chunk_and_send(run_id, raw_sims, block_size, sim_output_producer)


def main(argv):
    if len(argv) < 2 or len(argv) > 3:
        usage(argv[0])
        return 1
    init_logging('fake-sim-publisher')
    sim_filename = argv[1]
    block_size = 100 if len(argv) == 2 else int(argv[2])
    sim_run_producer = F1TopicProducer(kafka_topic_names.SANDBOX_SIM_RUNS, dry_run=False, dry_run_verbose=False)
    sim_output_producer = F1TopicProducer(kafka_topic_names.SANDBOX_SIM_OUTPUT, dry_run=False, dry_run_verbose=False)
    write_sims_to_producer(sim_filename, block_size, sim_run_producer, sim_output_producer)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
