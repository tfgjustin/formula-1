import kafka_config
import kafka_topic_names
import logging
import math
import sys

from f1_logging import init_logging
from kafka_consumers import F1TopicConsumer
from kafka_producers import F1TopicProducer
from kafka_topics import TaggerOutput


def driver_name_to_entity_id(driver_name):
    driver_name = driver_name.lower()
    if driver_name.endswith('_jr'):
        driver_name = driver_name.replace('_jr', '')
    if 'chumacher' not in driver_name or 'mick_' in driver_name:
        return '_'.join(driver_name.split('_')[1:])
    return driver_name.lower()


def print_one_finish_position(sim_id, event_id, position, entity, num_entities, tagger_output):
    if not hasattr(entity, 'driver_name'):
        return
    entity_id = driver_name_to_entity_id(entity.driver_name)
    tagger_output.add_tag(entity_id, event_id, 'exact=%d' % position, sim_id)
    compare_position = 0.5
    while compare_position < position:
        tagger_output.add_tag(entity_id, event_id, 'over=%.1f' % compare_position, sim_id)
        compare_position += 1
    while compare_position < num_entities:
        tagger_output.add_tag(entity_id, event_id, 'under=%.1f' % compare_position, sim_id)
        compare_position += 1


def print_finish_position(sim_output, sim_id, event_id, tagger_output):
    entities = sim_output.entities
    num_entities = len(entities)
    for end_position, entity in entities.items():
        print_one_finish_position(sim_id, event_id, end_position, entity, num_entities, tagger_output)


def print_h2h_for_driver(sim_id, event_id, entity, defeated_entities, tagger_output):
    if not hasattr(entity, 'driver_name'):
        return
    entity_id = driver_name_to_entity_id(entity.driver_name)
    for defeated in defeated_entities:
        defeated_entity_id = driver_name_to_entity_id(defeated.driver_name)
        tagger_output.add_tag(entity_id, event_id, defeated_entity_id, sim_id)


def print_h2h_results(sim_output, sim_id, event_id, tagger_output):
    entities = sim_output.entities
    sorted_results = [entity for _, entity in sorted(entities.items())]
    for idx, entity in enumerate(sorted_results):
        print_h2h_for_driver(sim_id, event_id, entity, sorted_results[idx+1:], tagger_output)


def process_classified(sim_id, event_id, num_classified, num_entries, tagger_output):
    event_entity_id = '_event_'
    tagger_output.add_tag(event_entity_id, event_id, 'classify_exact=%d' % num_classified, sim_id)
    # classify_under
    under = num_entries + 0.5
    while num_classified < under:
        tagger_output.add_tag(event_entity_id, event_id, 'classify_under=%.1f' % under, sim_id)
        under -= 1
    over = 0.5
    while num_classified > over:
        tagger_output.add_tag(event_entity_id, event_id, 'classify_over=%.1f' % over, sim_id)
        over += 1


def process_classified_finishers(sim_output, sim_id, event_id, tagger_output):
    entities = sim_output.entities
    if 1 not in entities:
        return
    max_laps = int(entities[1].num_laps)
    classify_laps = math.floor(max_laps * 0.9)
    num_classified = 0
    for entity in entities.values():
        if not hasattr(entity, 'driver_name'):
            continue
        if entity.num_laps >= classify_laps:
            num_classified += 1
            tagger_output.add_tag(driver_name_to_entity_id(entity.driver_name), event_id, 'classify=y', sim_id)
        else:
            tagger_output.add_tag(driver_name_to_entity_id(entity.driver_name), event_id, 'classify=n', sim_id)
    process_classified(sim_id, event_id, num_classified, len(entities), tagger_output)


def process_one_sim(sim_output, tagger_output):
    sim_id = int(sim_output.sim_id)
    event_id = sim_output.event_id
    process_classified_finishers(sim_output, sim_id, event_id, tagger_output)
    print_finish_position(sim_output, sim_id, event_id, tagger_output)
    print_h2h_results(sim_output, sim_id, event_id, tagger_output)


def process_one_consumer_record(sim_output_block, producer):
    run_id = sim_output_block.run_id
    start_idx = sim_output_block.start_idx
    end_idx = sim_output_block.end_idx
    tagger_output = TaggerOutput(run_id=run_id, start_idx=start_idx, end_idx=end_idx)
    for sim in sim_output_block.simulations:
        process_one_sim(sim, tagger_output)
    for entity_tag_output_shards in tagger_output.tags.values():
        for tag_output_shard in entity_tag_output_shards.values():
            producer.send_message(tag_output_shard)


def process_topics(consumer, producer):
    msg = consumer.consume_message()
    while msg is not None:
        topic = msg.topic
        message = msg.value
        assert topic == kafka_topic_names.SANDBOX_SIM_OUTPUT, 'tagger received message from unknown topic: %s' % topic
        logging.info('Tagger.consume(topic="%s" key="%s" subkey="%s")' % (
            topic, message.topic_key(), message.topic_subkey()
        ))
        process_one_consumer_record(message, producer)
        msg = consumer.consume_message()


def main(argv):
    if len(argv) != 2:
        print('Usage: %s <config_txt>' % argv[0])
        return 1
    init_logging('tagger')
    configuration = kafka_config.parse_configuration_file(argv[1])
    consumer_config = kafka_config.get_configuration_dict(configuration, kafka_config.CLIENTS_CONSUMER)
    producer_config = kafka_config.get_configuration_dict(configuration, kafka_config.CLIENTS_PRODUCER)
    consumer = F1TopicConsumer(kafka_topic_names.SANDBOX_SIM_OUTPUT, group_id='tagger', **consumer_config)
    producer = F1TopicProducer(kafka_topic_names.SANDBOX_TAG_OUTPUT, dry_run=False, dry_run_verbose=False,
                               compression_type='gzip', **producer_config)
    process_topics(consumer, producer)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
