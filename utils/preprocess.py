import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import yaml

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the YAML file
with open('cfg/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

DATASET_NAME = config["DATSET_NAME"]
MODEL_NAME = config["MODEL_NAME"]
MAX_LENGTH = config["MAX_LENGTH"]
STRIDE = config["STRIDE"]
raw_datasets = load_dataset(DATASET_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_training_examples(examples):
    # Extract the list of questions from examples and remove any extra whitespace
    questions = [q.strip() for q in examples["questions"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Extract offset_mapping from inputs and remove it from inputs
    offset_mapping = inputs.pop("offset_mapping")

    # Extract sample_map from inputs and remove it from inputs
    sample_map = inputs.pop("overflow_to_sample_mapping")

    # Extract information about answers from examples
    answers = examples["answers"]

    # Initialize lists for start and end positions of answers
    start_positions = []
    end_positions = []

    # Iterate through the offset_mapping list
    for i, offset in enumerate(offset_mapping):
        # Determine the sample index related to the current offset
        sample_idx = sample_map[i]

        # Extract sequence_ids from inputs
        sequence_ids = inputs.sequence_ids(i)

        # Determine the start and end positions of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # Extract answer information for this sample
        answer = answers[sample_idx]

        if len(answer["text"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])

            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise, assign start and end positions based on the positions of the offset
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
    # Add start and end position information to inputs
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs

def preprocess_validation_examples(examples):
    # Prepare the list of questions by removing leading and trailing whitespace
    questions = [q.strip() for q in examples["question"]]

    # Use tokenizer to encode the questions and related text
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Get mapping to remap reference examples for each line in inputs
    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    # Determine the reference example for each input line and adjust offset mapping
    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]

        # Remove offsets not matching sequence_ids
        inputs["offset_mapping"][i] = [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)]

    # Add reference example information to inputs
    inputs["example_id"] = example_ids

    return inputs