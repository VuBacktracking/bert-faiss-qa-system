import torch
from datasets import load_dataset
from transformers import AutoTokenizer

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 384
STRIDE = 384
DATASET_NAME = "squad_v2"
raw_datasets = load_dataset(DATASET_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_training_examples(examples):
  # Trích xuất danh sách câu hỏi từ examples và loại bỏ các khoảng trắng dư thừa
  questions = [q.strip() for q in examples["questions"]]
  inputs = tokenizer(
      questions,
      examples["context"],
      max_length=MAX_LENGTH,
      truncation = "only_second",
      stride = STRIDE,
      return_overflowing_tokens=True,
      return_offsets_mapping=True,
      padding = "max_length"
  )
  
  # Trích xuất offset_mapping từ inputs và loại bỏ nó ra khỏi inputs
  offset_mapping = inputs.pop("offset_mapping")
  
  # Trích xuất sample_map từ inputs và loại bỏ nó ra khỏi inputs
  sample_map = inputs.pop("overflow_to_sample_mapping")
  
  # Trích xuất thông tin về câu trả lời (answers) từ examples
  answers = examples["answers"]
  
  # Khởi tạo danh sách các vị trí bắt đầu và kết thúc câu trả lời
  start_positions = [] 
  end_positions = []
  
  # Duyệt qua danh sách offset_mapping
  for i, offset in enumerate(offset_mapping):
    # Xác định index của mẫu (sample) liên quan đến offset hiện tại
    sample_idx = sample_map[i]
    
    # Trích xuất sequence_ids từ inputs
    sequence_ids = inputs.sequence_ids(i)
    
    # Xác định vị trí bắt đầu và kết thúc của ngữ cảnh
    idx = 0
    while sequence_ids[idx] != 1:
      idx += 1
    context_start = idx
    while sequence_ids[idx] == 1:
      idx += 1
    context_end = idx - 1
    
    # Trích xuất thông tin về câu trả lời cho mẫu này
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
        # Nếu không, gán vị trí bắt đầu và kết thúc dựa trên vị trí của các mã thông tin
        idx = context_start
        while idx <= context_end and offset[idx][0] <= start_char:
          idx += 1
        start_positions.append(idx - 1)
        
        idx = context_end
        while idx >= context_start and offset[idx][1] >= end_char:
          idx -= 1 
        end_positions.append(idx + 1)
  # Thêm thông tin vị trí bắt đầu và kết thúc vào inputs
  inputs["start_positions"] = start_positions 
  inputs["end_positions"] = end_positions
  
  return inputs

def preprocess_validation_examples(examples):
  # Chuẩn bị danh sách câu hỏi bằng cách loại bỏ khoảng trắng ở đầu và cuối mỗi câu hỏi
  questions = [q.strip() for q in examples["question"]]

  # Sử dụng tokenizer để mã hóa các câu hỏi và văn bản liên quan
  inputs = tokenizer(
      questions,
      examples["context"],
      max_length=MAX_LENGTH,
      truncation = "only_second",
      stride = STRIDE,
      return_overflowing_tokens = True,
      return_offsets_mapping = True,
      padding = "max_length",
  )

  # Lấy ánh xạ để ánh xạ lại ví dụ tham chiếu cho từng dòng trong inputs
  sample_map = inputs.pop("overflow_to_sample_mapping")
  example_ids = []

  # Xác định ví dụ tham chiếu cho mỗi dòng đầu vào và điều chỉnh ánh xạ offset
  for i in range(len(inputs["input_ids"])):
    sample_idx = sample_map[i]
    example_ids.append(examples["id"][sample_idx])

    sequence_ids = inputs.sequence_ids(i)
    offset = inputs["offset_mapping"][i]

    # Loại bỏ các offset không phù hợp với sequence_ids
    inputs["offset_mapping"][i] =[o if sequence_ids[k] == 1 else None for k, o in enumerate(offset) ]

  # Thêm thông tin ví dụ tham chiếu vào đầu vào
  inputs["example_id"] = example_ids

  return inputs