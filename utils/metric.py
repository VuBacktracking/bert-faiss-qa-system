from tqdm.auto import tqdm
import collections
import evaluate
import numpy as np

metric = evaluate.load("squad_v2")

N_BEST = 20 # Số lượng kết quả tốt nhất được lựa chọn sau khi dự đoán
MAX_ANS_LENGTH = 30 # Độ dài tối đa cho câu trả lời dự đoán

def compute_metrics(start_logits, end_logits, features, examples):
  # Tạo một từ điển mặc định để ánh xạ mỗi ví dụ với danh sách các đặc trưng tương ứng
  example_to_features = collections.defaultdict(list)
  for idx, feature in enumerate(features):
    example_to_features[features["example_id"]].append(idx)
    
  predicted_answers = []
  for example in tqdm(examples):
    example_id = example["id"]
    context = example["context"]
    answers = []
    
    # Lặp qua tất cả các đặc trưng liên quan đến ví dụ đó
    for feature_index in example_to_features[example_id]:
      start_logit = start_logits[feature_index]
      end_logit = end_logits[feature_index]
      offsets = features[feature_index]['offset_mapping']
      
      # Lấy các chỉ số có giá trị lớn nhất cho start và end logits
      start_indexes = np.argsort(start_logit)[-1: -N_BEST - 1: -1].tolist()
      end_indexes = np.argsort(end_logit)[-1:-N_BEST -1:-1].tolist()
      
      for start_index in start_indexes:
        for end_index in end_indexes:
          # Bỏ qua các câu trả lời không hoàn toàn nằm trong ngữ cảnh
          if offsets[start_index] is None or offsets[end_index] is None:
            continue
          
          # Bỏ qua các câu trả lời có độ dài > max_answer_length
          if end_index - start_index + 1 > MAX_ANS_LENGTH:
            continue
          
          # Tạo một câu trả lời mới
          text = context[
              offsets[start_index][0]:offsets[end_index][1]
          ]
          
          logit_score = start_logit[start_index] + end_logit[end_index]
          
          answer = {
            "text" :  text,
            "logit_score": logit_score
          }
          answers.append(answer)
          
    if len(answers) > 0:
      best_answer = max(answers, key=lambda x: x["logit_score"])
      answer_dict = {
        "id": example_id,
        "prediction_text": best_answer["text"],
        "no_answer_probability": 1 - best_answer["logit_score"]
      }
    else:
      answer_dict = {
        "id": example_id,
        "prediction_text": "",
        "no_answer_probability": 1.0
      }
    predicted_answers.append(answer_dict)
  
  # Tạo danh sách câu trả lời lý thuyết từ các ví dụ
  theoretical_answers = [
          {'id': ex['id'], 'answers': ex['answers']} for ex in examples
  ]
  
  return metric.compute(
    predictions=predicted_answers,
    references=theoretical_answers
  )