from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import TrainingArguments
from transformers import Trainer
import evaluate

from utils.preprocess import preprocess_training_examples, preprocess_validation_examples
from utils.metric import compute_metrics

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 384
STRIDE = 384
DATASET_NAME = "squad_v2"

raw_datasets = load_dataset(DATASET_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

# SET UP DATASET
train_dataset = raw_datasets["train"].map(
          preprocess_training_examples ,
          batched=True, 
          remove_columns=raw_datasets["train"].column_names ,
          )

validation_dataset = raw_datasets["validation"].map(
      preprocess_validation_examples,
      batched=True, 
      remove_columns=raw_datasets["validation"].column_names ,
)

args = TrainingArguments(
    output_dir="distilbert-finetuned-squadv2", # Thư mục lưu output
    evaluation_strategy="no", # Chế độ đánh giá không tự động sau mỗi epoch
    save_strategy="epoch", # Lưu checkpoint sau mỗi epoch
    learning_rate=2e-5, # Tốc độ học
    num_train_epochs=3, # Số epoch huấn luyện
    weight_decay=0.01, # Giảm trọng lượng mô hình để tránh overfitting
    fp16=True, # Sử dụng kiểu dữ liệu half-precision để tối ưu tài nguyên
    push_to_hub=True, # Đẩy kết quả huấn luyện lên HuggingFace Hub )
)

# Khởi tạo một đối tượng Trainer để huấn luyện mô hình
trainer = Trainer(
      model=model, # Sử dụng mô hình đã tạo trước đó
      args=args, # Các tham số và cấu hình huấn luyện
      train_dataset=train_dataset, # Sử dụng tập dữ liệu huấn luyện
      eval_dataset=validation_dataset, # Sử dụng tập dữ liệu đánh giá
      tokenizer=tokenizer, # Sử dụng tokenizer để xử lý văn bản 
      )
# Bắt đầu quá trình huấn luyện
trainer.train()

# THỰC HIỆN ĐÁNH GIÁ MÔ HÌNH

# Tải metric "squad" từ thư viện evaluate
metric = evaluate.load("squad_v2")

# Thực hiện dự đoán trên tập dữ liệu validation
predictions , _, _ = trainer.predict(validation_dataset)

#Lấy ra thông tin về các điểm bắt đầu và ddiểm kết thúc của câu trả lời dự đoán
start_logits , end_logits = predictions
# Tính toán các chỉ số đánh giá sử dụng hàm compute_metrics
results = compute_metrics(
              start_logits ,
              end_logits ,
              validation_dataset ,
              raw_datasets["validation"]
            )

# In kết quả đánh giá mô hình
print(results)

trainer.push_to_hub(commit_message="Training complete")