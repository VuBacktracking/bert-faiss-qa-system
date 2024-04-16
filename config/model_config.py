class ModelConfig:
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 384
    STRIDE = 384
    DATASET_NAME = "squad_v2"
    N_BEST = 20 # Số lượng kết quả tốt nhất được lựa chọn sau khi dự đoán
    MAX_ANS_LENGTH = 30 # Độ dài tối đa cho câu trả lời dự đoán
    EMBEDDING_COLUMN = "question_embedding"