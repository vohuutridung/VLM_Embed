from huggingface_hub import upload_folder

# 🪪 Access token của bạn
token = ""

# 📁 Thư mục bạn muốn upload (ví dụ: model, checkpoints, v.v.)
folder_path = "/workspace/ComfyUI/models/gligen/VLM_Embed/training/no_deepspeed_propose_kd_weight/checkpoint-final"

# 🏷️ Repo đã có sẵn trên Hugging Face
repo_id = "DVLe/vlm_propose_hateful"

# 🚀 Upload toàn bộ folder lên repo đó
upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    token=token,
    path_in_repo="",     # thư mục gốc trong repo, có thể đổi ví dụ "models/"
)

print("✅ Đã upload folder lên Hugging Face thành công!")
