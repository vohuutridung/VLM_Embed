from huggingface_hub import create_repo, upload_folder

# 🪪 Access token của bạn
token = ""

# 📁 Thư mục bạn muốn upload (ví dụ: model, checkpoints, v.v.)
folder_path = "/workspace/VLM_Embed/training/SGD_FastVLM_full_cls_r32_bs16/checkpoint-final"

# 🏷️ Repo trên Hugging Face
repo_id = "sonspeed/SGD_FastVLM_full_cls_r32_bs16"

# 🚀 Tạo repo (nếu chưa có) rồi upload
create_repo(repo_id=repo_id, token=token, exist_ok=True)
upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    token=token,
    path_in_repo="",     # thư mục gốc trong repo, có thể đổi ví dụ "models/"
)

print("✅ Đã upload folder lên Hugging Face thành công!")
