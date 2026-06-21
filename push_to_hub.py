from huggingface_hub import HfApi, create_repo, upload_folder

token = ""
folder_path = "training/FastVLM_cls_r32_bs12_bg1.0_cmrd0.0/checkpoint-final"
repo_id = "vohuutridung/FastVLM_cls_r32_bs12_bg1.0_cmrd0.0"

create_repo(repo_id, token=token, exist_ok=True)

upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    token=token,
    path_in_repo="",
)

print("✅ Uploaded folder to Hugging Face successfully!")
