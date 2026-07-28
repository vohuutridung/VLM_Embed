from huggingface_hub import create_repo, upload_folder

token = "" # your token

folder_path = "/workspace/VLM_Embed/training/SEGD_FastVLM_cls_r32_bs8_cka_last/checkpoint-final"
repo_id = "vohuutridung/SEGD_FastVLM_ImageNet_r32_bs8"

create_repo(repo_id=repo_id, token=token, exist_ok=True)
upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    token=token,
    path_in_repo="",
)

print("Uploaded folder to Hugging Face successfully!")
