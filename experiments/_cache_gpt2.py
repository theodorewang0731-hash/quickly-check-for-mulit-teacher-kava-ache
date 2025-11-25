from huggingface_hub import snapshot_download
import os

def main():
    target = r"H:\hf_cache\gpt2"
    os.makedirs(target, exist_ok=True)
    print(f"Starting snapshot_download to {target}")
    path = snapshot_download(repo_id="gpt2", cache_dir=target, resume_download=True)
    print("snapshot_download finished. Path:", path)

if __name__ == '__main__':
    main()
