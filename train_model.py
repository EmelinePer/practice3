import os
import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from huggingface_hub import HfApi, HfFolder, create_repo, upload_file

from sklearn.linear_model import LogisticRegression

def train_and_save_model(output_path: str = "model.joblib"):
    # Create tiny synthetic dataset
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        random_state=42,
    )
    model = LogisticRegression()
    model.fit(X, y)

    # Save locally
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")
    return output_path

def upload_to_hf(local_model_path: str, hf_repo_id: str, hf_token: str):
    if not hf_token:
        raise ValueError("HF_TOKEN is not set. Cannot upload to Hugging Face.")

    api = HfApi()
    HfFolder.save_token(hf_token)

    # Create the repo if it doesn't exist (under your user or org)
    create_repo(repo_id=hf_repo_id, token=hf_token, private=False, exist_ok=True)

    # Upload file to the root of the repo
    upload_file(
        path_or_fileobj=local_model_path,
        path_in_repo="model.joblib",
        repo_id=hf_repo_id,
        token=hf_token,
    )
    print(f"Uploaded {local_model_path} to hf repo {hf_repo_id}")

if __name__ == "__main__":
    # Read repo id from env, e.g. "EmelinePer/my-ml-hf-demo"
    hf_repo_id = os.getenv("HF_REPO_ID")
    hf_token = os.getenv("HF_TOKEN")

    model_path = train_and_save_model()

    if hf_repo_id and hf_token:
        upload_to_hf(model_path, hf_repo_id, hf_token)
    else:
        print("HF_REPO_ID or HF_TOKEN not set, skipping upload to Hugging Face.")