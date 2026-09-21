from pathlib import Path
import shutil
import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_DIR = PROJECT_ROOT / "data" / "raw" / "home_credit_default_risk"

def download_dataset() -> str:
    print("Downloading latest version of Home Credit Default Risk dataset via kagglehub...")
    try:
        path = kagglehub.competition_download("home-credit-default-risk")
    except Exception as e:
        print(f"Authentication notice: {e}")
        print("Authenticating with KaggleHub (please enter Kaggle credentials or token)...")
        kagglehub.login()
        path = kagglehub.competition_download("home-credit-default-risk")

    print("Path to competition files:", path)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for item in Path(path).iterdir():
        target = TARGET_DIR / item.name
        if not target.exists():
            if item.is_file():
                shutil.copy2(item, target)
            elif item.is_dir():
                shutil.copytree(item, target)
    print(f"Dataset ready in: {TARGET_DIR}")
    return path

if __name__ == "__main__":
    download_dataset()
