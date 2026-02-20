from pathlib import Path

import kagglehub


def main() -> None:
    path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
    dataset_dir = Path(path)

    source = dataset_dir / "heart.csv"
    destination = Path("data/heart.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(f"heart.csv not found in {dataset_dir}")

    destination.write_bytes(source.read_bytes())
    print(f"dataset copied to: {destination}")


if __name__ == "__main__":
    main()
