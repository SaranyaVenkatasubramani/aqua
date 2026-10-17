from __future__ import annotations

from components.engine import MODEL_PATH, load_dataset, train_bundle


def main() -> None:
    frame = load_dataset()
    print(
        f"Loaded {len(frame):,} readings from {frame.index.min()} through "
        f"{frame.index.max()}."
    )
    print("Fitting leak-free 2018 profiles and the notebook's classifier...")
    bundle = train_bundle(frame, MODEL_PATH)
    print(
        f"Saved {MODEL_PATH} "
        f"({len(bundle['feature_cols'])} features; "
        f"scikit-learn {bundle['sklearn_version']})."
    )


if __name__ == "__main__":
    main()
