"""End-to-end orchestrator: simulate -> preprocess -> model -> visualize."""

from src.simulate_data import main as simulate
from src.visualize import make_all


def main() -> None:
    print("=" * 60)
    print("Step 1/2: Simulating restaurant data...")
    print("=" * 60)
    simulate()

    print("\n" + "=" * 60)
    print("Step 2/2: Running preprocessing, models, and visualizations...")
    print("=" * 60)
    paths = make_all()

    print("\n" + "=" * 60)
    print("Done. Generated figures:")
    print("=" * 60)
    for k, v in paths.items():
        print(f"  {k:>22}  ->  {v}")


if __name__ == "__main__":
    main()
