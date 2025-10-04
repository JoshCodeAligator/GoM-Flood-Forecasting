import argparse
from pathlib import Path
from .io import load_baseline
from .visualize import save_simple

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["check-baseline"], required=True)
    args = p.parse_args()

    if args.task == "check-baseline":
        df = load_baseline()
        # Example: quick plot of Red/Emerson baseline value (demonstration)
        s = df.query("basin=='Red' and station=='Emerson' and metric=='flow'").set_index("asof_date")["value_SI"]
        save_simple("Red – Emerson baseline flow (SI)", s, Path("reports/figures/red_emerson_baseline.png"))

if __name__ == "__main__":
    main()
