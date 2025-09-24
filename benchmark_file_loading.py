from pathlib import Path
import pandas as pd
from astropy.table import Table
import timeit


NROWS = 10_000_000


def process(data):
    data["f"] = data["f"] * 8


def get_data():
    data = pd.DataFrame({c: [124.1414] * NROWS for c in "abcdefghijklmnopqrstuvwxyz"})
    return data


def save_csv():
    fn = "testcsv.csv"
    get_data().to_csv(fn)


def load_csv():
    for t in pd.read_csv("testcsv.csv", chunksize=100_000):
        process(t)


def save_fits():
    fn = "testfits.fits"
    Table.from_pandas(get_data()).write(fn, format="fits", overwrite=True)


def load_fits():
    t = Table.read("testfits.fits")
    process(t)


if __name__ == "__main__":
    outfn = f"benchmark_file_loading_{NROWS}rows.txt"
    n = 10
    with open(outfn, "w") as f:
        for method in ["csv", "fits"]:
            print(f"--- {method} ---")
            f.write(f"--- {method} ---\n")
            save_res = (
                timeit.timeit(
                    f"save_{method}()",
                    setup=f"from __main__ import save_{method}",
                    number=n,
                )
                / n
            )
            print("saving took", save_res)
            f.write(f"saving took {save_res} s\n")
            sgb = Path(f"test{method}.{method}").stat().st_size / 1e9
            print("used [GB]", sgb)
            f.write(f"used {sgb} GB\n")
            load_res = (
                timeit.timeit(
                    f"load_{method}()",
                    setup=f"from __main__ import load_{method}",
                    number=n,
                )
                / n
            )
            print("loading took", load_res)
            f.write(f"loading took {load_res} s\n")
