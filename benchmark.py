import subprocess
import re
from multiprocessing import Pool


def run_backtest(block_number):
    cmd = [
        "timeout",
        "1m",
        "./target/debug/backtest-build-block",
        "--config", "config-backtest-baseline.toml",
        "--builders", "mgp-ordering",
        "--builders", "mp-ordering",
        "--builders", "parallel",
        "--builders", "default-builder",
        str(block_number)
    ]

    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = e.stdout + "\n" + e.stderr
        if "Error: Block data not found" in output:
            return (block_number, None, None, "Block data not found")
        return (block_number, None, None, e.stderr)

    if "Error: Block data not found" in output:
        return (block_number, None, None, "Block data not found")

    match = re.search(r"Winning builder:\s+(\S+)\s+with profit:\s+([0-9.eE+-]+)", output)
    if match:
        builder = match.group(1)
        profit = float(match.group(2))
        return (block_number, builder, profit, None)
    else:
        return (block_number, None, None, "No winning builder found")


if __name__ == "__main__":
    block_numbers = [22220024, 22220019, 22220002]

    with Pool(processes=3) as pool:
        results = pool.map(run_backtest, block_numbers)

    for block, builder, profit, error in results:
        if error:
            print(f"Block {block}: Error - {error}")
        else:
            print(f"Block {block}: Winning builder = {builder}, Profit = {profit}")
    
    with open("results.csv", "w") as f:
        f.write("Block,Winning Builder,Profit,Error\n")
        for block, builder, profit, error in results:
            if error:
                f.write(f"{block},,,{error}\n")
            else:
                f.write(f"{block},{builder},{profit},\n")
    print("Results written to results.csv")

    
# This script runs a backtest for different block numbers using the specified command.