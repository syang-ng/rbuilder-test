import os
import subprocess
import re
from multiprocessing import Pool


def run_backtest(block_number):
    if os.path.exists(f"outputs-t20/{block_number}.txt"):
        print(f"Output for block {block_number} already exists. Skipping.")
        return
        # return (block_number, None, None, "Output already exists")
    # 21748156
    # timeout 2m ./target/release/backtest-build-block --config config-backtest-baseline.toml --builders mgp-ordering --builders mp-ordering --builders parallel --builders default 21748156
    cmd = [
        "timeout",
        "2m",
        "./build-t20/release/backtest-build-block",
        "--config", "config-backtest-baseline.toml",
        "--builders", "mgp-ordering",
        "--builders", "mp-ordering",
        "--builders", "parallel",
        "--builders", "default",
        str(block_number)
    ]
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = e.stdout + "\n" + e.stderr
    
    with open(f"outputs-t20/{block_number}.txt", "w") as f:
        f.write(output)
        # if "Error: Block data not found" in output:
            # return (block_number, None, None, "Block data not found")
        # return (block_number, None, None, e.stderr)

    # if "Error: Block data not found" in output:
    #     return (block_number, None, None, "Block data not found")

    # match = re.search(r"Winning builder:\s+(\S+)\s+with profit:\s+([0-9.eE+-]+)", output)
    # if match:
    #     builder = match.group(1)
    #     profit = float(match.group(2))
    #     return (block_number, builder, profit, None)
    # else:
    #     return (block_number, None, None, "No winning builder found")
import sqlite3

# with sqlite3.connect("/root/.rbuilder/backtest/main.sqlite") as conn:
#     cursor = conn.cursor()
  
#     # cursor.execute("SELECT distinct(block_number) from orders where block_number>21765264 and block_number<=21768564;")
#     # cursor.execute("SELECT distinct(block_number) from orders where block_number>21768564 and block_number<=21769300;")
#     # cursor.execute("SELECT distinct(block_number) from orders where block_number>21769750 and block_number<= 21769820;")
#     rows = cursor.fetchall()
#     block_numbers_in_db = {row[0] for row in rows}
#     print(f"Total blocks in DB: {len(block_numbers_in_db)}")

import os
files = os.listdir('outputs-v4')
block_numbers_in_db = [i.split('.')[0] for i in files if i.endswith('.txt')]


if __name__ == "__main__":
    # block_numbers = [22220024, 22220019, 22220002]
    block_numbers = list(block_numbers_in_db)
    print(f"Total blocks to process: {len(block_numbers)}")
    import random
    random.shuffle(block_numbers)

    with Pool(processes=4) as pool:
        results = pool.map(run_backtest, block_numbers)
    # for block_number in block_numbers:
    #     run_backtest(block_number)

    # for block, builder, profit, error in results:
    #     if error:
    #         print(f"Block {block}: Error - {error}")
    #     else:
    #         print(f"Block {block}: Winning builder = {builder}, Profit = {profit}")
    
    # with open("results.csv", "w") as f:
    #     f.write("Block,Winning Builder,Profit,Error\n")
    #     for block, builder, profit, error in results:
    #         if error:
    #             f.write(f"{block},,,{error}\n")
    #         else:
    #             f.write(f"{block},{builder},{profit},\n")
    # print("Results written to results.csv")

    
# This script runs a backtest for different block numbers using the specified command.