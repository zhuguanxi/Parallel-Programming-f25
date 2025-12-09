import subprocess
import re
import pandas as pd
import os
import sys

# ================= 參數設定區 =================

# 1. 執行次數 (取平均用)
NUM_RUNS = 5

# 2. 資料集路徑
DATASETS = [
    "../archive/text8",
    # "../enwik9/enwik9",
    "../pg11.txt"
]

# 3. 執行檔名稱
PROGRAMS = {
    "Sequential": "./WorldCount_sequential",
    "Sequential_map_reduce": "./WorldCount_map_reduce",
    "Pthread":    "./WorldCount_map_reduce_pthread",
    "OpenMP":     "./WorldCount_map_reduce_omp",
    "MPI":        "./WorldCount_map_reduce_mpi", 
    "CUDA":       "./WorldCount_cuda"
}

# 4. 測試的執行緒數量 (NP)
THREAD_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 20, 21]

# ==========================================

def parse_time(output_str, pattern):
    """ 使用 Regex 抓取時間數值 """
    match = re.search(pattern, output_str, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0  # 若抓不到預設為 0.0 以免加總出錯

def run_benchmark():
    results = []

    # 1. 先重新編譯確保執行檔最新
    print("正在編譯所有程式 (make all)...")
    try:
        subprocess.run(["make", "all"], check=True, stdout=subprocess.DEVNULL)
        print("編譯完成。\n")
    except subprocess.CalledProcessError:
        print("編譯失敗！請檢查 Makefile 或程式碼。")
        sys.exit(1)

    # 2. 開始迴圈測試
    for dataset in DATASETS:
        if not os.path.exists(dataset):
            print(f"警告: 找不到檔案 {dataset}，跳過。")
            continue
            
        dataset_name = os.path.basename(dataset)
        print(f"=== 正在測試資料集: {dataset_name} ===")

        for prog_name, binary_path in PROGRAMS.items():
            if not os.path.exists(binary_path):
                print(f"警告: 找不到執行檔 {binary_path}，跳過。")
                continue

            for np in THREAD_COUNTS:
                # 過濾不必要的測試組合
                if prog_name == "Sequential" and np > 1: continue
                if prog_name == "Sequential_map_reduce" and np > 1: continue
                if prog_name == "CUDA" and np > 1: continue

                # 初始化累加變數
                valid_runs = 0
                sum_map = 0.0
                sum_shuffle = 0.0
                sum_reduce = 0.0
                sum_total = 0.0

                print(f"  執行 [{prog_name}] (NP={np})", end="", flush=True)

                # --- 重複執行 NUM_RUNS 次 ---
                for r in range(NUM_RUNS):
                    # 建立指令
                    cmd = []
                    if prog_name == "MPI":
                        cmd = ["mpirun", "--oversubscribe", "-np", str(np), binary_path, dataset]
                    elif prog_name in ["Pthread", "OpenMP"]:
                        cmd = [binary_path, "-c", str(np), dataset]
                    else:
                        cmd = [binary_path, dataset]

                    try:
                        # 執行並擷取輸出
                        proc = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=300
                        )

                        if proc.returncode != 0:
                            print("x", end="", flush=True) # 失敗印 x
                            continue

                        # 解析輸出
                        out = proc.stdout
                        total_t = parse_time(out, r"Total\s*time\s*:\s*([\d\.]+)\s*ms")
                        
                        # 只有當成功抓到 Total Time 才算有效執行
                        if total_t > 0:
                            sum_map += parse_time(out, r"Map\s*time\s*:\s*([\d\.]+)\s*ms")
                            sum_shuffle += parse_time(out, r"Shuffle\s*time\s*:\s*([\d\.]+)\s*ms")
                            sum_reduce += parse_time(out, r"Reduce\s*time\s*:\s*([\d\.]+)\s*ms")
                            sum_total += total_t
                            valid_runs += 1
                            print(".", end="", flush=True) # 成功印 .
                        else:
                            print("?", end="", flush=True) # 解析失敗印 ?

                    except subprocess.TimeoutExpired:
                        print("T", end="", flush=True) # 超時印 T
                    except Exception as e:
                        print("E", end="", flush=True)

                # --- 計算平均並儲存 ---
                if valid_runs > 0:
                    avg_total = sum_total / valid_runs
                    avg_map = sum_map / valid_runs
                    avg_shuffle = sum_shuffle / valid_runs
                    avg_reduce = sum_reduce / valid_runs
                    
                    print(f" 完成 (平均: {avg_total:.2f} ms)")
                    
                    results.append({
                        "Dataset": dataset_name,
                        "Method": prog_name,
                        "NP": np,
                        "Avg Map Time (ms)": avg_map,
                        "Avg Shuffle Time (ms)": avg_shuffle,
                        "Avg Reduce Time (ms)": avg_reduce,
                        "Avg Total Time (ms)": avg_total,
                        "Runs": valid_runs # 紀錄實際成功次數
                    })
                else:
                    print(" 失敗 (無有效數據)")

    # 3. 儲存結果到 Excel
    if results:
        df = pd.DataFrame(results)
        output_file = "test_notask_ms.xlsx"
        df.to_excel(output_file, index=False)
        print(f"\n所有測試結束！平均結果已儲存至: {output_file}")
    else:
        print("\n沒有產生任何數據。")

if __name__ == "__main__":
    run_benchmark()