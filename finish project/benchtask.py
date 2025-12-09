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
    # "../archive/text8",
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
    return 0.0

def get_optimized_core_list(n_threads):
    """
    針對 i5-13500HX (6P + 8E) 的最佳綁核策略：
    1. P-Cores Physical (0, 2, 4, 6, 8, 10) - 最優先，獨佔 P 核
    2. P-Cores Logical  (1, 3, 5, 7, 9, 11) - 次優先，開啟 P 核 HT
    3. E-Cores          (12 ~ 19)           - 最後，使用 E 核
    """
    
    # P-Core 的實體核心 (偶數)
    p_cores_physical = [0, 2, 4, 6, 8, 10]
    
    # P-Core 的 HT 邏輯核心 (奇數)
    p_cores_logical = [1, 3, 5, 7, 9, 11]
    
    # E-Cores (通常在 P-Core 之後)
    e_cores = list(range(12, 20)) # 12 到 19
    
    # 串接優先順序
    preferred_order = p_cores_physical + p_cores_logical + e_cores
    
    # 取出前 n_threads 個
    # 如果 n_threads 超過 CPU 總數 (20)，就用循環或取模的方式 (這裡簡單用取模防止 IndexError)
    selected_cores = []
    for i in range(n_threads):
        core_id = preferred_order[i % len(preferred_order)]
        selected_cores.append(core_id)
    
    # 轉成字串，例如 NP=7 -> "0,2,4,6,8,10,1"
    return ",".join(map(str, selected_cores))

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

                valid_runs = 0
                sum_map = 0.0
                sum_shuffle = 0.0
                sum_reduce = 0.0
                sum_total = 0.0

                print(f"  執行 [{prog_name}] (NP={np})", end="", flush=True)

                for r in range(NUM_RUNS):
                    cmd = []
                    
                    # [修改點] 使用針對架構優化的綁核列表
                    core_range = get_optimized_core_list(np)
                    
                    if prog_name == "MPI":
                        # --oversubscribe 允許 process 數量超過實體核心數
                        cmd = ["taskset", "-c", core_range, "mpirun", "--oversubscribe", "-np", str(np), binary_path, dataset]
                    
                    elif prog_name == "Pthread":
                        cmd = ["taskset", "-c", core_range, binary_path, "-c", str(np), dataset]
                        
                    elif prog_name == "OpenMP":
                        cmd = ["taskset", "-c", core_range, binary_path, "-c", str(np), dataset]
                        
                    elif prog_name == "CUDA":
                        cmd = ["taskset", "-c", "0", binary_path, dataset]
                    else:
                        cmd = ["taskset", "-c", "0", binary_path, dataset]

                    # 環境變數設定
                    my_env = os.environ.copy()
                    if prog_name == "OpenMP":
                        my_env["OMP_PROC_BIND"] = "true" 
                        my_env["OMP_NUM_THREADS"] = str(np)

                    try:
                        proc = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=300,
                            env=my_env 
                        )

                        if proc.returncode != 0:
                            print("x", end="", flush=True) 
                            continue

                        out = proc.stdout
                        total_t = parse_time(out, r"Total\s*time\s*:\s*([\d\.]+)\s*ms")
                        
                        if total_t > 0:
                            sum_map += parse_time(out, r"Map\s*time\s*:\s*([\d\.]+)\s*ms")
                            sum_shuffle += parse_time(out, r"Shuffle\s*time\s*:\s*([\d\.]+)\s*ms")
                            sum_reduce += parse_time(out, r"Reduce\s*time\s*:\s*([\d\.]+)\s*ms")
                            sum_total += total_t
                            valid_runs += 1
                            print(".", end="", flush=True) 
                        else:
                            print("?", end="", flush=True) 

                    except subprocess.TimeoutExpired:
                        print("T", end="", flush=True) 
                    except Exception as e:
                        print("E", end="", flush=True)

                if valid_runs > 0:
                    avg_total = sum_total / valid_runs
                    avg_map = sum_map / valid_runs
                    avg_shuffle = sum_shuffle / valid_runs
                    avg_reduce = sum_reduce / valid_runs
                    
                    # 可以在這裡印出 core_range 來確認綁定順序是否正確
                    # print(f" (Cores: {core_range})", end="") 
                    print(f" 完成 (平均: {avg_total:.2f} ms)")
                    
                    results.append({
                        "Dataset": dataset_name,
                        "Method": prog_name,
                        "NP": np,
                        "Avg Map Time (ms)": avg_map,
                        "Avg Shuffle Time (ms)": avg_shuffle,
                        "Avg Reduce Time (ms)": avg_reduce,
                        "Avg Total Time (ms)": avg_total,
                        "Runs": valid_runs
                    })
                else:
                    print(f" 失敗 (無有效數據) Cores: {core_range}")

    if results:
        df = pd.DataFrame(results)
        output_file = "test_optimized_cores_13500HX.xlsx"
        df.to_excel(output_file, index=False)
        print(f"\n所有測試結束！平均結果已儲存至: {output_file}")
    else:
        print("\n沒有產生任何數據。")

if __name__ == "__main__":
    run_benchmark()