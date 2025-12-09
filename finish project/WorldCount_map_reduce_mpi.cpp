#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cstring>
#include <cctype>
#include <algorithm>
#include <iomanip> 
#include <unordered_set>
#include <unordered_map>
#include <chrono>

struct KVPair {
    std::string key;
    int value;
};

const char *stopwords[] = {"a", "about", 
    "above", "after", "again", "against", "ain",
    "all", "am", "an", "and", "any", "are", "aren", 
    "aren't", "as", "at", "be", "because", "been", 
    "before", "being", "below", "between", "both", 
    "but", "by", "can", "couldn", "couldn't", "d",
    "did", "didn", "didn't", "do", "does", "doesn", 
    "doesn't", "doing", "don", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn",
    "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't",
    "having", "he", "he'd", "he'll", "her", "here", "hers", "herself",
    "he's", "him", "himself", "his", "how", "i", "i'd", "if", "i'll",
    "i'm", "in", "into", "is", "isn", "isn't", "it", "it'd", "it'll",
    "it's", "its", "itself", "i've", "just", "ll", "m", "ma", "me", 
    "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my",
    "myself", "needn", "needn't","no", "nor", "not", "now", "o", "of", 
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", 
    "re", "s", "same", "shan", "shan't", "she", "she'd", "she'll",
    "she's", "should", "shouldn", "shouldn't", "should_ve", "so", 
    "some", "such", "t", "than", "that", "that'll", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", 
    "they'd", "they'll", "they're", "they've", "this", "those", "through",
    "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", 
    "wasn't", "we", "we'd", "we'll", "we're", "were", "weren", "weren't", 
    "we've", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd",
    "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've", NULL};

std::unordered_set<std::string> stopwords_set;

void initialize_stopwords_seq() {
    if (stopwords_set.empty()) {
        for (int i = 0; stopwords[i] != NULL; i++) {
            stopwords_set.insert(stopwords[i]);
        }
    }
}

bool is_stopword_fast_seq(const std::string& word) {
    return stopwords_set.count(word) > 0; 
}

// Map function
std::vector<KVPair> map_function(char* article_chunk) {
    std::vector<KVPair> mapped_data;
    mapped_data.reserve(4096); 

    const char* delimiters = " \t\n\r.,;:!?\"()[]{}<>-'";
    char* saveptr;
    char* token = strtok_r(article_chunk, delimiters, &saveptr);

    while (token != NULL) {
        for (int i = 0; token[i]; i++) {
            token[i] = tolower(token[i]);
        }

        if (strlen(token) <= 2 ) {
            token = strtok_r(NULL, delimiters, &saveptr);
            continue;
        }

        bool is_word_char = true;
        if (strlen(token) == 0) is_word_char = false;

        for (int i = 0; token[i]; i++) {
            if (!isalnum(token[i]) && token[i] != '_') { 
                is_word_char = false;
                break;
            }
        }

        if (!is_word_char) {
            token = strtok_r(NULL, delimiters, &saveptr);
            continue;
        }

        std::string current_word(token);
        if (is_stopword_fast_seq(current_word)) {
            token = strtok_r(NULL, delimiters, &saveptr);
            continue;
        }

        mapped_data.push_back({current_word, 1});

        token = strtok_r(NULL, delimiters, &saveptr);
    }
    return mapped_data;
}

int main(int argc, char** argv) {
    initialize_stopwords_seq();
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // std::string filename = "../enwik9/enwik9";
    std::string filename = "../archive/text8";

    if (argc > 1) {
        std::string last_arg = argv[argc - 1];
        if (last_arg.find(".txt") != std::string::npos || 
            last_arg.find("text8") != std::string::npos || 
            last_arg.find("enwik9") != std::string::npos) {
            filename = last_arg;
        }
    }

    auto t_total_start = std::chrono::high_resolution_clock::now();

    // Rank 0 取得大小並廣播，大家自己讀
    long long total_len_ll = 0;

    if (rank == 0) {
        std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
        if (!ifs.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        total_len_ll = static_cast<long long>(ifs.tellg());
        ifs.close();
    }

    // 廣播總長度
    MPI_Bcast(&total_len_ll, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (total_len_ll <= 0) {
        if (rank == 0) std::cerr << "File is empty or cannot get size.\n";
        MPI_Finalize();
        return 0;
    }
    // auto t_total_start = std::chrono::high_resolution_clock::now();
    auto t_map_start = std::chrono::high_resolution_clock::now();
    auto t_scatter_start = std::chrono::high_resolution_clock::now();

    // 計算各自負責的區塊
    long long base = total_len_ll / size;
    long long rem  = total_len_ll % size;
    long long local_len_ll;
    long long local_offset_ll;

    if (rank < rem) {
        local_len_ll    = base + 1;
        local_offset_ll = rank * (base + 1);
    } else {
        local_len_ll    = base;
        local_offset_ll = rem * (base + 1) + (rank - rem) * base;
    }

    int local_len = static_cast<int>(local_len_ll);
    std::vector<char> local_buf(local_len + 1, 0);

    {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) {
            std::cerr << "Rank " << rank << " failed to open file.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        ifs.seekg(local_offset_ll);
        ifs.read(local_buf.data(), local_len);
    }
    local_buf[local_len] = '\0';

    auto t_scatter_end = std::chrono::high_resolution_clock::now();

    // Map Phase
    std::vector<KVPair> local_mapped = map_function(local_buf.data());
    auto t_map_end = std::chrono::high_resolution_clock::now();

    // Shuffle Phase
    auto t_shuffle_start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::string> send_buffers(size);
    for(int i=0; i<size; i++) send_buffers[i].reserve(local_mapped.size() * 10 / size);

    for (const auto& pair : local_mapped) {
        std::hash<std::string> hasher;
        size_t h = hasher(pair.key);
        int dest_rank = h % size;
        send_buffers[dest_rank] += pair.key + " 1\n";
    }

    std::vector<int> send_data_sizes(size);
    std::vector<int> recv_data_sizes(size);
    for (int i = 0; i < size; i++) send_data_sizes[i] = static_cast<int>(send_buffers[i].size());

    MPI_Alltoall(send_data_sizes.data(), 1, MPI_INT,
                recv_data_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> sdispls(size), rdispls(size);
    int total_recv_size = 0;
    int total_send_size = 0;
    
    sdispls[0] = 0; rdispls[0] = 0;
    for (int i = 0; i < size; i++) {
        if (i > 0) {
            sdispls[i] = sdispls[i-1] + send_data_sizes[i-1];
            rdispls[i] = rdispls[i-1] + recv_data_sizes[i-1];
        }
        total_send_size += send_data_sizes[i];
        total_recv_size += recv_data_sizes[i];
    }

    std::string flat_send_buf;
    flat_send_buf.reserve(total_send_size);
    for (const auto& s : send_buffers) flat_send_buf += s;

    std::vector<char> flat_recv_buf(total_recv_size);

    MPI_Alltoallv(flat_send_buf.data(), send_data_sizes.data(), sdispls.data(), MPI_CHAR,
                flat_recv_buf.data(), recv_data_sizes.data(), rdispls.data(), MPI_CHAR,
                MPI_COMM_WORLD);
    
    auto t_shuffle_end = std::chrono::high_resolution_clock::now();
    auto t_reduce_start = std::chrono::high_resolution_clock::now();
    std::unordered_map<std::string, int> counts;
    
    if (total_recv_size > 0) {
        std::string recv_str(flat_recv_buf.begin(), flat_recv_buf.end());
        std::stringstream ss(recv_str);
        std::string key;
        int val;
        while (ss >> key >> val) {
            counts[key] += val;
        }
    }

    std::vector<KVPair> final_results;
    final_results.reserve(counts.size());
    for (const auto& pair : counts) {
        final_results.push_back({pair.first, pair.second});
    }

    // Gather all results to Rank 0
    std::stringstream ss_local_res;
    for (const auto& pair : final_results) {
        ss_local_res << pair.key << " " << pair.value << "\n";
    }
    std::string local_res_str = ss_local_res.str();
    int local_res_len = static_cast<int>(local_res_str.size());

    std::vector<int> recvcounts;
    if (rank == 0) recvcounts.resize(size);

    MPI_Gather(&local_res_len, 1, MPI_INT,
                rank == 0 ? recvcounts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displss;
    std::vector<char> all_results_buf;
    if (rank == 0) {
        displss.resize(size);
        int total_len_int = 0;
        for (int i = 0; i < size; i++) {
            displss[i] = total_len_int;
            total_len_int += recvcounts[i];
        }
        all_results_buf.resize(total_len_int);
    }

    MPI_Gatherv(local_res_str.data(), local_res_len, MPI_CHAR,
                rank == 0 ? all_results_buf.data() : nullptr,
                rank == 0 ? recvcounts.data() : nullptr,
                rank == 0 ? displss.data() : nullptr, MPI_CHAR,
                0, MPI_COMM_WORLD);

    auto t_reduce_end = std::chrono::high_resolution_clock::now();
    auto t_total_end = std::chrono::high_resolution_clock::now();

    double local_total_time     = std::chrono::duration<double>(t_total_end - t_total_start).count();
    double local_total_map_time = std::chrono::duration<double>(t_map_end - t_map_start).count();
    double local_scatter_time   = std::chrono::duration<double>(t_scatter_end - t_scatter_start).count();
    double local_truemap_time   = local_total_map_time - local_scatter_time;
    double local_shuffle_time   = std::chrono::duration<double>(t_shuffle_end - t_shuffle_start).count();
    double local_reduce_time    = std::chrono::duration<double>(t_reduce_end - t_reduce_start).count();
    
    double max_true_map_time, max_shuffle_time, max_reduce_time, max_total_time;
    double max_scatter_time;
    double max_total_map_time;
    MPI_Reduce(&local_truemap_time, &max_true_map_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_shuffle_time, &max_shuffle_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_reduce_time, &max_reduce_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_scatter_time, &max_scatter_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_map_time, &max_total_map_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "--- MPI MapReduce Results (Method 1: Parallel IO + Hash Aggregation) ---" << std::endl;
        
        std::string all_res_str(all_results_buf.begin(), all_results_buf.end());
        std::stringstream ss_all(all_res_str);
        std::string key;
        int val;
        
        std::map<std::string, int> global_results_map;
        while (ss_all >> key >> val) {
            global_results_map[key] = val;
        }

        // for (const auto& pair : global_results_map) {
        //     if (pair.second > 200000) {
        //         std::cout << pair.first << ": " << pair.second << std::endl;
        //     }
        // }

        std::cout << std::fixed << std::setprecision(4); 
        std::cout << "Total Time     : " << max_total_time*1000     << " ms" << std::endl;
        std::cout << "Map Time       : " << max_total_map_time*1000 << " ms" << std::endl;
        std::cout << "True Map Time  : " << max_true_map_time*1000  << " ms" << std::endl;
        std::cout << "IO Read Time   : " << max_scatter_time*1000   << " ms" << std::endl;
        std::cout << "Shuffle Time   : " << max_shuffle_time*1000   << " ms" << std::endl;
        std::cout << "Reduce Time    : " << max_reduce_time*1000    << " ms" << std::endl;
    }

    MPI_Finalize();
    return 0;
}