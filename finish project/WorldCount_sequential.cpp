#include<iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <cstring>  
#include <cctype>
#include <time.h>
#include <chrono>

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
    "she's", "should", "shouldn", "shouldn't", "should've", "so", 
    "some", "such", "t", "than", "that", "that'll", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", 
    "they'd", "they'll", "they're", "they've", "this", "those", "through",
    "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", 
    "wasn't", "we", "we'd", "we'll", "we're", "were", "weren", "weren't", 
    "we've", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd",
    "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've",NULL};

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

int main(int argc, char* argv[]){
    initialize_stopwords_seq();
    // 讀datasets名稱
    std::string filename = "../archive/text8"; 
    if (argc > 1) {
        std::string last_arg = argv[argc - 1];
        if (last_arg.find(".txt") != std::string::npos || 
            last_arg.find("text8") != std::string::npos || 
            last_arg.find("enwik9") != std::string::npos) {
            filename = last_arg;
        }
    }

    // read file
    auto total_start = std::chrono::high_resolution_clock::now();
    std::ifstream ifs(filename, std::ios::in);
    if (!ifs.is_open()) {
        std::cout << "Failed to open file.\n";
        return 1;
    }

    std::stringstream ss;
    ss << ifs.rdbuf();
    std::string str(ss.str());
    ifs.close(); 

    // unordermap 比 map更快
    std::unordered_map<std::string, int> word_counts;


    // split it into words
    char* token;
    const char* delimiters = " \t\n\r.,;:!?\"()[]{}<>-'";

    char* article_buffer = &str[0];

    char* saveptr;
    token = strtok_r(article_buffer, delimiters, &saveptr);

    while ((token !=NULL)) {
        // turn to lower
        for (int i = 0; token[i]; i++) {
            token[i] = tolower(token[i]);
        }

        // get the next token
        if (strlen(token) <= 2) {
            
            token = strtok_r(NULL, delimiters, &saveptr);
            continue;
        }

        // fliter out the word that has number or special char eg:123, c--
        bool is_word_char = true;
        if (strlen(token) == 0) { // strtok_r somtime generate empty token
            is_word_char = false;
        }

        for (int i = 0; token[i]; i++) {
            // isalnum() check if alpha or num
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

        word_counts[std::string(token)]++;
        token = strtok_r(NULL, delimiters, &saveptr);
    }
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_time = total_end - total_start;

    // word count
    std::cout << "--- Word Count Results (sequential)---" << std::endl;
    // for (const auto& pair : word_counts) {
    //     if (pair.second > 200000) { // show that exceeds 100
    //         std::cout << pair.first << ": " << pair.second << std::endl;
    //     }
    // }

    std::cout << "Total Time : " << total_time.count() << " ms" << std::endl;
    return 0;
}
