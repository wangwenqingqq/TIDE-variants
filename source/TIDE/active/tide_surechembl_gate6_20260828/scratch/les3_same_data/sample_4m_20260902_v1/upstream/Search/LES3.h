//
//  LES3.h
//  LES3
//

#ifndef LES3_h
#define LES3_h

#include <iostream>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <ctime>
#include <queue>
#include <set>
#include <unordered_set>
#include <map>
#include <sstream>
#include <filesystem>
#include "MEASURE.h"
#include "TGM_C.h"
using namespace std;
// typedef pair<float, multiset<int>> entry;
// typedef pair<float, int> entry_construct;
struct LES3 {
    // TGM tgm;
    
    int num_groups;
    TGM_C tgm;
    Measure measure;
    int total_num_sets;
    // vector<set<int>> database;
    vector<vector<multiset<int>>> grouped_database;
    vector<multiset<int>> database;
    vector<vector<int64_t>> grouped_ids;
    vector<int64_t> database_ids;
    vector<vector<int>> groups;
    LES3() = default;
    LES3(string path_to_sets, string path_to_groups, string path_to_ids = "") {
        preprocess(path_to_sets, path_to_groups, path_to_ids);
        tgm.construct(path_to_sets, path_to_groups);
    };
    
    int findKNN(multiset<int> &query_set, int k);
    int findDeltaNN(multiset<int> &query_set, float delta);
    vector<int64_t> findDeltaNNExact(const multiset<int> &query_set, uint32_t numerator, uint32_t denominator, int *checked_count = nullptr);
    void preprocess(string path_to_sets, string path_to_groups, string path_to_ids);
    void insert_helper(multiset<int> &set_to_insert);
    void insertExact(const multiset<int> &set_to_insert, int64_t stable_id);
    int get_size_in_MB();
    void insert(string path, float insert_ratio);
    double testKNN(int result_size);
    double testDeltaNN(float delta);
   
};
void LES3::preprocess(string path_to_sets, string path_to_groups, string path_to_ids) {
    // vector<multiset<int>> database;
    ifstream in(path_to_sets);
    string line;
    while(getline(in, line)) {
        istringstream line_stream(line);
        string temp;
        multiset<int> temp_set;
        while(line_stream>>temp) {
            temp_set.insert(stoi(temp));
        }
        database.push_back(temp_set);
    }
    in.close();
    total_num_sets = database.size();
    if(path_to_ids.empty()) {
        database_ids.reserve(database.size());
        for(size_t i = 0; i < database.size(); i++) {
            database_ids.push_back(static_cast<int64_t>(i));
        }
    }
    else {
        ifstream id_in(path_to_ids);
        int64_t stable_id;
        while(id_in >> stable_id) {
            database_ids.push_back(stable_id);
        }
        if(database_ids.size() != database.size()) {
            throw runtime_error("LES3 ID count does not match set count");
        }
    }

    in.open(path_to_groups);
    while(getline(in, line)) {
        vector<int> current_group;
        istringstream line_stream(line);
        string temp;
        while(line_stream>>temp) {
            int set_id = stoi(temp);
            current_group.push_back(set_id);
        }
        groups.push_back(current_group);
    }
   
    //construct grouped database
    for(int i = 0; i < groups.size(); i++) {
        vector<multiset<int>> this_group;
        vector<int64_t> this_group_ids;
        for(int set_id : groups[i]) {
            if(set_id < 0 || static_cast<size_t>(set_id) >= database.size()) {
                throw runtime_error("LES3 group contains an out-of-range set ID");
            }
            this_group.push_back(database[set_id]);
            this_group_ids.push_back(database_ids[set_id]);
        }
        grouped_database.push_back(this_group);
        grouped_ids.push_back(this_group_ids);
    }
    in.close();
    // database.clear();
    
    num_groups = static_cast<int>(groups.size());
    ::num_groups = num_groups;
   
}
int LES3::get_size_in_MB() {
    return tgm.get_size();
}
int LES3::findKNN(multiset<int> &query_set, int k) {
    int check_counts = 0;
    priority_queue<pair<float, int> > candidate_groups;
    for(unsigned j = 0; j < num_groups; j++) {
        float ub = tgm.getUB(query_set, j);
        candidate_groups.push({ub, j});
    }
    
    priority_queue<pair<float, multiset<int>>, vector<pair<float, multiset<int>>>, std::greater<pair<float, multiset<int>>> > result;
    while(!candidate_groups.empty()) {
        if(result.size() == k && result.top().first >= candidate_groups.top().first) {
            break;
        }
        for(auto &candidate_set : grouped_database[candidate_groups.top().second]) {
            check_counts++;
            
            float sim = measure.computeSim(query_set, candidate_set);
            if(result.size() >= k && sim < result.top().first)
                continue;
            
            result.push({sim,candidate_set});
            if(result.size() > k)
                result.pop();
        }
        candidate_groups.pop();
    }
    // cout<<check_counts<<endl;
    return check_counts;
}

int LES3::findDeltaNN(multiset<int> &query_set, float delta) {
    int check_counts = 0;
    priority_queue<pair<float, int> > candidate_groups;
    for(unsigned j = 0; j < num_groups; j++) {
        float ub = tgm.getUB(query_set, j);
        candidate_groups.push({ub, j});
    }
    vector<multiset<int>> result;
    while(!candidate_groups.empty() && candidate_groups.top().first >= delta) {
        // total_checked_count++;
        for(auto &candidate_set : grouped_database[candidate_groups.top().second]) {
            check_counts++;
           
            float sim = measure.computeSim(query_set, candidate_set);
            if(sim >= delta)
                result.push_back(candidate_set);
        }
        candidate_groups.pop();
    }
    return check_counts;
}

static uint32_t les3_intersection_size(const multiset<int> &left, const multiset<int> &right) {
    uint32_t count = 0;
    auto left_it = left.begin();
    auto right_it = right.begin();
    while(left_it != left.end() && right_it != right.end()) {
        if(*left_it < *right_it) {
            ++left_it;
        }
        else if(*right_it < *left_it) {
            ++right_it;
        }
        else {
            ++count;
            ++left_it;
            ++right_it;
        }
    }
    return count;
}

vector<int64_t> LES3::findDeltaNNExact(const multiset<int> &query_set,
                                        uint32_t numerator,
                                        uint32_t denominator,
                                        int *checked_count) {
    if(numerator == 0 || numerator > denominator || query_set.empty()) {
        throw invalid_argument("LES3 exact threshold requires 0 < numerator <= denominator and a nonempty query");
    }
    int checked = 0;
    vector<int64_t> result;
    for(int group_id = 0; group_id < num_groups; group_id++) {
        uint32_t group_common = 0;
        for(int token : query_set) {
            if(tgm.bit_map[group_id].contains(token)) {
                group_common++;
            }
        }
        if(static_cast<uint64_t>(denominator) * group_common <
           static_cast<uint64_t>(numerator) * query_set.size()) {
            continue;
        }
        for(size_t row = 0; row < grouped_database[group_id].size(); row++) {
            const auto &candidate = grouped_database[group_id][row];
            const uint32_t intersection = les3_intersection_size(query_set, candidate);
            const uint32_t union_size = static_cast<uint32_t>(query_set.size() + candidate.size() - intersection);
            checked++;
            if(static_cast<uint64_t>(denominator) * intersection >=
               static_cast<uint64_t>(numerator) * union_size) {
                result.push_back(grouped_ids[group_id][row]);
            }
        }
    }
    if(checked_count != nullptr) {
        *checked_count = checked;
    }
    return result;
}

void LES3::insertExact(const multiset<int> &set_to_insert, int64_t stable_id) {
    if(num_groups == 0) {
        throw runtime_error("LES3 cannot insert without a constructed group");
    }
    multiset<int> mutable_set(set_to_insert);
    const int group_id = tgm.insert(mutable_set);
    grouped_database[group_id].push_back(mutable_set);
    grouped_ids[group_id].push_back(stable_id);
    database.push_back(mutable_set);
    database_ids.push_back(stable_id);
    groups[group_id].push_back(total_num_sets);
    total_num_sets++;
}

double LES3::testKNN(int result_size) {
    float query_ratio = 1000.0 / total_num_sets;
    int query_size = 0;
    unsigned total_checked_count = 0;
   
    auto start = chrono::system_clock::now();
    for(int i = 0; i < groups.size(); i++) {
        for(auto &query_set : grouped_database[i]) {
            float rand_v = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            if(rand_v > query_ratio)
                continue;
            query_size++;
            total_checked_count += findKNN(query_set, result_size);
        }   
    }
    auto end = chrono::system_clock::now();
    double total_time = (end-start).count();
    double avg_time_milliseconds = total_time/query_size/1000;
    // float avg_checked_count = total_checked_count / query_size;
   
    // cout<<"average checked sets: "<<avg_checked_count<<endl;
    
    cout<<"average search time: "<<avg_time_milliseconds<<endl;
    return avg_time_milliseconds;
}

double LES3::testDeltaNN(float delta) {
    float query_ratio = 1000.0 / total_num_sets;
    int query_size = 0;
    auto start = chrono::system_clock::now();
    unsigned total_checked_count = 0;
    // double ub_sum = 0;
    for(int i = 0; i < groups.size(); i++) {
        for(auto &query_set : grouped_database[i]) {

            float rand_v = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            if(rand_v > query_ratio)
                continue;
            query_size++;
            total_checked_count += findDeltaNN(query_set, delta);
        }
    }
    auto end = chrono::system_clock::now();
    double total_time = (end-start).count();
    double avg_time_milliseconds = total_time/query_size/1000;
    unsigned avg_checked_count = total_checked_count / query_size;
    // double avg_ub_sum = ub_sum / query_size;
    // cout<<"average checked sets: "<<avg_checked_count<<endl;
    // cout<<"average upper bounds: "<<avg_ub_sum<<endl;
    cout<<"average search time: "<<avg_time_milliseconds<<endl;
    return avg_time_milliseconds;
}

#endif /* LES3_h */
