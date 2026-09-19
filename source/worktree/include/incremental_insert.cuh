// ============================================================
// GTS++ Incremental Insert v10: Built-in Leaf Padding + Delete Fix
// O(log n) insert with ZERO search overhead
// v10 fix: track inserted data IDs for correct delete handling
// ============================================================

#pragma once
#include <cstdlib>
#include <cstring>

#ifndef LEAF_PAD_SLOTS
#define LEAF_PAD_SLOTS 64
#endif

static int *leaf_insert_count = nullptr;
static int leaf_insert_max_nodes = 0;
static TN *h_inc_node_list = nullptr;
static int *h_inc_empty_list = nullptr;
static float *h_inc_max_dis = nullptr;
static int h_inc_max_nodes = 0;
static int h_inc_generation = 0;
static int inc_generation = 0;

// v10: Tracking arrays
static int incr_total = 0;
static int *incr_data_ids = nullptr;
static int *incr_leaf_ids = nullptr;
static int incr_capacity = 0;

int getIncrTotal() { return incr_total; }
const int *getIncrDataIds() { return incr_data_ids; }

void initIncrementalInsert(int max_nodes, TN *node_list, int *empty_list)
{
    if (leaf_insert_count) free(leaf_insert_count);
    leaf_insert_count = (int*)calloc(max_nodes, sizeof(int));
    leaf_insert_max_nodes = max_nodes;
    if (h_inc_node_list) free(h_inc_node_list);
    if (h_inc_empty_list) free(h_inc_empty_list);
    if (h_inc_max_dis) free(h_inc_max_dis);
    h_inc_node_list = (TN*)malloc(max_nodes * sizeof(TN));
    h_inc_empty_list = (int*)malloc(max_nodes * sizeof(int));
    h_inc_max_dis = (float*)malloc(max_nodes * sizeof(float));
    cudaMemcpy(h_inc_node_list, node_list, max_nodes * sizeof(TN), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_inc_empty_list, empty_list, max_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    extern float *max_dis_d;
    if (max_dis_d) cudaMemcpy(h_inc_max_dis, max_dis_d, max_nodes * sizeof(float), cudaMemcpyDeviceToHost);
    else memset(h_inc_max_dis, 0, max_nodes * sizeof(float));
    h_inc_max_nodes = max_nodes;
    inc_generation++;
    h_inc_generation = inc_generation;
    incr_total = 0;
}

int findTargetLeaf(short *data_d, TN *node_list, int *empty_list,
                   int data_id, int tree_h, int *data_info,
                   char *data_s, int *size_s)
{
    if (h_inc_generation != inc_generation) {
        if (h_inc_node_list) free(h_inc_node_list);
        if (h_inc_empty_list) free(h_inc_empty_list);
        h_inc_node_list = (TN*)malloc(h_inc_max_nodes * sizeof(TN));
        h_inc_empty_list = (int*)malloc(h_inc_max_nodes * sizeof(int));
        cudaMemcpy(h_inc_node_list, node_list, h_inc_max_nodes * sizeof(TN), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_inc_empty_list, empty_list, h_inc_max_nodes * sizeof(int), cudaMemcpyDeviceToHost);
        h_inc_generation = inc_generation;
    }

    int current_node = 0;
    for (int level = 0; level < tree_h - 1; level++) {
        int pivot = -1;
        for (int c = 0; c < TREE_ORDER; c++) {
            int child_id = current_node * TREE_ORDER + c + 1;
            if (child_id < h_inc_max_nodes && h_inc_empty_list[child_id] == 0) {
                pivot = h_inc_node_list[child_id].pid;
                break;
            }
        }
        if (pivot < 0) break;

        float dist = 0;
        if (data_info[2] == 2) {
            for (int j = 0; j < data_info[0]; j++) {
                float diff = (float)data_d[data_id * data_info[0] + j] -
                             (float)data_d[pivot * data_info[0] + j];
                dist += diff * diff;
            }
            dist = sqrtf(dist);
        } else if (data_info[2] == 1) {
            for (int j = 0; j < data_info[0]; j++)
                dist += fabsf((float)data_d[data_id * data_info[0] + j] -
                              (float)data_d[pivot * data_info[0] + j]);
        } else if (data_info[2] == 0) {
            for (int j = 0; j < data_info[0]; j++) {
                float d = fabsf((float)data_d[data_id * data_info[0] + j] -
                                (float)data_d[pivot * data_info[0] + j]);
                if (d > dist) dist = d;
            }
        } else if (data_info[2] == 5) {
            float sa1 = 0, sa2 = 0, sa3 = 0;
            for (int j = 0; j < data_info[0]; j++) {
                sa1 += data_d[data_id * data_info[0] + j] * data_d[data_id * data_info[0] + j];
                sa2 += data_d[pivot * data_info[0] + j] * data_d[pivot * data_info[0] + j];
                sa3 += data_d[data_id * data_info[0] + j] * data_d[pivot * data_info[0] + j];
            }
            sa1 = sqrtf(sa1); sa2 = sqrtf(sa2);
            if (sa1 * sa2 > 0) {
                float cv = sa3 / (sa1 * sa2);
                if (cv > 1) cv = 0.99999999f;
                dist = fabsf(acosf(cv) * 180.0f / 3.1415926f);
            }
        } else if (data_info[2] == 6) {
            int n = size_s[data_id], m = size_s[pivot];
            if (n == 0) dist = m;
            else if (m == 0) dist = n;
            else {
                int table[110][110];
                for (int j = 0; j <= n; j++) table[j][0] = j;
                for (int k = 0; k <= m; k++) table[0][k] = k;
                for (int j = 1; j <= n; j++)
                    for (int k = 1; k <= m; k++) {
                        int cost = (data_s[data_id * M + j - 1] == data_s[pivot * M + k - 1]) ? 0 : 1;
                        table[j][k] = 1 + std::min(table[j-1][k], table[j][k-1]);
                        table[j][k] = std::min(table[j-1][k-1] + cost, table[j][k]);
                    }
                dist = table[n][m];
            }
        }

        int best_child = -1;
        float best_gap = 1e30f;
        for (int c = 0; c < TREE_ORDER; c++) {
            int child_id = current_node * TREE_ORDER + c + 1;
            if (child_id >= h_inc_max_nodes) continue;
            if (h_inc_empty_list[child_id] != 0) continue;
            float cmin = h_inc_node_list[child_id].min_dis;
            float cmax = h_inc_max_dis[child_id];
            if (dist >= cmin && dist <= cmax) { best_child = child_id; break; }
            float gap = (dist < cmin) ? (cmin - dist) : (dist - cmax);
            if (gap < best_gap) { best_gap = gap; best_child = child_id; }
        }
        if (best_child < 0) break;
        if (h_inc_node_list[best_child].is_leaf == 1) return best_child;
        current_node = best_child;
    }

    for (int c = 0; c < TREE_ORDER; c++) {
        int child_id = current_node * TREE_ORDER + c + 1;
        if (child_id >= h_inc_max_nodes) continue;
        if (h_inc_empty_list[child_id] != 0) continue;
        if (h_inc_node_list[child_id].is_leaf == 1) return child_id;
    }
    return current_node;
}

int incrementalInsert(short *data_d, TN *node_list, int *empty_list,
                      int data_id, int tree_h, int *data_info, int *id_list,
                      char *data_s, int *size_s)
{
    int leaf_id = findTargetLeaf(data_d, node_list, empty_list,
                                  data_id, tree_h, data_info, data_s, size_s);
    if (leaf_insert_count[leaf_id] >= LEAF_PAD_SLOTS) return 1;

    TN leaf = h_inc_node_list[leaf_id];
    int write_pos = leaf.lid + leaf.size;
    cudaMemcpy(id_list + write_pos, &data_id, sizeof(int), cudaMemcpyHostToDevice);

    leaf.size += 1;
    h_inc_node_list[leaf_id] = leaf;
    cudaMemcpy(&node_list[leaf_id], &leaf, sizeof(TN), cudaMemcpyHostToDevice);

    {
        extern float *max_dis_d;
        int pivot = leaf.pid;
        float dist = 0;
        if (data_info[2] == 2) {
            for (int j = 0; j < data_info[0]; j++) {
                float diff = (float)data_d[data_id * data_info[0] + j] -
                             (float)data_d[pivot * data_info[0] + j];
                dist += diff * diff;
            }
            dist = sqrtf(dist);
        } else if (data_info[2] == 1) {
            for (int j = 0; j < data_info[0]; j++)
                dist += fabsf((float)data_d[data_id * data_info[0] + j] -
                              (float)data_d[pivot * data_info[0] + j]);
        } else if (data_info[2] == 0) {
            for (int j = 0; j < data_info[0]; j++) {
                float d = fabsf((float)data_d[data_id * data_info[0] + j] -
                                (float)data_d[pivot * data_info[0] + j]);
                if (d > dist) dist = d;
            }
        }
        if (h_inc_max_dis && dist > h_inc_max_dis[leaf_id]) {
            h_inc_max_dis[leaf_id] = dist;
            if (max_dis_d)
                cudaMemcpy(&max_dis_d[leaf_id], &dist, sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    leaf_insert_count[leaf_id]++;

    // v10: Track for delete handling
    if (incr_total >= incr_capacity) {
        incr_capacity = incr_capacity == 0 ? 8192 : incr_capacity * 2;
        incr_data_ids = (int*)realloc(incr_data_ids, incr_capacity * sizeof(int));
        incr_leaf_ids = (int*)realloc(incr_leaf_ids, incr_capacity * sizeof(int));
    }
    incr_data_ids[incr_total] = data_id;
    incr_leaf_ids[incr_total] = leaf_id;
    incr_total++;
    return 0;
}

int deleteIncrementalInsert(int incr_idx, TN *node_list, int *id_list)
{
    if (incr_idx < 0 || incr_idx >= incr_total) return -1;
    int leaf_id = incr_leaf_ids[incr_idx];
    int data_id = incr_data_ids[incr_idx];

    TN leaf = h_inc_node_list[leaf_id];
    int *h_ids = (int*)malloc(leaf.size * sizeof(int));
    cudaMemcpy(h_ids, id_list + leaf.lid, leaf.size * sizeof(int), cudaMemcpyDeviceToHost);

    int found_pos = -1;
    for (int i = 0; i < leaf.size; i++) {
        if (h_ids[i] == data_id) { found_pos = i; break; }
    }
    if (found_pos < 0) {
        printf("WARNING: deleteIncrementalInsert: data_id=%d not found in leaf %d\n", data_id, leaf_id);
        free(h_ids);
        return -1;
    }

    for (int i = found_pos; i < leaf.size - 1; i++)
        h_ids[i] = h_ids[i + 1];
    h_ids[leaf.size - 1] = -1;
    cudaMemcpy(id_list + leaf.lid, h_ids, leaf.size * sizeof(int), cudaMemcpyHostToDevice);
    free(h_ids);

    leaf.size--;
    h_inc_node_list[leaf_id] = leaf;
    cudaMemcpy(&node_list[leaf_id], &leaf, sizeof(TN), cudaMemcpyHostToDevice);
    leaf_insert_count[leaf_id]--;

    for (int j = incr_idx; j < incr_total - 1; j++) {
        incr_data_ids[j] = incr_data_ids[j + 1];
        incr_leaf_ids[j] = incr_leaf_ids[j + 1];
    }
    incr_total--;
    return 0;
}

void freeIncrementalInsert()
{
    if (leaf_insert_count) { free(leaf_insert_count); leaf_insert_count = nullptr; }
    if (h_inc_max_dis) { free(h_inc_max_dis); h_inc_max_dis = nullptr; }
    if (incr_data_ids) { free(incr_data_ids); incr_data_ids = nullptr; }
    if (incr_leaf_ids) { free(incr_leaf_ids); incr_leaf_ids = nullptr; }
    incr_total = 0;
    incr_capacity = 0;
}
