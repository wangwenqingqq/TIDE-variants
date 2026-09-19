// GTS index
// Created on 24-01-05

#pragma once
#include <chrono>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <math.h>
#include "file.cuh"
#include "config.cuh"

#define THREAD_NUM 512
#define CHECK(call)                                                \
	{                                                              \
		const cudaError_t error = call;                            \
		if (error != cudaSuccess)                                  \
		{                                                          \
			fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
			fprintf(stderr, "code: %d, reason: %s\n", error,       \
					cudaGetErrorString(error));                    \
			exit(1);                                               \
		}                                                          \
	}

__managed__ int TREE_ORDER = 10;
__managed__ int MAX_SIZE = 20;
__managed__ int MAX_H = 3;
__managed__ int DIS_CODE = 100;
__managed__ int INFI_DIS = 10000;

typedef struct TN
{
	int pid;
	float min_dis;
	int size;
	int lid;
	int is_leaf;
};

int *split_list;
int *pid_list;
double *dis_list;
int *split_num;
float *max_dis_d;  // max distance from pivot to farthest point in each partition
int cur_level;
int start_idx;

__global__ void getPivotDis(short *data_d, char *data_s, int *size_s, TN *node_list, int *split_list, double *dis_list,
							int *id_list, int start_idx, int *data_info, int *pid_list)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int nid = start_idx + bid;

	if (split_list[nid] == 1)
	{
		TN node = node_list[nid];
		int lid = node.lid;
		int size = node.size;
		int rid = lid + size - 1;
		__shared__ int pid[1];

		if (tid == 0)
		{
			// curandState_t state;
			// curand_init(nid, nid, 0, &state);
			// int random_id = lid + (rid - lid) * curand_uniform(&state);
			pid[0] = id_list[(lid + rid) / 2];
			pid_list[nid] = pid[0];
		}
		__syncthreads();

		for (int i = tid + lid; (i >= lid && i <= rid); i += THREAD_NUM)
		{
			double result = 0;
			if (data_info[2] == 2)
			{ // L2 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					{ float _d = data_d[id_list[i] * data_info[0] + j] - data_d[pid[0] * data_info[0] + j]; result += _d * _d; }
				}
				result = sqrtf(result);
			}
			else if (data_info[2] == 1)
			{ // L1 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					result += abs(data_d[id_list[i] * data_info[0] + j] - data_d[pid[0] * data_info[0] + j]);
				}
			}
			else if (data_info[2] == 0)
			{ // Max value
				float temp = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					temp = abs(data_d[id_list[i] * data_info[0] + j] - data_d[pid[0] * data_info[0] + j]);
					if (temp > result)
						result = temp;
				}
			}
			else if (data_info[2] == 5)
			{
				float sa1 = 0, sa2 = 0, sa3 = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					sa1 += data_d[id_list[i] * data_info[0] + j] * data_d[id_list[i] * data_info[0] + j];
					sa2 += data_d[pid[0] * data_info[0] + j] * data_d[pid[0] * data_info[0] + j];
					sa3 += data_d[id_list[i] * data_info[0] + j] * data_d[pid[0] * data_info[0] + j];
				}
				sa1 = sqrtf(sa1);
				sa2 = sqrtf(sa2);
				if (sa1 * sa2 == 0)
				{
					printf("Error!!!\n");
				}
				result = sa3 / (sa1 * sa2);
				if (result > 1)
				{
					result = 0.99999999999999999;
				}
				result = abs(acos(result) * 180 / 3.1415926);
			}
			else if (data_info[2] == 6)
			{
				int n = size_s[id_list[i]];
				int m = size_s[pid[0]];
				int table[M][M];
				if (n == 0)
					result = m;
				if (m == 0)
					result = n;
				if (n != 0 && m != 0)
				{
					for (int j = 0; j <= n; j++)
						table[j][0] = j;
					for (int k = 0; k <= m; k++)
						table[0][k] = k;
					for (int j = 1; j <= n; j++)
					{
						for (int k = 1; k <= m; k++)
						{
							int cost = (data_s[id_list[i] * M + j - 1] == data_s[pid[0] * M + k - 1]) ? 0 : 1;
							table[j][k] = 1 + min(table[j - 1][k], table[j][k - 1]);
							table[j][k] = min(table[j - 1][k - 1] + cost, table[j][k]);
						}
					}
					result = table[n][m];
				}
			}
			dis_list[i] = double(result / INFI_DIS + bid * DIS_CODE);
		}
	}

	else
	{
		TN node = node_list[nid];
		int lid = node.lid;
		int size = node.size;
		int rid = lid + size - 1;

		for (int i = tid + lid; (i >= lid && i <= rid); i += THREAD_NUM)
		{
			dis_list[i] = bid * DIS_CODE;
		}
	}
}

__global__ void nodeSplit(TN *node_list, int *split_list, double *dis_list, int *empty_list, int start_idx,
						  int *pid_list, short *data_d, int *id_list, int *data_info, char *data_s, int *size_s, float *max_dis_d)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int nid = start_idx + bid;

	if (split_list[nid] == 1)
	{
		TN node = node_list[nid];
		int lid = node.lid;
		int size = node.size;
		int rid = lid + size - 1;
		__syncthreads();

		if (tid == 0)
			split_list[nid] = 0;

		for (int i = tid; i < TREE_ORDER; i += THREAD_NUM)
		{
			int avg_size = size / TREE_ORDER;
			TN node_child;

			node_child.lid = lid + avg_size * i;
			int id_child = nid * TREE_ORDER + i + 1;
			node_child.pid = pid_list[nid];
			if (i < TREE_ORDER - 1)
				node_child.size = avg_size;
			else
				node_child.size = size - (TREE_ORDER - 1) * avg_size;
			// node_child.min_dis = (dis_list[node_child.lid] - bid * DIS_CODE) * DIS_CODE;

			float result = 0;
			if (data_info[2] == 2)
			{ // L2 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					{ float _d = data_d[id_list[node_child.lid] * data_info[0] + j] - data_d[node_child.pid * data_info[0] + j]; result += _d * _d; }
				}
				result = sqrtf(result);
			}
			else if (data_info[2] == 1)
			{ // L1 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					result += abs(data_d[id_list[node_child.lid] * data_info[0] + j] - data_d[node_child.pid * data_info[0] + j]);
				}
			}
			else if (data_info[2] == 0)
			{ // Max value
				float temp = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					temp = abs(data_d[id_list[node_child.lid] * data_info[0] + j] - data_d[node_child.pid * data_info[0] + j]);
					if (temp > result)
						result = temp;
				}
			}
			else if (data_info[2] == 5)
			{
				float sa1 = 0, sa2 = 0, sa3 = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					sa1 += data_d[id_list[node_child.lid] * data_info[0] + j] * data_d[id_list[node_child.lid] * data_info[0] + j];
					sa2 += data_d[node_child.pid * data_info[0] + j] * data_d[node_child.pid * data_info[0] + j];
					sa3 += data_d[id_list[node_child.lid] * data_info[0] + j] * data_d[node_child.pid * data_info[0] + j];
				}
				sa1 = sqrtf(sa1);
				sa2 = sqrtf(sa2);
				if (sa1 * sa2 == 0)
				{
					printf("Error!!!\n");
				}
				result = sa3 / (sa1 * sa2);
				if (result > 1)
				{
					result = 0.99999999999999999;
				}
				result = abs(acos(result) * 180 / 3.1415926);
			}
			else if (data_info[2] == 6)
			{
				int n = size_s[id_list[node_child.lid]];
				int m = size_s[node_child.pid];
				int table[M][M];
				if (n == 0)
					result = m;
				if (m == 0)
					result = n;
				if (n != 0 && m != 0)
				{
					for (int j = 0; j <= n; j++)
						table[j][0] = j;
					for (int k = 0; k <= m; k++)
						table[0][k] = k;
					for (int j = 1; j <= n; j++)
					{
						for (int k = 1; k <= m; k++)
						{
							int cost = (data_s[id_list[node_child.lid] * M + j - 1] == data_s[node_child.pid * M + k - 1]) ? 0 : 1;
							table[j][k] = 1 + min(table[j - 1][k], table[j][k - 1]);
							table[j][k] = min(table[j - 1][k - 1] + cost, table[j][k]);
						}
					}
					result = table[n][m];
				}
			}
			node_child.min_dis = result;

			// Compute max_dis: distance from farthest point in partition to pivot
			{
				int last_idx = node_child.lid + node_child.size - 1;
				float max_result = 0;
				if (data_info[2] == 2) { // L2
					for (int j = 0; j < data_info[0]; j++) {
						float _d = data_d[id_list[last_idx] * data_info[0] + j] - data_d[node_child.pid * data_info[0] + j];
						max_result += _d * _d;
					}
					max_result = sqrtf(max_result);
				} else if (data_info[2] == 1) { // L1
					for (int j = 0; j < data_info[0]; j++)
						max_result += abs(data_d[id_list[last_idx] * data_info[0] + j] - data_d[node_child.pid * data_info[0] + j]);
				} else if (data_info[2] == 0) { // Max
					float temp = 0;
					for (int j = 0; j < data_info[0]; j++) {
						temp = abs(data_d[id_list[last_idx] * data_info[0] + j] - data_d[node_child.pid * data_info[0] + j]);
						if (temp > max_result) max_result = temp;
					}
				} else if (data_info[2] == 5) { // Cosine
					float sa1 = 0, sa2 = 0, sa3 = 0;
					for (int j = 0; j < data_info[0]; j++) {
						sa1 += data_d[id_list[last_idx] * data_info[0] + j] * data_d[id_list[last_idx] * data_info[0] + j];
						sa2 += data_d[node_child.pid * data_info[0] + j] * data_d[node_child.pid * data_info[0] + j];
						sa3 += data_d[id_list[last_idx] * data_info[0] + j] * data_d[node_child.pid * data_info[0] + j];
					}
					sa1 = sqrtf(sa1); sa2 = sqrtf(sa2);
					if (sa1 * sa2 > 0) { max_result = sa3 / (sa1 * sa2); if (max_result > 1) max_result = 0.99999999f; max_result = abs(acos(max_result) * 180 / 3.1415926); }
				} else if (data_info[2] == 6) { // Edit distance
					int n = size_s[id_list[last_idx]];
					int m = size_s[node_child.pid];
					int table[M][M];
					if (n == 0) max_result = m;
					else if (m == 0) max_result = n;
					else {
						for (int j = 0; j <= n; j++) table[j][0] = j;
						for (int k = 0; k <= m; k++) table[0][k] = k;
						for (int j = 1; j <= n; j++)
							for (int k = 1; k <= m; k++) {
								int cost = (data_s[id_list[last_idx] * M + j - 1] == data_s[node_child.pid * M + k - 1]) ? 0 : 1;
								table[j][k] = 1 + min(table[j-1][k], table[j][k-1]);
								table[j][k] = min(table[j-1][k-1] + cost, table[j][k]);
							}
						max_result = table[n][m];
					}
				}
				max_dis_d[id_child] = max_result;
			}

			if (node_child.size > MAX_SIZE)
			{
				split_list[id_child] = 1;
				node_child.is_leaf = 0;
			}
			else
			{
				split_list[id_child] = 0;
				node_child.is_leaf = 1;
			}

			empty_list[id_child] = 0;
			node_list[id_child] = node_child;
		}
	}
}

__global__ void initIndexData(int *data_info, TN *node_list, int *split_list, int *empty_list, int *id_list)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < data_info[1]; idx += total_num)
	{
		id_list[idx] = idx;
	}
	if (id == 0)
	{
		node_list[0].size = data_info[1];
		node_list[0].lid = 0;
		node_list[0].pid = -1;
		node_list[0].min_dis = 0;
		node_list[0].is_leaf = 0;
		split_list[0] = 1;
		empty_list[0] = 0;
	}
}

__global__ void showRes(double *dis_list, int size, TN *node_list)
{
	int id = 500;

	printf("%lf\n", dis_list[size - 1]);
	printf("node_list.size : %d\n", node_list[id].size);
	printf("node_list.pid : %d\n", node_list[id].pid);
	printf("node_list.min_dis : %f\n", node_list[id].min_dis);
	printf("node_list.lid : %d\n", node_list[id].lid);
	printf("node_list.is_leaf : %d\n", node_list[id].is_leaf);
}

#ifndef LEAF_PAD_SLOTS
#define LEAF_PAD_SLOTS 64
#endif

void indexConstru(short *data_d, char *data_s, int *size_s, int *data_info, int *&id_list, TN *&node_list, int *&max_node_num,
				  int &tree_h, int *&empty_list)
{
	printf("Index construction...\n");

	auto computeMaxHeight = [](int order, int object_count) {
		if (order <= 1 || object_count <= 0)
		{
			return 1;
		}
		// Integer-based: find smallest h where order^(h-1) leaves * MAX_SIZE >= object_count
		int h = 1;
		long long leaf_count = 1;  // height 1 = 1 leaf (root)
		while (leaf_count * MAX_SIZE < object_count && h < 20) {
			leaf_count *= order;
			h++;
		}
		return h;
	};

	MAX_H = computeMaxHeight(TREE_ORDER, data_info[1]);

	CHECK(cudaMallocManaged((void **)&max_node_num, sizeof(int)));
	CHECK(cudaMallocManaged((void **)&split_num, sizeof(int)));
	max_node_num[0] = (pow(TREE_ORDER, MAX_H) - 1) / (TREE_ORDER - 1);
	CHECK(cudaMalloc((void **)&split_list, max_node_num[0] * sizeof(int)));
	CHECK(cudaMalloc((void **)&pid_list, max_node_num[0] * sizeof(int)));
	CHECK(cudaMalloc((void **)&dis_list, data_info[1] * sizeof(double)));
	CHECK(cudaMalloc((void **)&empty_list, max_node_num[0] * sizeof(int)));
	CHECK(cudaMalloc((void **)&id_list, data_info[1] * sizeof(int)));
	CHECK(cudaMalloc((void **)&node_list, max_node_num[0] * sizeof(TN)));
	CHECK(cudaMalloc((void **)&max_dis_d, max_node_num[0] * sizeof(float)));
	CHECK(cudaMemset(max_dis_d, 0, max_node_num[0] * sizeof(float)));
	CHECK(cudaMemset(split_list, 0, max_node_num[0] * sizeof(int)));
	CHECK(cudaMemset(empty_list, 1, max_node_num[0] * sizeof(int)));
	split_num[0] = 1;
	cur_level = 0;
	start_idx = 0;
	initIndexData<<<(data_info[1] - 1) / THREAD_NUM + 1, THREAD_NUM>>>(data_info, node_list, split_list,
																	   empty_list, id_list);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "initIndexData error: %s\n", cudaGetErrorString(cudaStatus));

	// printf("split_num: %d, max_node_num: %d\n", split_num[0], max_node_num[0]);

	while ((cur_level < MAX_H - 1) && (split_num[0] > 0))
	{
		int block_num = pow(TREE_ORDER, cur_level);
		// printf("start_idx: %d\n", start_idx);

		getPivotDis<<<block_num, THREAD_NUM>>>(data_d, data_s, size_s, node_list, split_list, dis_list, id_list,
											   start_idx, data_info, pid_list);
		cudaDeviceSynchronize();
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "getPivotDis error: %s\n", cudaGetErrorString(cudaStatus));

		thrust::sort_by_key(thrust::device, dis_list, dis_list + data_info[1], id_list);

		nodeSplit<<<block_num, THREAD_NUM>>>(node_list, split_list, dis_list, empty_list, start_idx,
											 pid_list, data_d, id_list, data_info, data_s, size_s, max_dis_d);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "getPivotDis error: %s\n", cudaGetErrorString(cudaStatus));

		start_idx += pow(TREE_ORDER, cur_level);
		cur_level++;
		split_num[0] = thrust::reduce(thrust::device, split_list, split_list + max_node_num[0], 0);
	}

	// Mark remaining non-empty childless nodes as leaf
	{
		TN *h_nl = (TN*)malloc(max_node_num[0] * sizeof(TN));
		int *h_el = (int*)malloc(max_node_num[0] * sizeof(int));
		cudaMemcpy(h_nl, node_list, max_node_num[0] * sizeof(TN), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_el, empty_list, max_node_num[0] * sizeof(int), cudaMemcpyDeviceToHost);
		int fixed = 0;
		for (int i = 0; i < max_node_num[0]; i++) {
			if (h_el[i] == 0 && h_nl[i].is_leaf == 0) {
				int has_children = 0;
				for (int cc = 0; cc < TREE_ORDER; cc++) {
					int cid = i * TREE_ORDER + cc + 1;
					if (cid < max_node_num[0] && h_el[cid] == 0) { has_children = 1; break; }
				}
				if (!has_children) { h_nl[i].is_leaf = 1; fixed++; }
			}
		}
		if (fixed > 0) {
			cudaMemcpy(node_list, h_nl, max_node_num[0] * sizeof(TN), cudaMemcpyHostToDevice);
		}
		free(h_nl); free(h_el);
	}
	

	// ========== Leaf Padding for O(log n) Incremental Insert ==========
	// Expand id_list so each leaf has LEAF_PAD_SLOTS reserved empty slots.
	// This enables direct insertion without post-hoc rearrangement.
	{
		// Step 1: Copy node_list and id_list to host
		TN *h_nodes = (TN*)malloc(max_node_num[0] * sizeof(TN));
		int *h_empty = (int*)malloc(max_node_num[0] * sizeof(int));
		int *h_ids = (int*)malloc(data_info[1] * sizeof(int));
		cudaMemcpy(h_nodes, node_list, max_node_num[0] * sizeof(TN), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_empty, empty_list, max_node_num[0] * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_ids, id_list, data_info[1] * sizeof(int), cudaMemcpyDeviceToHost);

		// Step 2: Count leaves
		int num_leaves = 0;
		for (int i = 0; i < max_node_num[0]; i++)
			if (h_empty[i] == 0 && h_nodes[i].is_leaf == 1) num_leaves++;

		if (num_leaves > 0) {
			int padded_total = data_info[1] + num_leaves * LEAF_PAD_SLOTS;

			// Step 3: Build sorted leaf list by lid
			int *leaf_ids = (int*)malloc(num_leaves * sizeof(int));
			int li = 0;
			for (int i = 0; i < max_node_num[0]; i++)
				if (h_empty[i] == 0 && h_nodes[i].is_leaf == 1) leaf_ids[li++] = i;
			// Sort by lid using qsort
			{
				TN *_sort_nodes = h_nodes; // capture for lambda
				auto cmp = [](const void *a, const void *b) -> int {
					return 0; // placeholder
				};
				// Simple insertion sort (O(n log n) not needed for ~100K)
				for (int i = 1; i < num_leaves; i++) {
					int key = leaf_ids[i];
					int key_lid = h_nodes[key].lid;
					int j = i - 1;
					while (j >= 0 && h_nodes[leaf_ids[j]].lid > key_lid) {
						leaf_ids[j+1] = leaf_ids[j];
						j--;
					}
					leaf_ids[j+1] = key;
				}
			}

			// Step 4: Build padded id_list
			int *h_padded = (int*)malloc(padded_total * sizeof(int));
			memset(h_padded, -1, padded_total * sizeof(int));
			int write_pos = 0;
			for (int i = 0; i < num_leaves; i++) {
				int nid = leaf_ids[i];
				int old_lid = h_nodes[nid].lid;
				int sz = h_nodes[nid].size;
				memcpy(h_padded + write_pos, h_ids + old_lid, sz * sizeof(int));
				h_nodes[nid].lid = write_pos;  // update lid to padded position
				write_pos += sz + LEAF_PAD_SLOTS;
			}

			// Step 5: Replace id_list with padded version on GPU
			cudaFree(id_list);
			CHECK(cudaMalloc((void**)&id_list, padded_total * sizeof(int)));
			cudaMemcpy(id_list, h_padded, padded_total * sizeof(int), cudaMemcpyHostToDevice);

			// Step 6: Write back updated node_list (with new lids)
			cudaMemcpy(node_list, h_nodes, max_node_num[0] * sizeof(TN), cudaMemcpyHostToDevice);

			free(h_padded);
			free(leaf_ids);
		}
		free(h_nodes); free(h_empty); free(h_ids);
	}

	tree_h = cur_level + 1;
	printf("Tree height: %d\n", tree_h);

	cudaFree(dis_list);
	cudaFree(split_list);
	cudaFree(split_num);
	cudaFree(pid_list);
}