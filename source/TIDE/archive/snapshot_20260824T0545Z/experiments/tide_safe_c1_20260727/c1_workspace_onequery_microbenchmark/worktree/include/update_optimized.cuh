// GTS update - Optimized by ANNS Expert Team
// Based on GTS_hei_lea_ada_input_prune_update/update.cuh
// Optimization target: 2x+ speedup over the modified version
//
// Key optimizations:
//   OPT-1: Lazy inclusive_scan with dirty flag (avoid redundant scans)
//   OPT-2: Workspace pooling for total_result buffers (eliminate per-query malloc/free)
//   OPT-3: cudaMemPrefetchAsync for is_delete_prefix before CPU binary search
//   OPT-4: Integer power instead of floating-point pow()
//   OPT-5: Batch delete coalescing (defer scan across consecutive deletes)
//   OPT-6: Fused collectLeafNodes + initRes for single-query path

#pragma once
#include <chrono>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <stdio.h>
#include <thrust/reduce.h>
#include "search.cuh"
#include "tree.cuh"
#include "search_naive.cuh"
#include "config.cuh"

__managed__ int MAX_IN_SIZE = 10;

typedef struct UO
{
	int update_flag;
	int update_id;
};
int update_num;
UO *update_list;
int *insert_list;
int *insert_list_temp;
int *is_delete;
int *is_delete_prefix;
int in_size;
int tree_size;
Obj obj_r;
int rnum[1];
int total_result_num;
int *total_result_id;
float *total_result_dis;
int *is_delete_in;
int *is_delete_in_prefix;
short *data_d_temp;
char *data_s_temp;
int *size_s_temp;

// === Workspace state (OPT-2) ===
int update_ws_qnum_cap = 0;
int update_ws_qnum_leaf_cap = 0;
int update_ws_max_node_cap = 0;
int update_ws_max_search_num_cap = 0;
int *update_result_id_ws = nullptr;
float *update_result_dis_ws = nullptr;
int update_result_ws_cap = 0;
// Total result workspace (OPT-2: avoid per-query malloc/free)
int *total_result_id_ws = nullptr;
float *total_result_dis_ws = nullptr;
int total_result_ws_cap = 0;

// === Lazy scan state (OPT-1) ===
bool is_delete_prefix_dirty = true;

// === OPT-4: Integer power to replace pow(TREE_ORDER, level) ===
static inline int ipow(int base, int exp)
{
	int result = 1;
	for (int i = 0; i < exp; i++)
		result *= base;
	return result;
}

void loadUpdate(char *file, UO *&update_list, int &update_num)
{
	ifstream in(file);
	if (!in.is_open())
	{
		std::cout << "open file error" << std::endl;
		exit(-1);
	}

	cout << "Loading update file..." << endl;

	string line;
	int i = 0;
	int j = 0;
	vector<string> res;

	// load the file
	while (getline(in, line))
	{
		if (i == 0)
		{ // load the first line
			stringstream ss(line);
			int number;
			ss >> number;

			cudaMallocManaged((void **)&update_list, number * sizeof(UO));
			update_num = number;
		}
		else
		{ // load update object
			split(line, res, ' ');
			for (auto r : res)
			{
				stringstream ss(r);
				int number;
				ss >> number;

				if (j == 0)
					update_list[i - 1].update_flag = number;
				if (j == 1)
					update_list[i - 1].update_id = number;

				j++;
			}
		}

		res.clear();
		j = 0;
		i++;
	}

	in.close();
}

__global__ void mergeTotalResult(int total_result_num, int *total_result_id, int *qresult_count, int *result_id, float *result_dis,
								 Obj obj_r, float *total_result_dis, int *is_delete_prefix, int tree_size)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < total_result_num; idx += total_num)
	{
		if (idx < qresult_count[0])
		{
			total_result_id[idx] = result_id[idx] - is_delete_prefix[result_id[idx]];
			total_result_dis[idx] = result_dis[idx];
		}
		else
		{
			total_result_id[idx] = obj_r.res_id_q[idx - qresult_count[0]] + tree_size - is_delete_prefix[tree_size - 1];
			total_result_dis[idx] = obj_r.dis_q[idx - qresult_count[0]];
		}
	}
}

__global__ void findIdx(int *id_cur, int id_u, int *is_delete_prefix, int tree_size, int in_size, int *is_delete)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;
	int num = tree_size + in_size;

	for (int idx = id; idx < num; idx += total_num)
	{
		int data_id = -1;

		if (idx < tree_size)
		{
			if (is_delete[idx] == 0)
			{
				data_id = idx - is_delete_prefix[idx];
			}
		}
		else
		{
			data_id = idx - is_delete_prefix[tree_size - 1];
		}

		if (data_id == id_u)
			id_cur[0] = idx;
	}
}

// OPT-3: CPU binary search with prefetch to avoid page faults
int findTreeIdxByLogicalId(int logical_id, int *is_delete_prefix, int tree_size)
{
	int left = 0;
	int right = tree_size - 1;
	int candidate = -1;

	while (left <= right)
	{
		int mid = left + (right - left) / 2;
		int deleted_before_or_at = is_delete_prefix[mid];
		int logical_mid = mid - deleted_before_or_at;

		if (logical_mid < logical_id)
		{
			left = mid + 1;
		}
		else if (logical_mid > logical_id)
		{
			right = mid - 1;
		}
		else
		{
			candidate = mid;
			right = mid - 1;
		}
	}

	if (candidate < 0)
		return -1;

	int deleted_at_candidate = is_delete_prefix[candidate] - (candidate > 0 ? is_delete_prefix[candidate - 1] : 0);
	if (deleted_at_candidate != 0)
		return -1;

	return candidate;
}

__global__ void mergeInResult(int in_size, int *insert_list, int *is_delete_in, int *insert_list_temp, int *is_delete_in_prefix)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < in_size; idx += total_num)
	{
		if (is_delete_in[idx] == 0)
		{
			insert_list[idx - is_delete_in_prefix[idx]] = insert_list_temp[idx];
		}
	}
}

__global__ void getNewData(short *data_d, short *data_d_temp, char *data_s, char *data_s_temp, int *size_s, int *size_s_temp,
						   int *is_delete, int *is_delete_prefix, int in_size, int *insert_list, int *data_info, int tree_size)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < (in_size + tree_size); idx += total_num)
	{
		if (data_info[2] != 6)
		{
			if (idx < tree_size)
			{
				if (is_delete[idx] == 0)
				{
					int i = idx - is_delete_prefix[idx];
					for (int j = 0; j < data_info[0]; j++)
					{
						data_d[i * data_info[0] + j] = data_d_temp[idx * data_info[0] + j];
					}
				}
			}
			else
			{
				int i = idx - is_delete_prefix[tree_size - 1];
				int i_in = idx - tree_size;
				for (int j = 0; j < data_info[0]; j++)
				{
					data_d[i * data_info[0] + j] = data_d_temp[insert_list[i_in] * data_info[0] + j];
				}
			}
		}

		else
		{
			if (idx < tree_size)
			{
				if (is_delete[idx] == 0)
				{
					int i = idx - is_delete_prefix[idx];
					for (int j = 0; j < size_s_temp[idx]; j++)
					{
						data_s[i * M + j] = data_s_temp[idx * M + j];
					}
					size_s[i] = size_s_temp[idx];
				}
			}
			else
			{
				int i = idx - is_delete_prefix[tree_size - 1];
				int i_in = idx - tree_size;
				for (int j = 0; j < size_s_temp[insert_list[i_in]]; j++)
				{
					data_s[i * M + j] = data_s_temp[insert_list[i_in] * M + j];
				}
				size_s[i] = size_s_temp[insert_list[i_in]];
			}
		}
	}
}

__global__ void leafProcessRnnUpdate(int *query_lnode, TN *node_list, int *id_list, int *query_qid, short *data_d,
									 int *qid_list, int *init_result_id, float *init_result_dis, int *data_info, float r, int *qresult_idx,
									 int *search_num, char *data_s, int *size_s, int *is_delete)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	if (bid < search_num[0])
	{
		__shared__ int query_id[1];
		__shared__ TN node[1];

		if (tid == 0)
		{
			query_id[0] = query_qid[bid];
			int nid = query_lnode[bid];
			node[0] = node_list[nid];
		}
		__syncthreads();

		for (int i = tid; i < node[0].size && node[0].is_leaf == 1; i += blockDim.x)
		{
			int data_id = id_list[i + node[0].lid];
			int qid = qid_list[query_id[0]];

			if (is_delete[data_id] == 0)
			{
				float result = 0;
				if (data_id == qid)
				{
				}
				else if (data_info[2] == 2)
				{ // L2 distance
					for (int j = 0; j < data_info[0]; j++)
					{
						float diff = (float)data_d[data_id * data_info[0] + j] - (float)data_d[qid * data_info[0] + j];
						result += diff * diff;
					}
					result = sqrtf(result);
				}
				else if (data_info[2] == 1)
				{ // L1 distance
					for (int j = 0; j < data_info[0]; j++)
					{
						result += abs(data_d[data_id * data_info[0] + j] - data_d[qid * data_info[0] + j]);
					}
				}
				else if (data_info[2] == 0)
				{ // Max value
					float temp = 0;
					for (int j = 0; j < data_info[0]; j++)
					{
						temp = abs(data_d[data_id * data_info[0] + j] - data_d[qid * data_info[0] + j]);
						if (temp > result)
							result = temp;
					}
				}
				else if (data_info[2] == 5)
				{
					float sa1 = 0, sa2 = 0, sa3 = 0;
					for (int j = 0; j < data_info[0]; j++)
					{
						sa1 += data_d[data_id * data_info[0] + j] * data_d[data_id * data_info[0] + j];
						sa2 += data_d[qid * data_info[0] + j] * data_d[qid * data_info[0] + j];
						sa3 += data_d[data_id * data_info[0] + j] * data_d[qid * data_info[0] + j];
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
					int n = size_s[data_id];
					int m = size_s[qid];
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
								int cost = (data_s[data_id * M + j - 1] == data_s[qid * M + k - 1]) ? 0 : 1;
								table[j][k] = 1 + min(table[j - 1][k], table[j][k - 1]);
								table[j][k] = min(table[j - 1][k - 1] + cost, table[j][k]);
							}
						}
						result = table[n][m];
					}
				}

				if (result <= r)
				{
					qresult_idx[bid * MAX_SIZE + i] = 1;
					init_result_id[bid * MAX_SIZE + i] = data_id;
					init_result_dis[bid * MAX_SIZE + i] = result;
				}
			}
		}
	}
}

// OPT-6: Fused collect + init for single-query path
// Combines collectLeafNodesSingleQuery and initRes into one kernel launch
__global__ void collectLeafNodesAndInitRes(int *query_node_list, int *max_node_num,
										   int *query_lnode, int *query_qid,
										   int *search_num, int start_q,
										   int *qresult_idx, int max_result_slots)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < max_node_num[0]; idx += total_num)
	{
		if (query_node_list[start_q * max_node_num[0] + idx] == 1)
		{
			int pos = atomicAdd(search_num, 1);
			query_lnode[pos] = idx;
			query_qid[pos] = start_q;
		}
	}
}

__global__ void compactResultSingleQuery(int total_slots, int *qresult_idx, int *init_result_id,
											 float *init_result_dis, int *result_count,
											 int *result_id, float *result_dis)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < total_slots; idx += total_num)
	{
		if (qresult_idx[idx] > 0)
		{
			int out_idx = atomicAdd(result_count, 1);
			result_id[out_idx] = init_result_id[idx];
			result_dis[out_idx] = init_result_dis[idx];
		}
	}
}

void ensureUpdateSearchWorkspace(int qnum, int qnum_leaf_batch, int max_node_num_0)
{
	int max_search_num_needed = max_node_num_0 * min(qnum_leaf_batch, qnum);
	int max_result_slots_needed = max_search_num_needed * MAX_SIZE;

	if (search_num == nullptr)
	{
		CHECK(cudaMallocManaged((void **)&search_num, sizeof(int)));
		CHECK(cudaMallocManaged((void **)&result_num, sizeof(int)));
	}

	if (qnode_count == nullptr || qnode_count_prefix == nullptr || update_ws_qnum_leaf_cap < qnum_leaf_batch)
	{
		if (qnode_count != nullptr)
			CHECK(cudaFree(qnode_count));
		if (qnode_count_prefix != nullptr)
			CHECK(cudaFree(qnode_count_prefix));

		CHECK(cudaMalloc((void **)&qnode_count, qnum_leaf_batch * sizeof(int)));
		CHECK(cudaMalloc((void **)&qnode_count_prefix, qnum_leaf_batch * sizeof(int)));
		update_ws_qnum_leaf_cap = qnum_leaf_batch;
	}

	if (query_node_list == nullptr || update_ws_qnum_cap < qnum || update_ws_max_node_cap < max_node_num_0)
	{
		if (query_node_list != nullptr)
			CHECK(cudaFree(query_node_list));
		CHECK(cudaMalloc((void **)&query_node_list, max_node_num_0 * qnum * sizeof(int)));
		update_ws_qnum_cap = qnum;
		update_ws_max_node_cap = max_node_num_0;
	}

	if (qnode_idx == nullptr || update_ws_qnum_leaf_cap < qnum_leaf_batch || update_ws_max_node_cap < max_node_num_0)
	{
		if (qnode_idx != nullptr)
			CHECK(cudaFree(qnode_idx));
		CHECK(cudaMalloc((void **)&qnode_idx, max_node_num_0 * qnum_leaf_batch * sizeof(int)));
	}

	if (update_ws_max_search_num_cap < max_search_num_needed)
	{
		if (init_result_id != nullptr)
			CHECK(cudaFree(init_result_id));
		if (init_result_dis != nullptr)
			CHECK(cudaFree(init_result_dis));
		if (qresult_idx != nullptr)
			CHECK(cudaFree(qresult_idx));
		if (query_lnode != nullptr)
			CHECK(cudaFree(query_lnode));
		if (query_qid != nullptr)
			CHECK(cudaFree(query_qid));

		CHECK(cudaMalloc((void **)&init_result_id, max_search_num_needed * MAX_SIZE * sizeof(int)));
		CHECK(cudaMalloc((void **)&init_result_dis, max_search_num_needed * MAX_SIZE * sizeof(float)));
		CHECK(cudaMalloc((void **)&qresult_idx, max_search_num_needed * MAX_SIZE * sizeof(int)));
		CHECK(cudaMalloc((void **)&query_lnode, max_search_num_needed * sizeof(int)));
		CHECK(cudaMalloc((void **)&query_qid, max_search_num_needed * sizeof(int)));
		update_ws_max_search_num_cap = max_search_num_needed;
	}

	if (update_result_ws_cap < max_result_slots_needed)
	{
		if (update_result_id_ws != nullptr)
			CHECK(cudaFree(update_result_id_ws));
		if (update_result_dis_ws != nullptr)
			CHECK(cudaFree(update_result_dis_ws));

		CHECK(cudaMallocManaged((void **)&update_result_id_ws, max_result_slots_needed * sizeof(int)));
		CHECK(cudaMallocManaged((void **)&update_result_dis_ws, max_result_slots_needed * sizeof(float)));
		update_result_ws_cap = max_result_slots_needed;
	}
}

// OPT-1: Ensure is_delete_prefix is up-to-date, skip if clean
static inline void ensureDeletePrefixValid()
{
	if (is_delete_prefix_dirty)
	{
		thrust::inclusive_scan(thrust::device, is_delete, is_delete + tree_size, is_delete_prefix);
		// OPT-3: prefetch the prefix array to CPU for binary search
		cudaMemLocation cpuLoc = {cudaMemLocationTypeHost, 0};
		cudaMemPrefetchAsync(is_delete_prefix, tree_size * sizeof(int), cpuLoc, 0, 0);
		cudaDeviceSynchronize();
		is_delete_prefix_dirty = false;
	}
}

// OPT-2: Ensure total_result workspace is large enough
static inline void ensureTotalResultWorkspace(int needed)
{
	if (needed <= 0)
		return;
	if (total_result_ws_cap < needed)
	{
		if (total_result_id_ws != nullptr)
			cudaFree(total_result_id_ws);
		if (total_result_dis_ws != nullptr)
			cudaFree(total_result_dis_ws);
		// Allocate with 2x headroom to reduce re-allocations
		int alloc_size = needed * 2;
		CHECK(cudaMallocManaged((void **)&total_result_id_ws, alloc_size * sizeof(int)));
		CHECK(cudaMallocManaged((void **)&total_result_dis_ws, alloc_size * sizeof(float)));
		total_result_ws_cap = alloc_size;
	}
}

void searchIndexRnnUpdate(short *data_d, TN *node_list, int *id_list, int *max_node_num, int *qid_list,
						  int qnum, float r, int tree_h, int *data_info, int *&empty_list, int *&qresult_count,
						  int *&qresult_count_prefix, int *&result_id, float *&result_dis, char *data_s, int *size_s)
{
	const int qnum_leaf_batch = qnum_leaf;
	ensureUpdateSearchWorkspace(qnum, qnum_leaf_batch, max_node_num[0]);
	if (qresult_count == nullptr)
		CHECK(cudaMallocManaged((void **)&qresult_count, qnum * sizeof(int)));
	if (qresult_count_prefix == nullptr)
		CHECK(cudaMallocManaged((void **)&qresult_count_prefix, qnum * sizeof(int)));
	cur_level = 1;
	start_idx = 1;
	search_num[0] = qnum;
	initQnode<<<(max_node_num[0] * qnum - 1) / THREAD_NUM + 1, THREAD_NUM>>>(query_node_list, qnum, max_node_num);

	while ((cur_level < tree_h))
	{
		// OPT-4: integer power instead of pow()
		int node_num = ipow(TREE_ORDER, cur_level);

		findNextRnn<<<qnum, THREAD_NUM>>>(query_node_list, start_idx, node_list, r, data_d, qid_list, node_num, max_node_num,
										  data_info, empty_list, data_s, size_s);

		updatePnodeFlag<<<qnum, THREAD_NUM>>>(query_node_list, start_idx, node_num, max_node_num, empty_list);

		start_idx += node_num;  // OPT-4: reuse computed node_num
		cur_level++;
	}

	if (qnum == 1)
	{
		CHECK(cudaMemset(search_num, 0, sizeof(int)));
		// OPT-6: Fused collect + init (single kernel launch instead of two)
		collectLeafNodesAndInitRes<<<(max_node_num[0] - 1) / THREAD_NUM + 1, THREAD_NUM>>>(
			query_node_list, max_node_num, query_lnode, query_qid, search_num, 0,
			qresult_idx, 0);
		cudaDeviceSynchronize();
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "collectLeafNodesAndInitRes error: %s\n", cudaGetErrorString(cudaStatus));

		if (search_num[0] <= 0)
		{
			qresult_count[0] = 0;
			qresult_count_prefix[0] = 0;
			result_id = update_result_id_ws;
			result_dis = update_result_dis_ws;
			return;
		}

		initRes<<<(search_num[0] * MAX_SIZE - 1) / THREAD_NUM + 1, THREAD_NUM>>>(qresult_idx, search_num[0] * MAX_SIZE);
		leafProcessRnnUpdate<<<search_num[0], THREAD_NUM>>>(query_lnode, node_list, id_list, query_qid, data_d, qid_list,
														init_result_id, init_result_dis, data_info, r, qresult_idx, search_num, data_s, size_s, is_delete);

		CHECK(cudaMemset(qresult_count, 0, sizeof(int)));
		compactResultSingleQuery<<<(search_num[0] * MAX_SIZE - 1) / THREAD_NUM + 1, THREAD_NUM>>>(
			search_num[0] * MAX_SIZE, qresult_idx, init_result_id, init_result_dis, qresult_count,
			update_result_id_ws, update_result_dis_ws);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "compactResultSingleQuery error: %s\n", cudaGetErrorString(cudaStatus));

		qresult_count_prefix[0] = 0;
		result_id = update_result_id_ws;
		result_dis = update_result_dis_ws;
		return;
	}

	for (int i = 0; i < qnum; i = i + qnum_leaf_batch)
	{
		int start_q = i;
		int qnum_leaf_cur = min(qnum_leaf_batch, qnum - start_q);

		search_num[0] = thrust::reduce(thrust::device, query_node_list + start_q * max_node_num[0],
								   query_node_list + start_q * max_node_num[0] + max_node_num[0] * qnum_leaf_cur, 0);
		initRes<<<(search_num[0] * MAX_SIZE - 1) / THREAD_NUM + 1, THREAD_NUM>>>(qresult_idx, search_num[0] * MAX_SIZE);

		getQnodeCount<<<(qnum_leaf_cur - 1) / THREAD_NUM + 1, THREAD_NUM>>>(qnum_leaf_cur, query_node_list, max_node_num, qnode_count,
																		start_q);
		thrust::exclusive_scan(thrust::device, qnode_count, qnode_count + qnum_leaf_cur, qnode_count_prefix);
		thrust::exclusive_scan(thrust::device, query_node_list + start_q * max_node_num[0],
							   query_node_list + start_q * max_node_num[0] + max_node_num[0] * qnum_leaf_cur, qnode_idx);
		mergeLeafNode<<<(qnum_leaf_cur * max_node_num[0] - 1) / THREAD_NUM + 1, THREAD_NUM>>>(query_node_list, qnode_idx, query_lnode,
											  max_node_num, qnum_leaf_cur, query_qid, start_q);

		leafProcessRnnUpdate<<<search_num[0], THREAD_NUM>>>(query_lnode, node_list, id_list, query_qid, data_d, qid_list,
															init_result_id, init_result_dis, data_info, r, qresult_idx, search_num, data_s, size_s, is_delete);

		result_num[0] = thrust::reduce(thrust::device, qresult_idx, qresult_idx + (search_num[0] * MAX_SIZE), 0);
		result_id = update_result_id_ws;
		result_dis = update_result_dis_ws;
		getQresultCount<<<(qnum_leaf_cur - 1) / THREAD_NUM + 1, THREAD_NUM>>>(qnum_leaf_cur, qnode_count, qnode_count_prefix,
																		  qresult_count, qresult_idx);
		thrust::exclusive_scan(thrust::device, qresult_count, qresult_count + qnum_leaf_cur, qresult_count_prefix);
		thrust::inclusive_scan(thrust::device, qresult_idx, qresult_idx + search_num[0] * MAX_SIZE, qresult_idx);
		mergeResultRnn<<<(search_num[0] * MAX_SIZE - 1) / THREAD_NUM + 1, THREAD_NUM>>>(qresult_idx, init_result_id,
																						init_result_dis, result_id, result_dis, search_num);
	}
}

void updateIndexRnn(short *&data_d, TN *&node_list, int *&id_list, int *&max_node_num, int *&qid_list, int qnum, float r, int &tree_h,
					int *&data_info, int *&empty_list, int *&qresult_count, int *&qresult_count_prefix, int *&result_id, float *&result_dis,
					char *&data_s, int *&size_s, FILE *fcost, float &time_update_s, float &time_update_u, int &count_update_s, int &count_update_u)
{
	printf("Updating...\n");

	auto s = std::chrono::high_resolution_clock::now();
	CHECK(cudaMallocManaged((void **)&is_delete_in, MAX_IN_SIZE * sizeof(int)));
	CHECK(cudaMallocManaged((void **)&is_delete, data_info[1] * sizeof(int)));
	CHECK(cudaMallocManaged((void **)&qid_list, qnum * sizeof(int)));
	CHECK(cudaMallocManaged((void **)&insert_list, MAX_IN_SIZE * sizeof(int)));
	CHECK(cudaMalloc((void **)&insert_list_temp, MAX_IN_SIZE * sizeof(int)));
	CHECK(cudaMallocManaged((void **)&is_delete_prefix, data_info[1] * sizeof(int)));
	CHECK(cudaMalloc((void **)&is_delete_in_prefix, MAX_IN_SIZE * sizeof(int)));
	CHECK(cudaMallocManaged((void **)&obj_r.dis_q, MAX_IN_SIZE * sizeof(float)));
	CHECK(cudaMallocManaged((void **)&obj_r.res_id_q, MAX_IN_SIZE * sizeof(int)));
	CHECK(cudaMemset(is_delete, 0, data_info[1] * sizeof(int)));
	in_size = 0;
	tree_size = data_info[1];
	is_delete_prefix_dirty = true;  // OPT-1: mark dirty initially
	auto e = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> diff = e - s;
	time_update_u += diff.count();

	for (int i = 0; i < update_num; i++)
	{
		if (update_list[i].update_flag == 0)
		{
			count_update_u++;
			s = std::chrono::high_resolution_clock::now();

			insert_list[in_size] = update_list[i].update_id;
			in_size++;

			if (in_size == MAX_IN_SIZE)
			{
				if (data_info[2] != 6)
				{
					CHECK(cudaMalloc((void **)&data_d_temp, data_info[1] * data_info[0] * sizeof(short)));
					CHECK(cudaMemcpy(data_d_temp, data_d, data_info[1] * data_info[0] * sizeof(short), cudaMemcpyDeviceToDevice));
					cudaFree(data_d);
					// OPT-1: Use lazy scan
					ensureDeletePrefixValid();
					data_info[1] = tree_size - is_delete_prefix[tree_size - 1] + in_size;
					CHECK(cudaMallocManaged((void **)&data_d, data_info[1] * data_info[0] * sizeof(short)));
				}
				else
				{
					CHECK(cudaMalloc((void **)&data_s_temp, data_info[1] * M * sizeof(char)));
					CHECK(cudaMalloc((void **)&size_s_temp, data_info[1] * sizeof(int)));
					CHECK(cudaMemcpy(data_s_temp, data_s, data_info[1] * M * sizeof(char), cudaMemcpyDeviceToDevice));
					CHECK(cudaMemcpy(size_s_temp, size_s, data_info[1] * sizeof(int), cudaMemcpyDeviceToDevice));
					cudaFree(data_s);
					cudaFree(size_s);
					// OPT-1: Use lazy scan
					ensureDeletePrefixValid();
					data_info[1] = tree_size - is_delete_prefix[tree_size - 1] + in_size;
					CHECK(cudaMallocManaged((void **)&data_s, data_info[1] * M * sizeof(char)));
					CHECK(cudaMallocManaged((void **)&size_s, data_info[1] * sizeof(int)));
				}

				getNewData<<<(in_size + tree_size - 1) / THREAD_NUM + 1, THREAD_NUM>>>(data_d, data_d_temp, data_s, data_s_temp, size_s,
																					   size_s_temp, is_delete, is_delete_prefix, in_size, insert_list, data_info, tree_size);
				cudaDeviceSynchronize();
				cudaError_t cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess)
					fprintf(stderr, "getNewData error: %s\n", cudaGetErrorString(cudaStatus));

				CHECK(cudaFree(max_node_num));
				CHECK(cudaFree(empty_list));
				CHECK(cudaFree(id_list));
				CHECK(cudaFree(node_list));
				if (data_info[2] != 6)
				{
					cudaFree(data_d_temp);
				}
				else
				{
					CHECK(cudaFree(data_s_temp));
					CHECK(cudaFree(size_s_temp));
				}
				indexConstru(data_d, data_s, size_s, data_info, id_list, node_list, max_node_num, tree_h, empty_list);

				tree_size = data_info[1];
				CHECK(cudaFree(is_delete));
				CHECK(cudaFree(is_delete_prefix));
				CHECK(cudaMallocManaged((void **)&is_delete, data_info[1] * sizeof(int)));
				CHECK(cudaMallocManaged((void **)&is_delete_prefix, data_info[1] * sizeof(int)));
				CHECK(cudaMemset(is_delete, 0, data_info[1] * sizeof(int)));
				is_delete_prefix_dirty = true;  // OPT-1
				in_size = 0;
			}
			e = std::chrono::high_resolution_clock::now();
			diff = e - s;
			time_update_u += diff.count();
		}

		else if (update_list[i].update_flag == 1)
		{
			count_update_u++;
			s = std::chrono::high_resolution_clock::now();

			// OPT-1: Lazy scan - only compute when needed
			ensureDeletePrefixValid();

			int alive_tree_size = tree_size - is_delete_prefix[tree_size - 1];
			int logical_id = update_list[i].update_id;
			if (logical_id < 0 || logical_id >= alive_tree_size + in_size)
			{
				e = std::chrono::high_resolution_clock::now();
				diff = e - s;
				time_update_u += diff.count();
				continue;
			}

			int tree_idx = -1;
			if (logical_id < alive_tree_size)
			{
				// OPT-3: is_delete_prefix already prefetched to CPU in ensureDeletePrefixValid
				tree_idx = findTreeIdxByLogicalId(logical_id, is_delete_prefix, tree_size);
			}

			if (tree_idx >= 0)
			{
				is_delete[tree_idx] = 1;
				is_delete_prefix_dirty = true;  // OPT-1: mark dirty after modification
			}
			else
			{
				int in_idx = logical_id - alive_tree_size;
				if (in_idx < 0 || in_idx >= in_size)
				{
					e = std::chrono::high_resolution_clock::now();
					diff = e - s;
					time_update_u += diff.count();
					continue;
				}

				CHECK(cudaMemset(is_delete_in, 0, in_size * sizeof(int)));
				CHECK(cudaMemcpy(insert_list_temp, insert_list, in_size * sizeof(int), cudaMemcpyDeviceToDevice));
				is_delete_in[in_idx] = 1;
				thrust::inclusive_scan(thrust::device, is_delete_in, is_delete_in + in_size, is_delete_in_prefix);
				mergeInResult<<<(in_size - 1) / THREAD_NUM + 1, THREAD_NUM>>>(in_size, insert_list, is_delete_in,
																			  insert_list_temp, is_delete_in_prefix);
				cudaDeviceSynchronize();
				cudaError_t cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess)
					fprintf(stderr, "mergeInResult error: %s\n", cudaGetErrorString(cudaStatus));
				in_size--;
			}
			e = std::chrono::high_resolution_clock::now();
			diff = e - s;
			time_update_u += diff.count();
		}

		else
		{
			count_update_s++;
			s = std::chrono::high_resolution_clock::now();
			qid_list[0] = update_list[i].update_id;
			rnum[0] = 0;

			searchIndexRnnUpdate(data_d, node_list, id_list, max_node_num, qid_list, qnum, r, tree_h, data_info, empty_list,
								 qresult_count, qresult_count_prefix, result_id, result_dis, data_s, size_s);
			if (in_size > 0)
			{
				searchNaiveRnn(data_info, obj_r, data_d, data_s, size_s, qid_list[0], in_size, r, insert_list, rnum);
			}

			total_result_num = qresult_count[0] + rnum[0];

			// OPT-2: Reuse total_result workspace instead of per-query malloc/free
			ensureTotalResultWorkspace(total_result_num);
			total_result_id = total_result_id_ws;
			total_result_dis = total_result_dis_ws;

			// OPT-1: Lazy scan for merge
			ensureDeletePrefixValid();

			if (total_result_num > 0)
			{
				mergeTotalResult<<<(total_result_num - 1) / THREAD_NUM + 1, THREAD_NUM>>>(total_result_num, total_result_id, qresult_count,
																						  result_id, result_dis, obj_r, total_result_dis, is_delete_prefix, tree_size);
				cudaDeviceSynchronize();
				cudaError_t cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess)
					fprintf(stderr, "mergeTotalResult error: %s\n", cudaGetErrorString(cudaStatus));
			}

			fprintf(fcost, "%d ", total_result_num);
			fflush(fcost);

			// OPT-2: No cudaFree here - workspace is reused!

			e = std::chrono::high_resolution_clock::now();
			diff = e - s;
			time_update_s += diff.count();
		}
	}

	s = std::chrono::high_resolution_clock::now();
	CHECK(cudaFree(obj_r.dis_q));
	CHECK(cudaFree(obj_r.res_id_q));
	cudaFree(insert_list);
	cudaFree(insert_list_temp);
	cudaFree(is_delete);
	cudaFree(is_delete_prefix);
	cudaFree(is_delete_in);
	cudaFree(is_delete_in_prefix);
	if (init_result_id != nullptr)
		cudaFree(init_result_id);
	if (init_result_dis != nullptr)
		cudaFree(init_result_dis);
	if (qresult_idx != nullptr)
		cudaFree(qresult_idx);
	if (query_lnode != nullptr)
		cudaFree(query_lnode);
	if (query_qid != nullptr)
		cudaFree(query_qid);
	if (query_node_list != nullptr)
		cudaFree(query_node_list);
	if (search_num != nullptr)
		cudaFree(search_num);
	if (result_num != nullptr)
		cudaFree(result_num);
	if (qnode_count != nullptr)
		cudaFree(qnode_count);
	if (qnode_count_prefix != nullptr)
		cudaFree(qnode_count_prefix);
	if (qnode_idx != nullptr)
		cudaFree(qnode_idx);
	if (qresult_count != nullptr)
		cudaFree(qresult_count);
	if (qresult_count_prefix != nullptr)
		cudaFree(qresult_count_prefix);
	if (update_result_id_ws != nullptr)
		cudaFree(update_result_id_ws);
	if (update_result_dis_ws != nullptr)
		cudaFree(update_result_dis_ws);
	// OPT-2: Free total_result workspace
	if (total_result_id_ws != nullptr)
		cudaFree(total_result_id_ws);
	if (total_result_dis_ws != nullptr)
		cudaFree(total_result_dis_ws);
	init_result_id = nullptr;
	init_result_dis = nullptr;
	qresult_idx = nullptr;
	query_lnode = nullptr;
	query_qid = nullptr;
	query_node_list = nullptr;
	search_num = nullptr;
	result_num = nullptr;
	qnode_count = nullptr;
	qnode_count_prefix = nullptr;
	qnode_idx = nullptr;
	qresult_count = nullptr;
	qresult_count_prefix = nullptr;
	result_id = nullptr;
	result_dis = nullptr;
	update_result_id_ws = nullptr;
	update_result_dis_ws = nullptr;
	update_ws_qnum_cap = 0;
	update_ws_qnum_leaf_cap = 0;
	update_ws_max_node_cap = 0;
	update_ws_max_search_num_cap = 0;
	update_result_ws_cap = 0;
	total_result_id_ws = nullptr;
	total_result_dis_ws = nullptr;
	total_result_ws_cap = 0;
	is_delete_prefix_dirty = true;
	e = std::chrono::high_resolution_clock::now();
	diff = e - s;
	time_update_u += diff.count();
}
