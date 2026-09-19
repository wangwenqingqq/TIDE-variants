// Search v2. Dynamically adjust memory allocation
// Created on 24-01-05

#pragma once
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <stdio.h>
#include <thrust/reduce.h>
#include <stack>
#include <thrust/count.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

// A traversal error must escape the copied GTS host routine.  Printing and
// continuing can otherwise fabricate a partial result/receipt after a failed
// CUDA launch.  The Safe-C1 caller catches this and enters FailedStop.  The
// transitive archive tree.cuh CHECK macro still uses exit(1) on CUDA failure;
// that is explicitly a nonzero process fail-stop, never a successful receipt.
inline void safe_c1_traversal_require_cuda_success(cudaError_t status, const char* stage) {
	if (status == cudaSuccess) return;
	throw std::runtime_error(std::string("Safe-C1 GTS traversal CUDA failure at ") +
	                         stage + ": " + cudaGetErrorString(status));
}
[[noreturn]] inline void safe_c1_traversal_fail_stop(const char* stage) {
	throw std::runtime_error(std::string("Safe-C1 GTS traversal fail-stop at ") + stage);
}

// G3 production-owned Safe-C1 traversal receipt copy. This copied header records only leaf pairs
// emitted by the actual GTS KNN traversal; G3 maps returned local rows to stable IDs.
struct SafeC1VisitedLeafPair { int query_id; int leaf_id; };
static std::vector<SafeC1VisitedLeafPair> safe_c1_last_visited_leaf_pairs;
#include "file.cuh"
#include "tree.cuh"
#include "config.cuh"
#include "mlp_constant.cuh"
#include "residual_pruning.cuh"
using namespace std;
// [新增] 用于收集训练数据的结构体
// struct TrainingSample {
//     float feature_lb;    // 特征1: 三角不等式下界 (node.min_dis - dis_q)
//     float feature_r;     // 特征2: 当前搜索半径 (disk[qid])
//     float feature_node;  // 特征3: 节点自身属性 (node.min_dis)
//     int node_id;         // 辅助: 节点ID，用于后续Python打标签
//     int query_id;        // 辅助: 查询ID
// };

// [新增] 全局托管内存指针 (在 main.cu 中分配，或者在这里偷懒定义)
// __managed__ TrainingSample* debug_logs;
// __managed__ int debug_log_count = 0;
// __managed__ int MAX_LOGS = 1000000; // 限制采集100万条，防止显存爆炸
stack<int> st;			  // Instead of recursive calls, control the query interval of each layer (qs, qe, cur_level, qnum_up, offset_n, qs_up, size_a).
int *res;				  // The output result of range query.
float *res_dis;			  // The output result of knn query.
int qs, qe;				  // The start id and the end id of query at each process.
int *p_list;			  // The total list of queries to process for all layers.
double *p_list_k;		  // The total list of queries to process for all layers for knn queries.
int size_avg;			  // The average size of remaining levels.
int *size_list;			  // The real size of each level.
int qnum_l;				  // The number of queries that can be processed simultaneously at the current layer.
int qnum_up;			  // The number of queries that can be processed simultaneously at the upper layer.
int nnum_l;				  // The number of nodes at the current level.
int size_a;				  // The available size.
int offset_p;			  // The offset of the starting position of the p_list at current layer.
int offset_up_p;		  // The offset of the starting position of the p_list at upper layer.
int offset_n;			  // The offset of the starting position of the node_list at current layer.
int qs_up;				  // The start id of query at upper layer.
float *disk;			  // The distance to the current k-th neighbor.
bool update_disk = false; // The flag to determine whether the disk has been updated.
float input_size = 0;

// Normalization factors (1 / max_value)
// extern __device__ __constant__ float input_scale[3];

// extern __device__ __constant__ float c_W1[3 * 8];
// extern __device__ __constant__ float c_b1[8];
// extern __device__ __constant__ float c_W2[8];
// extern __device__ __constant__ float c_b2;


// [Replaced] MLP inference replaced by Residual Pruning (see residual_pruning.cuh)
// New inference: rp_predict(dis_lb, disk_val, cur_level) -- 4 FLOPs vs ~40 FLOPs

// Process the nodes of the current layer and determine if the node will be pruned.
// A thread is assigned for a (query, node) pair.
__global__ void nodeProcessRnn(TN *node_list, float r, short *data_d, int *qid_list, int *data_info, int *empty_list,
							   char *data_s, int *size_s, int qnum_l, int qnum_up, int nnum_l, int *p_list, int offset_p, int offset_up_p, int offset_n,
							   int qs, int qs_up)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < nnum_l * qnum_l; i += total_num)
	{
		int qid_p = i % qnum_l;				   // Query idx in p_list
		int qid_p_up = qid_p + qs - qs_up;	   // Query idx at upper layer in p_list
		int qid = qid_p + qs;				   // Query idx in qid_lsit
		int nid_p = i / qnum_l;				   // Node idx in p_list at current level.
		int nid_parent_p = nid_p / TREE_ORDER; // Parent node idx in p_list at upper level.
		int nid = nid_p + offset_n;			   // Node idx in node_list

		// Reset p_list
		p_list[offset_p + i] = 0;

		if (p_list[offset_up_p + nid_parent_p * qnum_up + qid_p_up] == 1 && (empty_list[nid] == 0))
		{
			TN node = node_list[nid];

			float dis_q = 0;
			if (data_info[2] == 2)
			{ // L2 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					{ float _d = data_d[node.pid * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]; dis_q += _d * _d; }
				}
				dis_q = sqrtf(dis_q);
			}
			else if (data_info[2] == 1)
			{ // L1 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					dis_q += abs(data_d[node.pid * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]);
				}
			}
			else if (data_info[2] == 0)
			{ // Max value
				float temp = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					temp = abs(data_d[node.pid * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]);
					if (temp > dis_q)
						dis_q = temp;
				}
			}
			else if (data_info[2] == 5)
			{
				float sa1 = 0, sa2 = 0, sa3 = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					sa1 += data_d[node.pid * data_info[0] + j] * data_d[node.pid * data_info[0] + j];
					sa2 += data_d[qid_list[qid] * data_info[0] + j] * data_d[qid_list[qid] * data_info[0] + j];
					sa3 += data_d[node.pid * data_info[0] + j] * data_d[qid_list[qid] * data_info[0] + j];
				}
				sa1 = sqrtf(sa1);
				sa2 = sqrtf(sa2);
				if (sa1 * sa2 == 0)
				{
					printf("Error!!!\n");
				}
				dis_q = sa3 / (sa1 * sa2);
				if (dis_q > 1)
				{
					dis_q = 0.99999999999999999;
				}
				dis_q = abs(acos(dis_q) * 180 / 3.1415926);
			}
			else if (data_info[2] == 6)
			{
				int n = size_s[node.pid];
				int m = size_s[qid_list[qid]];
				int table[M][M];
				if (n == 0)
					dis_q = m;
				if (m == 0)
					dis_q = n;
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
							int cost = (data_s[node.pid * M + j - 1] == data_s[qid_list[qid] * M + k - 1]) ? 0 : 1;
							table[j][k] = 1 + min(table[j - 1][k], table[j][k - 1]);
							table[j][k] = min(table[j - 1][k - 1] + cost, table[j][k]);
						}
					}
					dis_q = table[n][m];
				}
			}

			float dis_lb = node.min_dis - dis_q;
			dis_lb = max(dis_lb, 0.0);
			if (nid % TREE_ORDER != 0)
			{
				float dis_lb2 = dis_q - node_list[nid + 1].min_dis;
				dis_lb = max(dis_lb, dis_lb2);
			}

			if (dis_lb <= r)
				p_list[offset_p + i] = 1;
		}
	}
}

// Process the nodes of the current layer and determine if the node will be pruned.
// A thread is assigned for a (query, node) pair.
// [C3 Optimized] Warp-level collaboration to reduce divergence
__global__ void nodeProcessKnn(TN *node_list, float *disk, int *empty_list, int qnum_l, int qnum_up, int nnum_l, double *p_list_k,
							   int offset_p, int offset_up_p, int offset_n, int qs, int qs_up, int cur_level)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;
	
#if C3_ENABLE_WARP_COLLAB
	// C3 Optimization: Warp-level collaboration
	const int warp_id = threadIdx.x / C3_WARP_SIZE;
	const int lane_id = threadIdx.x % C3_WARP_SIZE;
	
	for (int i = tid; i < nnum_l * qnum_l; i += total_num)
	{
		int qid_p = i % qnum_l;
		int qid_p_up = qid_p + qs - qs_up;
		int qid = qid_p + qs;
		int nid_p = (i / qnum_l);
		int nid_parent_p = nid_p / TREE_ORDER;
		int nid = nid_p + offset_n;
		int temp = (nnum_l / TREE_ORDER / TREE_ORDER * 3);
		int ofst_up = temp * qnum_up;
		int ofst = (nnum_l / TREE_ORDER * 3) * qnum_l;
		int idx_p = nid_parent_p * qnum_l + qid_p;

		// Reset p_list
		p_list_k[offset_p + ofst + i] = 0;

		// C3: Use ballot to check if any thread in warp needs processing
		int needs_process = (p_list_k[offset_up_p + ofst_up + nid_parent_p * qnum_up + qid_p_up] == 1 && (empty_list[nid] == 0)) ? 1 : 0;
		unsigned int mask = __ballot_sync(0xffffffff, needs_process);
		
		if (needs_process)
		{
			TN node = node_list[nid];

			float dis_q = p_list_k[offset_p + ofst / 3 + idx_p];

			float dis_lb = node.min_dis - dis_q;
			dis_lb = max(dis_lb, 0.0f);

			if (nid % TREE_ORDER != 0)
			{
				float dis_lb2 = dis_q - node_list[nid + 1].min_dis;
				dis_lb = max(dis_lb, dis_lb2);
			}

			// Residual Pruning
			float delta = rp_predict(dis_lb, disk[qid], cur_level);
			float dis_lb_eff = dis_lb + delta;

			// C3: Use ballot to check pruning decisions
			int should_keep = (dis_lb_eff <= disk[qid]) ? 1 : 0;
			
			if (should_keep)
			{
				p_list_k[offset_p + ofst + i] = 1;
			}
		}
	}
#else
	// Original implementation (fallback)
	for (int i = tid; i < nnum_l * qnum_l; i += total_num)
	{
		int qid_p = i % qnum_l;
		int qid_p_up = qid_p + qs - qs_up;
		int qid = qid_p + qs;
		int nid_p = (i / qnum_l);
		int nid_parent_p = nid_p / TREE_ORDER;
		int nid = nid_p + offset_n;
		int temp = (nnum_l / TREE_ORDER / TREE_ORDER * 3);
		int ofst_up = temp * qnum_up;
		int ofst = (nnum_l / TREE_ORDER * 3) * qnum_l;
		int idx_p = nid_parent_p * qnum_l + qid_p;

		p_list_k[offset_p + ofst + i] = 0;

		if (p_list_k[offset_up_p + ofst_up + nid_parent_p * qnum_up + qid_p_up] == 1 && (empty_list[nid] == 0))
		{
			TN node = node_list[nid];
			float dis_q = p_list_k[offset_p + ofst / 3 + idx_p];
			float dis_lb = node.min_dis - dis_q;
			dis_lb = max(dis_lb, 0.0f);

			if (nid % TREE_ORDER != 0)
			{
				float dis_lb2 = dis_q - node_list[nid + 1].min_dis;
				dis_lb = max(dis_lb, dis_lb2);
			}

			float delta = rp_predict(dis_lb, disk[qid], cur_level);
			float dis_lb_eff = dis_lb + delta;

			if (dis_lb_eff <= disk[qid])
			{
				p_list_k[offset_p + ofst + i] = 1;
			}
		}
	}
#endif
}

// Initialize p_list.
__global__ void initPList(int *p_list, int qnum)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < qnum; i += total_num)
	{
		p_list[i] = 1;
	}
}

// Initialize p_list.
__global__ void initPListKnn(double *p_list_k, int qnum)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < qnum; i += total_num)
	{
		p_list_k[i] = 1;
	}
}

// Get counts of query.
__global__ void getQCount(int ls, int le, int *p_list, int offset_up_p, int offset_p, int qnum_up, int nnum_up)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < qnum_up * nnum_up; i += total_num)
	{
		int qid_p = i % qnum_up; // Query idx in p_list
		int nid_p = i / qnum_up; // Node idx in p_list at current level.

		if (p_list[offset_up_p + i] == 1 && qid_p >= ls && qid_p < le)
		{
			p_list[offset_p + (qid_p - ls) * nnum_up + nid_p] = 1;
		}
		else if (p_list[offset_up_p + i] == 0 && qid_p >= ls && qid_p < le)
		{
			p_list[offset_p + (qid_p - ls) * nnum_up + nid_p] = 0;
		}
	}
}

// Get counts of query for knn.
__global__ void getQCountKnn(int ls, int le, double *p_list_k, int offset_up_p, int offset_p, int qnum_up, int nnum_up)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;
	int ofst_up = (nnum_up / TREE_ORDER * 3) * qnum_up; // Offset at upper level.

	for (int i = tid; i < qnum_up * nnum_up; i += total_num)
	{
		int qid_p = i % qnum_up; // Query idx in p_list
		int nid_p = i / qnum_up; // Node idx in p_list at current level.

		if (p_list_k[offset_up_p + ofst_up + i] == 1 && qid_p >= ls && qid_p < le)
		{
			p_list_k[offset_p + (qid_p - ls) * nnum_up + nid_p] = 1;
		}
		else if (p_list_k[offset_up_p + ofst_up + i] == 0 && qid_p >= ls && qid_p < le)
		{
			p_list_k[offset_p + (qid_p - ls) * nnum_up + nid_p] = 0;
		}
	}
}

// Merge leaf node.
// A thread is assigned for a (query, node) pair.
__global__ void mergeLNode(int ls, int le, int *p_list, int offset_up_p, int offset_up_n, int qs_up, int offset_p, int *size_list,
						   int cur_level, int qnum_up, int nnum_up)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < qnum_up * nnum_up; i += total_num)
	{
		int qid_p = i % qnum_up;	   // Query idx in p_list
		int qid = qid_p + qs_up;	   // Query idx in qid_lsit
		int nid_p = i / qnum_up;	   // Node idx in p_list at current level.
		int nid = nid_p + offset_up_n; // Node idx in node_list

		// Merge leaf node.
		if (p_list[offset_up_p + i] == 1 && qid_p >= ls && qid_p < le)
		{
			int idx_pre = p_list[offset_p + (qid_p - ls) * nnum_up + nid_p];			  // Idx in prefix sum list
			p_list[offset_p + size_list[cur_level] / (MAX_SIZE + 3) + idx_pre] = nid;	  // Merge the real node ID in plist
			p_list[offset_p + size_list[cur_level] / (MAX_SIZE + 3) * 2 + idx_pre] = qid; // Merge the real query ID in plist
		}
	}
}

// Merge leaf nodes for knn.
// A thread is assigned for a (query, node) pair.
__global__ void mergeLNodeKnn(int ls, int le, double *p_list_k, int offset_up_p, int offset_up_n, int qs_up, int offset_p, int *size_list,
							  int cur_level, int qnum_up, int nnum_up)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;
	int ofst_up = (nnum_up / TREE_ORDER * 3) * qnum_up; // Offset at upper level.

	for (int i = tid; i < qnum_up * nnum_up; i += total_num)
	{
		int qid_p = i % qnum_up;	   // Query idx in p_list
		int qid = qid_p + qs_up;	   // Query idx in qid_lsit
		int nid_p = i / qnum_up;	   // Node idx in p_list at current level.
		int nid = nid_p + offset_up_n; // Node idx in node_list

		// Merge leaf node.
		if (p_list_k[offset_up_p + ofst_up + i] == 1 && qid_p >= ls && qid_p < le)
		{
			int idx_pre = p_list_k[offset_p + (qid_p - ls) * nnum_up + nid_p];					// Idx in prefix sum list
			p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) + idx_pre] = nid;		// Merge the real node ID in plist
			p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * 2 + idx_pre] = qid; // Merge the real query ID in plist
		}
	}
}

// Process the nodes of the current layer and determine if the node will be pruned.
__global__ void dataProcessRnn(TN *node_list, float r, short *data_d, int *qid_list, int *data_info, char *data_s, int *size_s,
							   int *p_list, int offset_p, int *id_list, int cur_level, int *size_list, int nnum_up)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int nid_p = bid;																// Node idx in p_list.
	int qid_p = bid;																// Query idx in p_list.
	int did = tid;																	// Data idx in the leaf node.
	int nid = p_list[offset_p + size_list[cur_level] / (MAX_SIZE + 3) + nid_p];		// Node idx in node_list
	int qid = p_list[offset_p + size_list[cur_level] / (MAX_SIZE + 3) * 2 + qid_p]; // Query idx in qid_lsit
	TN node = node_list[nid];														// Leaf node

	if (did < node.size && node.is_leaf == 1)
	{
		int data_id = id_list[node.lid + did]; // Data idx in dataset.
		// printf("data_id: %d\n", data_id);

		float result = 0;
		if (data_id == qid_list[qid])
		{
		}
		else if (data_info[2] == 2)
		{ // L2 distance
			for (int j = 0; j < data_info[0]; j++)
			{
				{ float _d = data_d[data_id * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]; result += _d * _d; }
			}
			result = sqrtf(result);
		}
		else if (data_info[2] == 1)
		{ // L1 distance
			for (int j = 0; j < data_info[0]; j++)
			{
				result += abs(data_d[data_id * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]);
			}
		}
		else if (data_info[2] == 0)
		{ // Max value
			float temp = 0;
			for (int j = 0; j < data_info[0]; j++)
			{
				temp = abs(data_d[data_id * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]);
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
				sa2 += data_d[qid_list[qid] * data_info[0] + j] * data_d[qid_list[qid] * data_info[0] + j];
				sa3 += data_d[data_id * data_info[0] + j] * data_d[qid_list[qid] * data_info[0] + j];
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
			int m = size_s[qid_list[qid]];
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
						int cost = (data_s[data_id * M + j - 1] == data_s[qid_list[qid] * M + k - 1]) ? 0 : 1;
						table[j][k] = 1 + min(table[j - 1][k], table[j][k - 1]);
						table[j][k] = min(table[j - 1][k - 1] + cost, table[j][k]);
					}
				}
				result = table[n][m];
			}
		}

		if (result <= r)
		{
			// p_list[2 * offset_p - offset_up_p + 2 * lnum + i] = qid + 1;
			// atomicAdd(&res[qid], 1);
			p_list[offset_p + size_list[cur_level] / (MAX_SIZE + 3) * 3 + bid * MAX_SIZE + did] = 1;
		}
		else
			p_list[offset_p + size_list[cur_level] / (MAX_SIZE + 3) * 3 + bid * MAX_SIZE + did] = 0;
	}

	else if (did < MAX_SIZE)
	{
		p_list[offset_p + size_list[cur_level] / (MAX_SIZE + 3) * 3 + bid * MAX_SIZE + did] = 0;
	}
}

// Process the nodes of the current layer and determine if the node will be pruned.
__global__ void dataProcessKnn(TN *node_list, float *disk, short *data_d, int *qid_list, int *data_info, char *data_s, int *size_s,
							   double *p_list_k, int offset_p, int *id_list, int cur_level, int *size_list, int nnum_up)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int nid_p = bid;																	  // Node idx in p_list.
	int qid_p = bid;																	  // Query idx in p_list.
	int nid = p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) + nid_p];	  // Node idx in node_list
	int qid = p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * 2 + qid_p]; // Query idx in qid_lsit
	TN node = node_list[nid];															  // Leaf node

	for (int did = tid; did < MAX_SIZE; did += THREAD_NUM)
	{
		double result = INFI_DIS;
		int data_id = -1;

		if (did < node.size && node.is_leaf == 1)
		{
			data_id = id_list[node.lid + did]; // Data idx in dataset.

			result = 0;
			if (data_id == qid_list[qid])
			{
			}
			else if (data_info[2] == 2)
			{ // L2 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					{ float _d = data_d[data_id * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]; result += _d * _d; }
				}
				result = sqrtf(result);
			}
			else if (data_info[2] == 1)
			{ // L1 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					result += abs(data_d[data_id * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]);
				}
			}
			else if (data_info[2] == 0)
			{ // Max value
				float temp = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					temp = abs(data_d[data_id * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]);
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
					sa2 += data_d[qid_list[qid] * data_info[0] + j] * data_d[qid_list[qid] * data_info[0] + j];
					sa3 += data_d[data_id * data_info[0] + j] * data_d[qid_list[qid] * data_info[0] + j];
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
				int m = size_s[qid_list[qid]];
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
							int cost = (data_s[data_id * M + j - 1] == data_s[qid_list[qid] * M + k - 1]) ? 0 : 1;
							table[j][k] = 1 + min(table[j - 1][k], table[j][k - 1]);
							table[j][k] = min(table[j - 1][k - 1] + cost, table[j][k]);
						}
					}
					result = table[n][m];
				}
			}

			if (result > disk[qid])
			{
				result = INFI_DIS;
			}
		}

		// Save result (payload uses data_id for top-k output).
		p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * 3 + bid * MAX_SIZE + did] = (double)data_id;
		p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * (3 + MAX_SIZE) + bid * MAX_SIZE + did] = result;
		p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * (3 + 2 * MAX_SIZE) + bid * MAX_SIZE + did] =
			double(result / INFI_DIS + qid * DIS_CODE);
	}
}

// Process the nodes of the current layer and determine if the node will be pruned. (Vector Queries)
__global__ void dataProcessKnnVec(TN *node_list, float *disk, short *data_d, float *query_data, int *data_info, char *data_s, int *size_s,
								   double *p_list_k, int offset_p, int *id_list, int cur_level, int *size_list, int nnum_up)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int nid_p = bid;
	int qid_p = bid;
	int nid = p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) + nid_p];
	int qid = p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * 2 + qid_p];
	TN node = node_list[nid];

	for (int did = tid; did < MAX_SIZE; did += THREAD_NUM)
	{
		double result = INFI_DIS;
		int data_id = -1;

		if (did < node.size && node.is_leaf == 1)
		{
			data_id = id_list[node.lid + did];
			result = 0;

			if (data_info[2] == 2)
			{ // L2 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					float diff = (float)data_d[data_id * data_info[0] + j] - query_data[qid * data_info[0] + j];
					result += diff * diff;
				}
				result = sqrtf(result);
			}
			else if (data_info[2] == 1)
			{ // L1 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					result += abs((float)data_d[data_id * data_info[0] + j] - query_data[qid * data_info[0] + j]);
				}
			}
			else if (data_info[2] == 0)
			{ // Max value
				float temp = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					temp = abs((float)data_d[data_id * data_info[0] + j] - query_data[qid * data_info[0] + j]);
					if (temp > result)
						result = temp;
				}
			}
			else if (data_info[2] == 5)
			{
				float sa1 = 0, sa2 = 0, sa3 = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					float v = (float)data_d[data_id * data_info[0] + j];
					sa1 += v * v;
					float qv = query_data[qid * data_info[0] + j];
					sa2 += qv * qv;
					sa3 += v * qv;
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
				result = INFI_DIS;
			}

			if (result > disk[qid])
			{
				result = INFI_DIS;
			}
		}

		p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * 3 + bid * MAX_SIZE + did] = (double)data_id;
		p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * (3 + MAX_SIZE) + bid * MAX_SIZE + did] = result;
		p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * (3 + 2 * MAX_SIZE) + bid * MAX_SIZE + did] =
			double(result / INFI_DIS + qid * DIS_CODE);
	}
}

// Merge result.
__global__ void mergeResRnn(int ls, int le, int *p_list, int offset_p, int *size_list, int cur_level, int nnum_up, int lnum, int *res)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < (le - ls); i += total_num)
	{
		int s = offset_p + size_list[cur_level] / (MAX_SIZE + 3) * 3 + p_list[offset_p + i * nnum_up] * MAX_SIZE;
		int e;
		if (i < le - ls - 1)
		{
			e = offset_p + size_list[cur_level] / (MAX_SIZE + 3) * 3 + p_list[offset_p + (i + 1) * nnum_up] * MAX_SIZE;
		}
		else
		{
			e = offset_p + size_list[cur_level] / (MAX_SIZE + 3) * 3 + lnum * MAX_SIZE;
		}

		int qid = p_list[offset_p + size_list[cur_level] / (MAX_SIZE + 3) * 2 + p_list[offset_p + i * nnum_up]];
		int num = thrust::reduce(thrust::device, p_list + s, p_list + e, 0);
		res[qid] = num;
	}
}

// Merge result.
__global__ void mergeResKnn(int ls, int le, double *p_list_k, int offset_p, int *size_list, int cur_level, int nnum_up, float *res_dis, int k)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < (le - ls); i += total_num)
	{
		int s_key = offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * (3 + 2 * MAX_SIZE) + p_list_k[offset_p + i * nnum_up] * MAX_SIZE;
		int idx_q = offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * 2 + p_list_k[offset_p + i * nnum_up];
		int qid = p_list_k[idx_q];
		double key_val = p_list_k[s_key + k - 1];
		res_dis[qid] = (float)((key_val - (double)qid * DIS_CODE) * INFI_DIS);
	}
}

// Merge result with IDs (top-k)
__global__ void mergeResKnnIds(int ls, int le, double *p_list_k, int offset_p, int *size_list, int cur_level, int nnum_up,
							int *res_ids, float *res_dis, int k)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < (le - ls); i += total_num)
	{
		int s_id = offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * 3 + p_list_k[offset_p + i * nnum_up] * MAX_SIZE;
		int s_key = offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * (3 + 2 * MAX_SIZE) + p_list_k[offset_p + i * nnum_up] * MAX_SIZE;
		int idx_q = offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * 2 + p_list_k[offset_p + i * nnum_up];
		int qid = p_list_k[idx_q];
		for (int jj = 0; jj < k; ++jj)
		{
			res_ids[qid * k + jj] = (int)p_list_k[s_id + jj];
			double key_val = p_list_k[s_key + jj];
			res_dis[qid * k + jj] = (float)((key_val - (double)qid * DIS_CODE) * INFI_DIS);
		}
	}
}

// Initialize res.
__global__ void initResV2(int *res, int qnum)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < qnum; i += total_num)
	{
		res[i] = 0;
	}
}

// Initialize disk.
__global__ void initDisK(float *disk, int qnum)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < qnum; i += total_num)
	{
		disk[i] = INFI_DIS;
	}
}

// Label child nodes without varification.
__global__ void labelCNode(int *empty_list, int qnum_l, int qnum_up, int nnum_l, double *p_list_k, int offset_p, int offset_up_p, int offset_n,
						   int qs, int qs_up)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;
	int ofst = (nnum_l / TREE_ORDER * 3) * qnum_l; // Offset at current level.
	int temp = (nnum_l / TREE_ORDER / TREE_ORDER * 3);
	int ofst_up = temp * qnum_up; // Offset at upper level.

	/*if (tid == 0) {
		printf("ofst: %d\n", ofst);
		printf("ofst_up: %d\n", ofst_up);
	}*/

	for (int i = tid; i < nnum_l * qnum_l; i += total_num)
	{
		int qid_p = i % qnum_l;				   // Query idx in p_list
		int qid_p_up = qid_p + qs - qs_up;	   // Query idx at upper layer in p_list
		int nid_p = i / qnum_l;				   // Node idx in p_list at current level.
		int nid_parent_p = nid_p / TREE_ORDER; // Parent node idx in p_list at upper level.
		int nid = nid_p + offset_n;			   // Node idx in node_list

		// Reset p_list
		p_list_k[offset_p + ofst + i] = 0;

		if (p_list_k[offset_up_p + ofst_up + nid_parent_p * qnum_up + qid_p_up] == 1 && (empty_list[nid] == 0))
		{
			p_list_k[offset_p + ofst + i] = 1;
		}
	}
}

// Compute the distances between pivots and queries at current level.
__global__ void getDisPQ(TN *node_list, short *data_d, int *qid_list, int *data_info, int *empty_list, char *data_s,
						 int *size_s, int qnum_l, int qnum_up, int nnum_l, double *p_list_k, int offset_p, int offset_up_p, int offset_n,
						 int qs, int qs_up, int pnum_level_total)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < pnum_level_total * qnum_l; i += total_num)
	{
		int qid_p = i % qnum_l;						   // Query idx in p_list
		int qid_p_up = qid_p + qs - qs_up;			   // Query idx at upper layer in p_list
		int qid = qid_p + qs;						   // Query idx in qid_lsit
		int nid_p = (i / qnum_l) * TREE_ORDER;		   // The first node idx in p_list at current level.
		int nid_parent_p = nid_p / TREE_ORDER;		   // Parent node idx in p_list at upper level.
		int nid = nid_p + offset_n;					   // Node idx in node_list
		int ofst = (nnum_l / TREE_ORDER * 3) * qnum_l; // Offset at current level.
		int temp = (nnum_l / TREE_ORDER / TREE_ORDER * 3);
		int ofst_up = temp * qnum_up; // Offset at upper level.

		double dis_q = INFI_DIS;

		if (p_list_k[offset_up_p + ofst_up + nid_parent_p * qnum_up + qid_p_up] == 1 && (empty_list[nid] == 0))
		{
			TN node = node_list[nid];

			dis_q = 0;
			if (data_info[2] == 2)
			{ // L2 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					{ float _d = data_d[node.pid * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]; dis_q += _d * _d; }
				}
				dis_q = sqrtf(dis_q);
			}
			else if (data_info[2] == 1)
			{ // L1 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					dis_q += abs(data_d[node.pid * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]);
				}
			}
			else if (data_info[2] == 0)
			{ // Max value
				float temp = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					temp = abs(data_d[node.pid * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]);
					if (temp > dis_q)
						dis_q = temp;
				}
			}
			else if (data_info[2] == 5)
			{
				float sa1 = 0, sa2 = 0, sa3 = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					sa1 += data_d[node.pid * data_info[0] + j] * data_d[node.pid * data_info[0] + j];
					sa2 += data_d[qid_list[qid] * data_info[0] + j] * data_d[qid_list[qid] * data_info[0] + j];
					sa3 += data_d[node.pid * data_info[0] + j] * data_d[qid_list[qid] * data_info[0] + j];
				}
				sa1 = sqrtf(sa1);
				sa2 = sqrtf(sa2);
				if (sa1 * sa2 == 0)
				{
					printf("Error!!!\n");
				}
				dis_q = sa3 / (sa1 * sa2);
				if (dis_q > 1)
				{
					dis_q = 0.99999999999999999;
				}
				dis_q = abs(acos(dis_q) * 180 / 3.1415926);
			}
			else if (data_info[2] == 6)
			{
				int n = size_s[node.pid];
				int m = size_s[qid_list[qid]];
				int table[M][M];
				if (n == 0)
					dis_q = m;
				if (m == 0)
					dis_q = n;
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
							int cost = (data_s[node.pid * M + j - 1] == data_s[qid_list[qid] * M + k - 1]) ? 0 : 1;
							table[j][k] = 1 + min(table[j - 1][k], table[j][k - 1]);
							table[j][k] = min(table[j - 1][k - 1] + cost, table[j][k]);
						}
					}
					dis_q = table[n][m];
				}
			}
		}

		// Save result.
		p_list_k[offset_p + i] = i;
		p_list_k[offset_p + ofst / 3 + i] = dis_q;
		p_list_k[offset_p + ofst / 3 * 2 + i] = double(dis_q / INFI_DIS + qid_p * DIS_CODE);
	}
}

// Compute the distances between pivots and queries at current level. (Vector Queries)
__global__ void getDisPQVec(TN *node_list, short *data_d, float *query_data, int *data_info, int *empty_list, char *data_s,
							int *size_s, int qnum_l, int qnum_up, int nnum_l, double *p_list_k, int offset_p, int offset_up_p, int offset_n,
							int qs, int qs_up, int pnum_level_total)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockIdx.x + threadIdx.x;
	int total = gridDim.x * blockDim.x;

	for (int i = tid; i < pnum_level_total * qnum_l; i += total)
	{
		int qid_p = i % qnum_l;
		int qid_p_up = qid_p + qs - qs_up;
		int qid = qid_p + qs;
		int nid_p = (i / qnum_l) * TREE_ORDER;
		int nid_parent_p = nid_p / TREE_ORDER;
		int nid = nid_p + offset_n;
		int ofst = (nnum_l / TREE_ORDER * 3) * qnum_l;
		int temp = (nnum_l / TREE_ORDER / TREE_ORDER * 3);
		int ofst_up = temp * qnum_up;

		double dis_q = INFI_DIS;

		if (p_list_k[offset_up_p + ofst_up + nid_parent_p * qnum_up + qid_p_up] == 1 && (empty_list[nid] == 0))
		{
			TN node = node_list[nid];

			dis_q = 0;
			if (data_info[2] == 2)
			{
				for (int j = 0; j < data_info[0]; j++)
				{
					float diff = (float)data_d[node.pid * data_info[0] + j] - query_data[qid * data_info[0] + j];
					dis_q += diff * diff;
				}
				dis_q = sqrtf(dis_q);
			}
			else if (data_info[2] == 1)
			{
				for (int j = 0; j < data_info[0]; j++)
				{
					dis_q += abs((float)data_d[node.pid * data_info[0] + j] - query_data[qid * data_info[0] + j]);
				}
			}
			else if (data_info[2] == 0)
			{
				float tempv = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					tempv = abs((float)data_d[node.pid * data_info[0] + j] - query_data[qid * data_info[0] + j]);
					if (tempv > dis_q)
						dis_q = tempv;
				}
			}
			else if (data_info[2] == 5)
			{
				float sa1 = 0, sa2 = 0, sa3 = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					float v = (float)data_d[node.pid * data_info[0] + j];
					sa1 += v * v;
					float qv = query_data[qid * data_info[0] + j];
					sa2 += qv * qv;
					sa3 += v * qv;
				}
				sa1 = sqrtf(sa1);
				sa2 = sqrtf(sa2);
				if (sa1 * sa2 == 0)
				{
					printf("Error!!!\n");
				}
				dis_q = sa3 / (sa1 * sa2);
				if (dis_q > 1)
				{
					dis_q = 0.99999999999999999;
				}
				dis_q = abs(acos(dis_q) * 180 / 3.1415926);
			}
			else if (data_info[2] == 6)
			{
				dis_q = INFI_DIS;
			}
		}

		p_list_k[offset_p + i] = i;
		p_list_k[offset_p + ofst / 3 + i] = dis_q;
		p_list_k[offset_p + ofst / 3 * 2 + i] = double(dis_q / INFI_DIS + qid_p * DIS_CODE);
	}
}

// Update disk.
__global__ void updateDisK(int qnum_l, double *p_list_k, float *disk, int nnum_l, int offset_p, int qs, int k)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;
	int ofst = (nnum_l / TREE_ORDER * 3) * qnum_l; // Offset at current level.

	for (int i = tid; i < qnum_l; i += total_num)
	{
		int qid_p = i;		  // Query idx in p_list
		int qid = qid_p + qs; // Query idx in qid_lsit
		// float dis = INFI_DIS / INFI_DIS + qid_p * DIS_CODE;
		/*float* address = thrust::find(thrust::device, p_list_k + offset_p + ofst / 3 * 2 + nnum_l / TREE_ORDER * qid_p,
			p_list_k + offset_p + ofst / 3 * 2 + nnum_l / TREE_ORDER * (qid_p + 1), dis);
		int idx = address - (p_list_k + offset_p + ofst / 3 * 2 + nnum_l / TREE_ORDER * qid_p);*/
		int idx = p_list_k[offset_p + nnum_l / TREE_ORDER * qid_p + k - 1];

		if (disk[qid] > p_list_k[offset_p + ofst / 3 + idx])
		{
			disk[qid] = p_list_k[offset_p + ofst / 3 + idx];
			// printf("disk[qid]: %f, qid: %d\n", disk[qid], qid);
		}
	}
}

// Range query
void searchIndexRnnV2(short *data_d, TN *node_list, int *id_list, int *max_node_num, int *qid_list,
					  int qnum, float r, int tree_h, int *data_info, int *empty_list, char *data_s, int *size_s)
{
	cout << "Searching..." << endl;

	CHECK(cudaMallocManaged((void **)&res, qnum * sizeof(int)));
	CHECK(cudaMallocManaged((void **)&size_list, (tree_h + 1) * sizeof(int)));

	// Get GPU available memory.
	size_t avail;
	size_t total;
	safe_c1_traversal_require_cuda_success(cudaMemGetInfo(&avail, &total), "cudaMemGetInfo");
	// if (input_size <= 0 || input_size > avail) {
	// 	printf("Out of memory !!!\n");
	// 	return;
	// }
	// cout << "avail: " << avail << endl;
	// cout << "input: " << input_size << endl;
	// avail = input_size;
	// avail = avail / 2; // Allocate storage space as a half of available space.
	// cout << "avail: " << avail << endl;
	// cout << "total: " << total << endl;
	size_t limit = 4ULL * 1024 * 1024 * 1024; 
	if (avail > limit * 2) {
		avail = limit;
	} else {
		avail = avail / 2;
	}
	// Allocate memory
	size_a = avail / sizeof(int); // Get the total int num.
	CHECK(cudaMalloc((void **)&p_list, size_a * sizeof(int)));
	// cout << "size_a: " << size_a << endl;

	// Initialize the query information
	CHECK(cudaMemset(size_list, 0, (tree_h + 1) * sizeof(int)));
	CHECK(cudaMemset(p_list, 0, size_a * sizeof(int)));
	size_list[0] = qnum;
	size_a -= qnum;
	size_avg = size_a / tree_h;
	nnum_l = TREE_ORDER;
	qnum_l = min(size_avg / nnum_l, qnum);
	size_a -= qnum_l * nnum_l;
	size_list[1] = qnum_l * nnum_l;
	for (int i = 0; i < qnum; i += qnum_l)
	{
		int end = min(i + qnum_l - 1, qnum - 1);
		st.push(i);
		st.push(end);
		st.push(1);
		st.push(qnum);
		st.push(1);
		st.push(0);
		st.push(size_a);
	}
	initPList<<<(qnum + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(p_list, qnum);
	safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	cudaError_t cudaStatus = cudaGetLastError();
	safe_c1_traversal_require_cuda_success(cudaStatus, "initPlist");
	/* Retired initResV2 path: any future CUDA status must call
	 * safe_c1_traversal_require_cuda_success rather than print-and-continue. */

	// Range query
	while (!st.empty())
	{
		// Get the preparation information for queries
		size_a = st.top();
		st.pop();
		qs_up = st.top();
		st.pop();
		offset_n = st.top();
		st.pop();
		qnum_up = st.top();
		st.pop();
		cur_level = st.top();
		st.pop();
		qe = st.top();
		st.pop();
		qs = st.top();
		st.pop();
		qnum_l = qe - qs + 1;
		offset_p = thrust::reduce(thrust::device, size_list, size_list + cur_level, 0);
		offset_up_p = offset_p - size_list[cur_level - 1];
		nnum_l = pow(TREE_ORDER, cur_level);
		int block_num = (qnum_l * nnum_l + THREAD_NUM - 1) / THREAD_NUM;

		// Evaluating
		if (cur_level < tree_h)
		{ // Processing node.
			nodeProcessRnn<<<block_num, THREAD_NUM>>>(node_list, r, data_d, qid_list, data_info, empty_list, data_s,
													  size_s, qnum_l, qnum_up, nnum_l, p_list, offset_p, offset_up_p, offset_n, qs, qs_up);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "nodeProcessRnn");
		}
		else
		{ // Processing data in leaf node.
			// Get query information.
			int ls = qs;
			int le = qe;
			int offset_up_n = offset_n;
			int nnum_up = pow(TREE_ORDER, cur_level - 1);

			// Get counts of query.
			block_num = (qnum_up * nnum_up + THREAD_NUM - 1) / THREAD_NUM;
			getQCount<<<block_num, THREAD_NUM>>>(ls, le, p_list, offset_up_p, offset_p, qnum_up, nnum_up);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "getQCount");

			// Gets the prefix sum of p_list at leaf node layer.
			int lnum = thrust::reduce(thrust::device, p_list + offset_p, p_list + offset_p + (le - ls) * nnum_up, 0);
			thrust::exclusive_scan(thrust::device, p_list + offset_p, p_list + offset_p + (le - ls) * nnum_up,
								   p_list + offset_p);

			// Merge leaf node.
			mergeLNode<<<block_num, THREAD_NUM>>>(ls, le, p_list, offset_up_p, offset_up_n, qs_up, offset_p, size_list,
												  cur_level, qnum_up, nnum_up);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "mergeLNode");

			// Processing data in leaf node.
			block_num = lnum;
			// printf("lnum: %d\n", block_num);
			dataProcessRnn<<<block_num, THREAD_NUM>>>(node_list, r, data_d, qid_list, data_info, data_s, size_s, p_list,
													  offset_p, id_list, cur_level, size_list, nnum_up);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "dataProcessRnn");

			// Merge result.
			block_num = (le - ls + THREAD_NUM - 1) / THREAD_NUM;
			mergeResRnn<<<block_num, THREAD_NUM>>>(ls, le, p_list, offset_p, size_list, cur_level, nnum_up, lnum, res);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "mergeResRnn");
		}

		// Update the query and storage space information and of lower layer.
		if (cur_level < tree_h)
		{
			int cur_level_low = cur_level + 1;
			int size_avg_low = size_a / (tree_h - cur_level);
			int offset_n_low = offset_n + nnum_l;

			if (cur_level_low < tree_h)
			{ // The lower layer evaluates the entire nodes.
				// Update the query and storage space information and of lower layer.
				int nnum_l_low = pow(TREE_ORDER, cur_level_low);
				int qnum_l_low = min(size_avg_low / nnum_l_low, size_list[cur_level] / nnum_l);
				size_list[cur_level_low] = qnum_l_low * nnum_l_low;
				for (int i = 0; i < qnum_l; i += qnum_l_low)
				{
					int end = min(i + qs + qnum_l_low - 1, qe);
					st.push(i + qs);
					st.push(end);
					st.push(cur_level_low);
					st.push(qnum_l);
					st.push(offset_n_low);
					st.push(qs);
					st.push(size_a - size_list[cur_level_low]);
				}
			}
			else
			{ // The lower layer evaluates the data in leaf nodes.
				// Update the query and storage space informationand of lower layer.
				unsigned long size_l = min((unsigned long)size_avg_low / (MAX_SIZE + 3), (unsigned long)size_list[cur_level]);
				size_list[cur_level_low] = size_l * (MAX_SIZE + 3);
				// printf("size_l: %d, MAX_SIZE: %d,  nnum_l: %d", size_l, MAX_SIZE, nnum_l);
				unsigned long qnum_l_low = size_l / (nnum_l);
				printf("qnum_l_low: %lu\n", qnum_l_low);
				for (int i = 0; i < qnum_l; i += qnum_l_low)
				{
					int end = min((int)(i + qnum_l_low), qnum_l);
					st.push(i);
					st.push(end);
					st.push(cur_level_low);
					st.push(qnum_l);
					st.push(offset_n);
					st.push(qs);
					st.push(size_a - size_list[cur_level_low]);
				}
			}
		}
	}

	// Release memory
	safe_c1_traversal_require_cuda_success(cudaFree(p_list), "cudaFree p_list");
	safe_c1_traversal_require_cuda_success(cudaFree(size_list), "cudaFree size_list");
}

// knn query
void searchIndexKnnV2(short *data_d, TN *node_list, int *id_list, int *max_node_num, int *qid_list,
					  int qnum, int k, int tree_h, int *data_info, int *empty_list, char *data_s, int *size_s)
{
	cout << "Searching..." << endl;

	// if (debug_logs == nullptr){ 
	// 	CHECK(cudaMallocManaged((void **)&debug_logs, MAX_LOGS * sizeof(TrainingSample)));
	// }
	// CHECK(cudaMemset(debug_logs, 0, MAX_LOGS * sizeof(TrainingSample)));

	// debug_log_count = 0;

	CHECK(cudaMallocManaged((void **)&res_dis, qnum * sizeof(float)));
	CHECK(cudaMallocManaged((void **)&size_list, (tree_h + 1) * sizeof(int)));
	CHECK(cudaMalloc((void **)&disk, qnum * sizeof(float)));

	// Get GPU available memory.
	size_t avail;
	size_t total;
	safe_c1_traversal_require_cuda_success(cudaMemGetInfo(&avail, &total), "cudaMemGetInfo");
	// if (input_size <= 0 || input_size > avail) {
	// 	printf("Out of memory !!!\n");
	// 	return;
	// }
	// cout << "avail: " << avail << endl;
	// cout << "input: " << input_size << endl;
	// avail = input_size;
	// avail = avail / 2; // Allocate storage space as a half of available space.
	// cout << "avail: " << avail << endl;
	// cout << "total: " << total << endl;
	size_t limit = 4ULL * 1024 * 1024 * 1024; 
	if (avail > limit * 2) {
		avail = limit;
	} else {
		avail = avail / 2;
	}
	// Allocate memory
	size_a = avail / (sizeof(double)); // Get the total num.
	CHECK(cudaMalloc((void **)&p_list_k, size_a * sizeof(double)));
	// CHECK(cudaMalloc((void**)&p_list_dis, size_a * sizeof(float)));
	// CHECK(cudaMalloc((void**)&p_list_disc, size_a * sizeof(float)));
	// cout << "size_a: " << size_a << endl;

	// Initialize the query information
	CHECK(cudaMemset(size_list, 0, (tree_h + 1) * sizeof(int)));
	// CHECK(cudaMemset(p_list, 0, size_a * sizeof(int)));
	size_list[0] = qnum;
	size_a -= qnum;
	size_avg = size_a / tree_h;
	nnum_l = TREE_ORDER;
	qnum_l = min(size_avg / (nnum_l + nnum_l / TREE_ORDER * 3), qnum);
	size_a -= qnum_l * (nnum_l + nnum_l / TREE_ORDER);
	size_list[1] = qnum_l * (nnum_l + nnum_l / TREE_ORDER);
	for (int i = 0; i < qnum; i += qnum_l)
	{
		int end = min(i + qnum_l - 1, qnum - 1);
		st.push(i);
		st.push(end);
		st.push(1);
		st.push(qnum);
		st.push(1);
		st.push(0);
		st.push(size_a);
	}
	initPListKnn<<<(qnum + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(p_list_k, qnum);
	safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	cudaError_t cudaStatus = cudaGetLastError();
	safe_c1_traversal_require_cuda_success(cudaStatus, "initPlist");
	initDisK<<<(qnum + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(disk, qnum);
	safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	cudaStatus = cudaGetLastError();
	safe_c1_traversal_require_cuda_success(cudaStatus, "initDisk");

	// knn query
	while (!st.empty())
	{
		// Get the preparation information for queries
		size_a = st.top();
		st.pop();
		qs_up = st.top();
		st.pop();
		offset_n = st.top();
		st.pop();
		qnum_up = st.top();
		st.pop();
		cur_level = st.top();
		st.pop();
		qe = st.top();
		st.pop();
		qs = st.top();
		st.pop();
		qnum_l = qe - qs + 1;
		offset_p = thrust::reduce(thrust::device, size_list, size_list + cur_level, 0);
		offset_up_p = offset_p - size_list[cur_level - 1];
		nnum_l = pow(TREE_ORDER, cur_level);
		int block_num = (qnum_l * nnum_l + THREAD_NUM - 1) / THREAD_NUM;

		// Evaluating
		if (cur_level < tree_h)
		{ // Processing node.
			int pnum_level = nnum_l - thrust::reduce(thrust::device, empty_list + start_idx, empty_list + start_idx + nnum_l, 0);
			pnum_level = pnum_level / TREE_ORDER;
			// printf("pnum level: %d\n", pnum_level);
			int pnum_level_total = nnum_l / TREE_ORDER;
			// printf("pnum level total: %d\n", pnum_level_total);

			if (update_disk == false && (pnum_level < k || cur_level <= 2))
			{
				labelCNode<<<block_num, THREAD_NUM>>>(empty_list, qnum_l, qnum_up, nnum_l, p_list_k, offset_p, offset_up_p, offset_n, qs, qs_up);
				safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
				cudaStatus = cudaGetLastError();
				safe_c1_traversal_require_cuda_success(cudaStatus, "labelCNode");
			}
			else
			{
				update_disk = true;

				if (pnum_level >= k)
				{
					// Compute the distances between pivots and queries at current level.
					block_num = (pnum_level_total * qnum_l + THREAD_NUM - 1) / THREAD_NUM;
					getDisPQ<<<block_num, THREAD_NUM>>>(node_list, data_d, qid_list, data_info, empty_list, data_s, size_s, qnum_l,
														qnum_up, nnum_l, p_list_k, offset_p, offset_up_p, offset_n, qs, qs_up, pnum_level_total);
					safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
					cudaStatus = cudaGetLastError();
					safe_c1_traversal_require_cuda_success(cudaStatus, "getDisPQ");

					// Sort by distances.
					int ofst = (nnum_l / TREE_ORDER * 3) * qnum_l; // Offset at current level.
					thrust::sort_by_key(thrust::device, p_list_k + offset_p + ofst / 3 * 2,
										p_list_k + offset_p + ofst / 3 * 2 + pnum_level_total * qnum_l, p_list_k + offset_p);
					cudaStatus = cudaGetLastError();
					safe_c1_traversal_require_cuda_success(cudaStatus, "sort_by_key");

					// Update disk.
					updateDisK<<<(qnum_l + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(qnum_l, p_list_k, disk, nnum_l, offset_p, qs, k);
					safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
					cudaStatus = cudaGetLastError();
					safe_c1_traversal_require_cuda_success(cudaStatus, "updateDisK");
				}

				// Process the nodes of the current layer and determine if the node will be pruned.
				block_num = (nnum_l * qnum_l + THREAD_NUM - 1) / THREAD_NUM;
				nodeProcessKnn<<<block_num, THREAD_NUM>>>(node_list, disk, empty_list, qnum_l, qnum_up, nnum_l, p_list_k, offset_p,
														  offset_up_p, offset_n, qs, qs_up, cur_level);
				safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
				cudaStatus = cudaGetLastError();
				safe_c1_traversal_require_cuda_success(cudaStatus, "nodeProcessKnn");
			}
		}
		else
		{ // Processing data in leaf node.
			// Get query information.
			int ls = qs;
			int le = qe;
			int offset_up_n = offset_n;
			int nnum_up = pow(TREE_ORDER, cur_level - 1);

			// Get counts of query.
			block_num = (qnum_up * nnum_up + THREAD_NUM - 1) / THREAD_NUM;
			getQCountKnn<<<block_num, THREAD_NUM>>>(ls, le, p_list_k, offset_up_p, offset_p, qnum_up, nnum_up);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "getQCountKnn");

			// Gets the prefix sum of p_list at leaf node layer.
			int lnum = thrust::reduce(thrust::device, p_list_k + offset_p, p_list_k + offset_p + (le - ls) * nnum_up, 0);
			thrust::exclusive_scan(thrust::device, p_list_k + offset_p, p_list_k + offset_p + (le - ls) * nnum_up,
								   p_list_k + offset_p);
			// printf("lnum: %d\n", lnum);

			// Merge leaf node.
			mergeLNodeKnn<<<block_num, THREAD_NUM>>>(ls, le, p_list_k, offset_up_p, offset_up_n, qs_up, offset_p, size_list,
													 cur_level, qnum_up, nnum_up);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "mergeLNodeKnn");

			// Processing data in leaf node.
			block_num = lnum;
			dataProcessKnn<<<block_num, THREAD_NUM>>>(node_list, disk, data_d, qid_list, data_info, data_s, size_s, p_list_k,
													  offset_p, id_list, cur_level, size_list, nnum_up);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "dataProcessKnn");

			// Sort by distances.
			thrust::sort_by_key(thrust::device, p_list_k + offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * (3 + 2 * MAX_SIZE),
								p_list_k + offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * (3 + 2 * MAX_SIZE) + lnum * MAX_SIZE,
								p_list_k + offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * 3);
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "sort_by_key");

			// Merge result.
			block_num = (le - ls + THREAD_NUM - 1) / THREAD_NUM;
			mergeResKnn<<<block_num, THREAD_NUM>>>(ls, le, p_list_k, offset_p, size_list, cur_level, nnum_up, res_dis, k);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "mergeResKnn");
		}

		// Update the query and storage space information and of lower layer.
		if (cur_level < tree_h)
		{
			int cur_level_low = cur_level + 1;
			int size_avg_low = size_a / (tree_h - cur_level);
			int offset_n_low = offset_n + nnum_l;

			if (cur_level_low < tree_h)
			{ // The lower layer evaluates the entire nodes.
				// Update the query and storage space information and of lower layer.
				int nnum_l_low = pow(TREE_ORDER, cur_level_low);
				int qnum_l_low = min(size_avg_low / (nnum_l_low + nnum_l_low / TREE_ORDER * 3), qnum_up);
				size_list[cur_level_low] = qnum_l_low * (nnum_l_low + nnum_l_low / TREE_ORDER * 3);
				for (int i = 0; i < qnum_l; i += qnum_l_low)
				{
					int end = min(i + qs + qnum_l_low - 1, qe);
					st.push(i + qs);
					st.push(end);
					st.push(cur_level_low);
					st.push(qnum_l);
					st.push(offset_n_low);
					st.push(qs);
					st.push(size_a - size_list[cur_level_low]);
				}
			}
			else
			{ // The lower layer evaluates the data in leaf nodes.
				// Update the query and storage space informationand of lower layer.
				unsigned long size_l = min((unsigned long)size_avg_low / (MAX_SIZE * 3 + 3), (unsigned long)qnum_l * nnum_l);
				size_list[cur_level_low] = size_l * (MAX_SIZE * 3 + 3);
				// printf("size_l: %d, MAX_SIZE: %d,  nnum_l: %d\n", size_l, MAX_SIZE, nnum_l);
				unsigned long qnum_l_low = size_l / (nnum_l);
				printf("qnum_l_low: %lu\n", static_cast<unsigned long>(qnum_l_low));
				for (int i = 0; i < qnum_l; i += qnum_l_low)
				{
					int end = min((int)(i + qnum_l_low), qnum_l);
					st.push(i);
					st.push(end);
					st.push(cur_level_low);
					st.push(qnum_l);
					st.push(offset_n);
					st.push(qs);
					st.push(size_a - size_list[cur_level_low]);
				}
			}
		}
	}
	safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	// //写入文件备份训练数据
	// FILE *fp = fopen("training_data_gist.csv", "w");
	// if (fp == nullptr)
	// {
	// Training-data debug logging is intentionally disabled in this isolated copy.
	// }
	// else
	// {
	// 	fprintf(fp, "lb,r,node_dist,nid,qid\n");
	// 	int valid_count = min(debug_log_count, MAX_LOGS);
	// 	for (int i = 0; i < valid_count; i++)
	// 	{
	// 		fprintf(fp, "%f,%f,%f,%d,%d\n",
	// 				debug_logs[i].feature_lb,
	// 				debug_logs[i].feature_r,
	// 				debug_logs[i].feature_node,
	// 				debug_logs[i].node_id,
	// 				debug_logs[i].query_id);
	// 	}
	// 	fclose(fp);
	// 	printf("Saved %d training samples.\n", debug_log_count);
	// }
	// debug_log_count = 0;
	// if (debug_logs != nullptr)
	// {
	// 	cudaFree(debug_logs);
	// 	debug_logs = nullptr;
	// }
	// Release memory
	safe_c1_traversal_require_cuda_success(cudaFree(p_list_k), "cudaFree p_list_k");
	safe_c1_traversal_require_cuda_success(cudaFree(size_list), "cudaFree size_list");
	safe_c1_traversal_require_cuda_success(cudaFree(disk), "cudaFree disk");
}

// knn query (vector queries, return kth distance)
void searchIndexKnnV2(short *data_d, TN *node_list, int *id_list, int *max_node_num, float *query_data,
				  int qnum, int k, int tree_h, int *data_info, int *empty_list, char *data_s, int *size_s)
{
	cout << "Searching..." << endl;

	CHECK(cudaMallocManaged((void **)&res_dis, qnum * sizeof(float)));
	CHECK(cudaMallocManaged((void **)&size_list, (tree_h + 1) * sizeof(int)));
	CHECK(cudaMalloc((void **)&disk, qnum * sizeof(float)));

	// Get GPU available memory.
	size_t avail;
	size_t total;
	safe_c1_traversal_require_cuda_success(cudaMemGetInfo(&avail, &total), "cudaMemGetInfo");
	size_t limit = 4ULL * 1024 * 1024 * 1024;
	if (avail > limit * 2)
	{
		avail = limit;
	}
	else
	{
		avail = avail / 2;
	}
	// Allocate memory
	size_a = avail / (sizeof(double));
	CHECK(cudaMalloc((void **)&p_list_k, size_a * sizeof(double)));

	// Initialize the query information
	CHECK(cudaMemset(size_list, 0, (tree_h + 1) * sizeof(int)));
	size_list[0] = qnum;
	size_a -= qnum;
	size_avg = size_a / tree_h;
	nnum_l = TREE_ORDER;
	qnum_l = min(size_avg / (nnum_l + nnum_l / TREE_ORDER * 3), qnum);
	size_a -= qnum_l * (nnum_l + nnum_l / TREE_ORDER);
	size_list[1] = qnum_l * (nnum_l + nnum_l / TREE_ORDER);
	for (int i = 0; i < qnum; i += qnum_l)
	{
		int end = min(i + qnum_l - 1, qnum - 1);
		st.push(i);
		st.push(end);
		st.push(1);
		st.push(qnum);
		st.push(1);
		st.push(0);
		st.push(size_a);
	}
	initPListKnn<<<(qnum + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(p_list_k, qnum);
	safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	cudaError_t cudaStatus = cudaGetLastError();
	safe_c1_traversal_require_cuda_success(cudaStatus, "initPlist");
	initDisK<<<(qnum + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(disk, qnum);
	safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	cudaStatus = cudaGetLastError();
	safe_c1_traversal_require_cuda_success(cudaStatus, "initDisk");

	// knn query
	while (!st.empty())
	{
		// Get the preparation information for queries
		size_a = st.top();
		st.pop();
		qs_up = st.top();
		st.pop();
		offset_n = st.top();
		st.pop();
		qnum_up = st.top();
		st.pop();
		cur_level = st.top();
		st.pop();
		qe = st.top();
		st.pop();
		qs = st.top();
		st.pop();
		qnum_l = qe - qs + 1;
		offset_p = thrust::reduce(thrust::device, size_list, size_list + cur_level, 0);
		offset_up_p = offset_p - size_list[cur_level - 1];
		nnum_l = pow(TREE_ORDER, cur_level);
		int block_num = (qnum_l * nnum_l + THREAD_NUM - 1) / THREAD_NUM;

		// Evaluating
		if (cur_level < tree_h)
		{
			int pnum_level = nnum_l - thrust::reduce(thrust::device, empty_list + start_idx, empty_list + start_idx + nnum_l, 0);
			pnum_level = pnum_level / TREE_ORDER;
			int pnum_level_total = nnum_l / TREE_ORDER;

			if (update_disk == false && (pnum_level < k || cur_level <= 2))
			{
				labelCNode<<<block_num, THREAD_NUM>>>(empty_list, qnum_l, qnum_up, nnum_l, p_list_k, offset_p, offset_up_p, offset_n, qs, qs_up);
				safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
				cudaStatus = cudaGetLastError();
				safe_c1_traversal_require_cuda_success(cudaStatus, "labelCNode");
			}
			else
			{
				update_disk = true;

				if (pnum_level >= k)
				{
					// Compute the distances between pivots and queries at current level.
					block_num = (pnum_level_total * qnum_l + THREAD_NUM - 1) / THREAD_NUM;
					getDisPQVec<<<block_num, THREAD_NUM>>>(node_list, data_d, query_data, data_info, empty_list, data_s, size_s, qnum_l,
										 qnum_up, nnum_l, p_list_k, offset_p, offset_up_p, offset_n, qs, qs_up, pnum_level_total);
					safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
					cudaStatus = cudaGetLastError();
					safe_c1_traversal_require_cuda_success(cudaStatus, "getDisPQVec");

					// Sort by distances.
					int ofst = (nnum_l / TREE_ORDER * 3) * qnum_l;
					thrust::sort_by_key(thrust::device, p_list_k + offset_p + ofst / 3 * 2,
											p_list_k + offset_p + ofst / 3 * 2 + pnum_level_total * qnum_l, p_list_k + offset_p);
					cudaStatus = cudaGetLastError();
					safe_c1_traversal_require_cuda_success(cudaStatus, "sort_by_key");

					// Update disk.
					updateDisK<<<(qnum_l + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(qnum_l, p_list_k, disk, nnum_l, offset_p, qs, k);
					safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
					cudaStatus = cudaGetLastError();
					safe_c1_traversal_require_cuda_success(cudaStatus, "updateDisK");
				}

				// Process the nodes of the current layer and determine if the node will be pruned.
				block_num = (nnum_l * qnum_l + THREAD_NUM - 1) / THREAD_NUM;
				nodeProcessKnn<<<block_num, THREAD_NUM>>>(node_list, disk, empty_list, qnum_l, qnum_up, nnum_l, p_list_k, offset_p,
										  offset_up_p, offset_n, qs, qs_up, cur_level);
				safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
				cudaStatus = cudaGetLastError();
				safe_c1_traversal_require_cuda_success(cudaStatus, "nodeProcessKnn");
			}
		}
		else
		{
			// Processing data in leaf node.
			int ls = qs;
			int le = qe;
			int offset_up_n = offset_n;
			int nnum_up = pow(TREE_ORDER, cur_level - 1);

			// Get counts of query.
			block_num = (qnum_up * nnum_up + THREAD_NUM - 1) / THREAD_NUM;
			getQCountKnn<<<block_num, THREAD_NUM>>>(ls, le, p_list_k, offset_up_p, offset_p, qnum_up, nnum_up);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "getQCountKnn");

			// Gets the prefix sum of p_list at leaf node layer.
			int lnum = thrust::reduce(thrust::device, p_list_k + offset_p, p_list_k + offset_p + (le - ls) * nnum_up, 0);
			thrust::exclusive_scan(thrust::device, p_list_k + offset_p, p_list_k + offset_p + (le - ls) * nnum_up,
								   p_list_k + offset_p);

			// Merge leaf node.
			mergeLNodeKnn<<<block_num, THREAD_NUM>>>(ls, le, p_list_k, offset_up_p, offset_up_n, qs_up, offset_p, size_list,
									 cur_level, qnum_up, nnum_up);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "mergeLNodeKnn");

			// Processing data in leaf node.
			block_num = lnum;
			dataProcessKnnVec<<<block_num, THREAD_NUM>>>(node_list, disk, data_d, query_data, data_info, data_s, size_s, p_list_k,
								   offset_p, id_list, cur_level, size_list, nnum_up);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "dataProcessKnnVec");

			// Sort by distances.
			thrust::sort_by_key(thrust::device, p_list_k + offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * (3 + 2 * MAX_SIZE),
										p_list_k + offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * (3 + 2 * MAX_SIZE) + lnum * MAX_SIZE,
										p_list_k + offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * 3);
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "sort_by_key");

			// Merge result.
			block_num = (le - ls + THREAD_NUM - 1) / THREAD_NUM;
			mergeResKnn<<<block_num, THREAD_NUM>>>(ls, le, p_list_k, offset_p, size_list, cur_level, nnum_up, res_dis, k);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "mergeResKnn");
		}

		// Update the query and storage space information and of lower layer.
		if (cur_level < tree_h)
		{
			int cur_level_low = cur_level + 1;
			int size_avg_low = size_a / (tree_h - cur_level);
			int offset_n_low = offset_n + nnum_l;

			if (cur_level_low < tree_h)
			{
				int nnum_l_low = pow(TREE_ORDER, cur_level_low);
				int qnum_l_low = min(size_avg_low / (nnum_l_low + nnum_l_low / TREE_ORDER * 3), qnum_up);
				size_list[cur_level_low] = qnum_l_low * (nnum_l_low + nnum_l_low / TREE_ORDER * 3);
				for (int i = 0; i < qnum_l; i += qnum_l_low)
				{
					int end = min(i + qs + qnum_l_low - 1, qe);
					st.push(i + qs);
					st.push(end);
					st.push(cur_level_low);
					st.push(qnum_l);
					st.push(offset_n_low);
					st.push(qs);
					st.push(size_a - size_list[cur_level_low]);
				}
			}
			else
			{
				unsigned long size_l = min((unsigned long)size_avg_low / (MAX_SIZE * 3 + 3), (unsigned long)qnum_l * nnum_l);
				size_list[cur_level_low] = size_l * (MAX_SIZE * 3 + 3);
				unsigned long qnum_l_low = size_l / (nnum_l);
				printf("qnum_l_low: %lu\n", static_cast<unsigned long>(qnum_l_low));
				for (int i = 0; i < qnum_l; i += qnum_l_low)
				{
					int end = min((int)(i + qnum_l_low), qnum_l);
					st.push(i);
					st.push(end);
					st.push(cur_level_low);
					st.push(qnum_l);
					st.push(offset_n);
					st.push(qs);
					st.push(size_a - size_list[cur_level_low]);
				}
			}
		}
	}

	// Release memory
	safe_c1_traversal_require_cuda_success(cudaFree(p_list_k), "cudaFree p_list_k");
	safe_c1_traversal_require_cuda_success(cudaFree(size_list), "cudaFree size_list");
	safe_c1_traversal_require_cuda_success(cudaFree(disk), "cudaFree disk");
}

// knn query (return top-k ids and distances) - legacy ID queries
void searchIndexKnnV2(short *data_d, TN *node_list, int *id_list, int *max_node_num, int *qid_list,
				  int *res_ids, int qnum, int k, int tree_h, int *data_info, int *empty_list, char *data_s, int *size_s)
{
	cout << "Searching..." << endl;

	CHECK(cudaMallocManaged((void **)&res_dis, qnum * k * sizeof(float)));
	CHECK(cudaMallocManaged((void **)&size_list, (tree_h + 1) * sizeof(int)));
	CHECK(cudaMalloc((void **)&disk, qnum * sizeof(float)));

	// Get GPU available memory.
	size_t avail;
	size_t total;
	safe_c1_traversal_require_cuda_success(cudaMemGetInfo(&avail, &total), "cudaMemGetInfo");
	size_t limit = 4ULL * 1024 * 1024 * 1024;
	if (avail > limit * 2)
	{
		avail = limit;
	}
	else
	{
		avail = avail / 2;
	}
	// Allocate memory
	size_a = avail / (sizeof(double));
	CHECK(cudaMalloc((void **)&p_list_k, size_a * sizeof(double)));

	// Initialize the query information
	CHECK(cudaMemset(size_list, 0, (tree_h + 1) * sizeof(int)));
	size_list[0] = qnum;
	size_a -= qnum;
	size_avg = size_a / tree_h;
	nnum_l = TREE_ORDER;
	qnum_l = min(size_avg / (nnum_l + nnum_l / TREE_ORDER * 3), qnum);
	size_a -= qnum_l * (nnum_l + nnum_l / TREE_ORDER);
	size_list[1] = qnum_l * (nnum_l + nnum_l / TREE_ORDER);
	for (int i = 0; i < qnum; i += qnum_l)
	{
		int end = min(i + qnum_l - 1, qnum - 1);
		st.push(i);
		st.push(end);
		st.push(1);
		st.push(qnum);
		st.push(1);
		st.push(0);
		st.push(size_a);
	}
	initPListKnn<<<(qnum + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(p_list_k, qnum);
	safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	cudaError_t cudaStatus = cudaGetLastError();
	safe_c1_traversal_require_cuda_success(cudaStatus, "initPlist");
	initDisK<<<(qnum + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(disk, qnum);
	safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	cudaStatus = cudaGetLastError();
	safe_c1_traversal_require_cuda_success(cudaStatus, "initDisk");

	// knn query
	while (!st.empty())
	{
		// Get the preparation information for queries
		size_a = st.top();
		st.pop();
		qs_up = st.top();
		st.pop();
		offset_n = st.top();
		st.pop();
		qnum_up = st.top();
		st.pop();
		cur_level = st.top();
		st.pop();
		qe = st.top();
		st.pop();
		qs = st.top();
		st.pop();
		qnum_l = qe - qs + 1;
		offset_p = thrust::reduce(thrust::device, size_list, size_list + cur_level, 0);
		offset_up_p = offset_p - size_list[cur_level - 1];
		nnum_l = pow(TREE_ORDER, cur_level);
		int block_num = (qnum_l * nnum_l + THREAD_NUM - 1) / THREAD_NUM;

		// Evaluating
		if (cur_level < tree_h)
		{
			int pnum_level = nnum_l - thrust::reduce(thrust::device, empty_list + start_idx, empty_list + start_idx + nnum_l, 0);
			pnum_level = pnum_level / TREE_ORDER;
			int pnum_level_total = nnum_l / TREE_ORDER;

			if (update_disk == false && (pnum_level < k || cur_level <= 2))
			{
				labelCNode<<<block_num, THREAD_NUM>>>(empty_list, qnum_l, qnum_up, nnum_l, p_list_k, offset_p, offset_up_p, offset_n, qs, qs_up);
				safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
				cudaStatus = cudaGetLastError();
				safe_c1_traversal_require_cuda_success(cudaStatus, "labelCNode");
			}
			else
			{
				update_disk = true;

				if (pnum_level >= k)
				{
					// Compute the distances between pivots and queries at current level.
					block_num = (pnum_level_total * qnum_l + THREAD_NUM - 1) / THREAD_NUM;
					getDisPQ<<<block_num, THREAD_NUM>>>(node_list, data_d, qid_list, data_info, empty_list, data_s, size_s, qnum_l,
									 qnum_up, nnum_l, p_list_k, offset_p, offset_up_p, offset_n, qs, qs_up, pnum_level_total);
					safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
					cudaStatus = cudaGetLastError();
					safe_c1_traversal_require_cuda_success(cudaStatus, "getDisPQ");

					// Sort by distances.
					int ofst = (nnum_l / TREE_ORDER * 3) * qnum_l;
					thrust::sort_by_key(thrust::device, p_list_k + offset_p + ofst / 3 * 2,
											p_list_k + offset_p + ofst / 3 * 2 + pnum_level_total * qnum_l, p_list_k + offset_p);
					cudaStatus = cudaGetLastError();
					safe_c1_traversal_require_cuda_success(cudaStatus, "sort_by_key");

					// Update disk.
					updateDisK<<<(qnum_l + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(qnum_l, p_list_k, disk, nnum_l, offset_p, qs, k);
					safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
					cudaStatus = cudaGetLastError();
					safe_c1_traversal_require_cuda_success(cudaStatus, "updateDisK");
				}

				// Process the nodes of the current layer and determine if the node will be pruned.
				block_num = (nnum_l * qnum_l + THREAD_NUM - 1) / THREAD_NUM;
				nodeProcessKnn<<<block_num, THREAD_NUM>>>(node_list, disk, empty_list, qnum_l, qnum_up, nnum_l, p_list_k, offset_p,
										  offset_up_p, offset_n, qs, qs_up, cur_level);
				safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
				cudaStatus = cudaGetLastError();
				safe_c1_traversal_require_cuda_success(cudaStatus, "nodeProcessKnn");
			}
		}
		else
		{
			// Processing data in leaf node.
			int ls = qs;
			int le = qe;
			int offset_up_n = offset_n;
			int nnum_up = pow(TREE_ORDER, cur_level - 1);

			// Get counts of query.
			block_num = (qnum_up * nnum_up + THREAD_NUM - 1) / THREAD_NUM;
			getQCountKnn<<<block_num, THREAD_NUM>>>(ls, le, p_list_k, offset_up_p, offset_p, qnum_up, nnum_up);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "getQCountKnn");

			// Gets the prefix sum of p_list at leaf node layer.
			int lnum = thrust::reduce(thrust::device, p_list_k + offset_p, p_list_k + offset_p + (le - ls) * nnum_up, 0);
			thrust::exclusive_scan(thrust::device, p_list_k + offset_p, p_list_k + offset_p + (le - ls) * nnum_up,
								   p_list_k + offset_p);

			// Merge leaf node.
			mergeLNodeKnn<<<block_num, THREAD_NUM>>>(ls, le, p_list_k, offset_up_p, offset_up_n, qs_up, offset_p, size_list,
									 cur_level, qnum_up, nnum_up);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "mergeLNodeKnn");

			// Processing data in leaf node.
			block_num = lnum;
			dataProcessKnn<<<block_num, THREAD_NUM>>>(node_list, disk, data_d, qid_list, data_info, data_s, size_s, p_list_k,
								 offset_p, id_list, cur_level, size_list, nnum_up);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "dataProcessKnn");

			// Sort by distances.
			thrust::sort_by_key(thrust::device, p_list_k + offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * (3 + 2 * MAX_SIZE),
										p_list_k + offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * (3 + 2 * MAX_SIZE) + lnum * MAX_SIZE,
										p_list_k + offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * 3);
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "sort_by_key");

			// Merge result.
			block_num = (le - ls + THREAD_NUM - 1) / THREAD_NUM;
			mergeResKnnIds<<<block_num, THREAD_NUM>>>(ls, le, p_list_k, offset_p, size_list, cur_level, nnum_up, res_ids, res_dis, k);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "mergeResKnnIds");
		}

		// Update the query and storage space information and of lower layer.
		if (cur_level < tree_h)
		{
			int cur_level_low = cur_level + 1;
			int size_avg_low = size_a / (tree_h - cur_level);
			int offset_n_low = offset_n + nnum_l;

			if (cur_level_low < tree_h)
			{
				int nnum_l_low = pow(TREE_ORDER, cur_level_low);
				int qnum_l_low = min(size_avg_low / (nnum_l_low + nnum_l_low / TREE_ORDER * 3), qnum_up);
				size_list[cur_level_low] = qnum_l_low * (nnum_l_low + nnum_l_low / TREE_ORDER * 3);
				for (int i = 0; i < qnum_l; i += qnum_l_low)
				{
					int end = min(i + qs + qnum_l_low - 1, qe);
					st.push(i + qs);
					st.push(end);
					st.push(cur_level_low);
					st.push(qnum_l);
					st.push(offset_n_low);
					st.push(qs);
					st.push(size_a - size_list[cur_level_low]);
				}
			}
			else
			{
				unsigned long size_l = min((unsigned long)size_avg_low / (MAX_SIZE * 3 + 3), (unsigned long)qnum_l * nnum_l);
				size_list[cur_level_low] = size_l * (MAX_SIZE * 3 + 3);
				unsigned long qnum_l_low = size_l / (nnum_l);
				printf("qnum_l_low: %lu\n", static_cast<unsigned long>(qnum_l_low));
				for (int i = 0; i < qnum_l; i += qnum_l_low)
				{
					int end = min((int)(i + qnum_l_low), qnum_l);
					st.push(i);
					st.push(end);
					st.push(cur_level_low);
					st.push(qnum_l);
					st.push(offset_n);
					st.push(qs);
					st.push(size_a - size_list[cur_level_low]);
				}
			}
		}
	}

	// Release memory
	safe_c1_traversal_require_cuda_success(cudaFree(p_list_k), "cudaFree p_list_k");
	safe_c1_traversal_require_cuda_success(cudaFree(size_list), "cudaFree size_list");
	safe_c1_traversal_require_cuda_success(cudaFree(disk), "cudaFree disk");
}

// knn query (return top-k ids and distances) - vector queries
void searchIndexKnnV2(short *data_d, TN *node_list, int *id_list, int *max_node_num, float *query_data,
				  int *res_ids, int qnum, int k, int tree_h, int *data_info, int *empty_list, char *data_s, int *size_s)
{
	// Safe-C1 receipt is reset per real GTS traversal call.
	safe_c1_last_visited_leaf_pairs.clear();
	cout << "Searching..." << endl;

	CHECK(cudaMallocManaged((void **)&res_dis, qnum * k * sizeof(float)));
	CHECK(cudaMallocManaged((void **)&size_list, (tree_h + 1) * sizeof(int)));
	CHECK(cudaMalloc((void **)&disk, qnum * sizeof(float)));

	// Get GPU available memory.
	size_t avail;
	size_t total;
	safe_c1_traversal_require_cuda_success(cudaMemGetInfo(&avail, &total), "cudaMemGetInfo");
	size_t limit = 4ULL * 1024 * 1024 * 1024;
	if (avail > limit * 2)
	{
		avail = limit;
	}
	else
	{
		avail = avail / 2;
	}
	// Allocate memory
	size_a = avail / (sizeof(double));
	CHECK(cudaMalloc((void **)&p_list_k, size_a * sizeof(double)));

	// Initialize the query information
	CHECK(cudaMemset(size_list, 0, (tree_h + 1) * sizeof(int)));
	size_list[0] = qnum;
	size_a -= qnum;
	size_avg = size_a / tree_h;
	nnum_l = TREE_ORDER;
	qnum_l = min(size_avg / (nnum_l + nnum_l / TREE_ORDER * 3), qnum);
	size_a -= qnum_l * (nnum_l + nnum_l / TREE_ORDER);
	size_list[1] = qnum_l * (nnum_l + nnum_l / TREE_ORDER);
	for (int i = 0; i < qnum; i += qnum_l)
	{
		int end = min(i + qnum_l - 1, qnum - 1);
		st.push(i);
		st.push(end);
		st.push(1);
		st.push(qnum);
		st.push(1);
		st.push(0);
		st.push(size_a);
	}
	initPListKnn<<<(qnum + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(p_list_k, qnum);
	safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	cudaError_t cudaStatus = cudaGetLastError();
	safe_c1_traversal_require_cuda_success(cudaStatus, "initPlist");
	initDisK<<<(qnum + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(disk, qnum);
	safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	cudaStatus = cudaGetLastError();
	safe_c1_traversal_require_cuda_success(cudaStatus, "initDisk");

	// knn query
	while (!st.empty())
	{
		// Get the preparation information for queries
		size_a = st.top();
		st.pop();
		qs_up = st.top();
		st.pop();
		offset_n = st.top();
		st.pop();
		qnum_up = st.top();
		st.pop();
		cur_level = st.top();
		st.pop();
		qe = st.top();
		st.pop();
		qs = st.top();
		st.pop();
		qnum_l = qe - qs + 1;
		offset_p = thrust::reduce(thrust::device, size_list, size_list + cur_level, 0);
		offset_up_p = offset_p - size_list[cur_level - 1];
		nnum_l = pow(TREE_ORDER, cur_level);
		int block_num = (qnum_l * nnum_l + THREAD_NUM - 1) / THREAD_NUM;

		// Evaluating
		if (cur_level < tree_h)
		{
			int pnum_level = nnum_l - thrust::reduce(thrust::device, empty_list + start_idx, empty_list + start_idx + nnum_l, 0);
			pnum_level = pnum_level / TREE_ORDER;
			int pnum_level_total = nnum_l / TREE_ORDER;

			if (update_disk == false && (pnum_level < k || cur_level <= 2))
			{
				labelCNode<<<block_num, THREAD_NUM>>>(empty_list, qnum_l, qnum_up, nnum_l, p_list_k, offset_p, offset_up_p, offset_n, qs, qs_up);
				safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
				cudaStatus = cudaGetLastError();
				safe_c1_traversal_require_cuda_success(cudaStatus, "labelCNode");
			}
			else
			{
				update_disk = true;

				if (pnum_level >= k)
				{
					// Compute the distances between pivots and queries at current level.
					block_num = (pnum_level_total * qnum_l + THREAD_NUM - 1) / THREAD_NUM;
					getDisPQVec<<<block_num, THREAD_NUM>>>(node_list, data_d, query_data, data_info, empty_list, data_s, size_s, qnum_l,
										 qnum_up, nnum_l, p_list_k, offset_p, offset_up_p, offset_n, qs, qs_up, pnum_level_total);
					safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
					cudaStatus = cudaGetLastError();
					safe_c1_traversal_require_cuda_success(cudaStatus, "getDisPQVec");

					// Sort by distances.
					int ofst = (nnum_l / TREE_ORDER * 3) * qnum_l;
					thrust::sort_by_key(thrust::device, p_list_k + offset_p + ofst / 3 * 2,
											p_list_k + offset_p + ofst / 3 * 2 + pnum_level_total * qnum_l, p_list_k + offset_p);
					cudaStatus = cudaGetLastError();
					safe_c1_traversal_require_cuda_success(cudaStatus, "sort_by_key");

					// Update disk.
					updateDisK<<<(qnum_l + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(qnum_l, p_list_k, disk, nnum_l, offset_p, qs, k);
					safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
					cudaStatus = cudaGetLastError();
					safe_c1_traversal_require_cuda_success(cudaStatus, "updateDisK");
				}

				// Process the nodes of the current layer and determine if the node will be pruned.
				block_num = (nnum_l * qnum_l + THREAD_NUM - 1) / THREAD_NUM;
				nodeProcessKnn<<<block_num, THREAD_NUM>>>(node_list, disk, empty_list, qnum_l, qnum_up, nnum_l, p_list_k, offset_p,
										  offset_up_p, offset_n, qs, qs_up, cur_level);
				safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
				cudaStatus = cudaGetLastError();
				safe_c1_traversal_require_cuda_success(cudaStatus, "nodeProcessKnn");
			}
		}
		else
		{
			// Processing data in leaf node.
			int ls = qs;
			int le = qe;
			int offset_up_n = offset_n;
			int nnum_up = pow(TREE_ORDER, cur_level - 1);

			// Get counts of query.
			block_num = (qnum_up * nnum_up + THREAD_NUM - 1) / THREAD_NUM;
			getQCountKnn<<<block_num, THREAD_NUM>>>(ls, le, p_list_k, offset_up_p, offset_p, qnum_up, nnum_up);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "getQCountKnn");

			// Gets the prefix sum of p_list at leaf node layer.
			int lnum = thrust::reduce(thrust::device, p_list_k + offset_p, p_list_k + offset_p + (le - ls) * nnum_up, 0);
			thrust::exclusive_scan(thrust::device, p_list_k + offset_p, p_list_k + offset_p + (le - ls) * nnum_up,
								   p_list_k + offset_p);

			// Merge leaf node.
			mergeLNodeKnn<<<block_num, THREAD_NUM>>>(ls, le, p_list_k, offset_up_p, offset_up_n, qs_up, offset_p, size_list,
									 cur_level, qnum_up, nnum_up);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "mergeLNodeKnn");

			// Safe-C1 instrumentation: materialize only the (query, leaf) pairs that
			// the real GTS traversal has already selected, before native leaf scanning.
			if (lnum > 0) {
				const int receipt_stride = size_list[cur_level] / (MAX_SIZE * 3 + 3);
				std::vector<double> receipt_leaf(static_cast<std::size_t>(lnum));
				std::vector<double> receipt_query(static_cast<std::size_t>(lnum));
				CHECK(cudaMemcpy(receipt_leaf.data(), p_list_k + offset_p + receipt_stride,
				                 static_cast<std::size_t>(lnum) * sizeof(double), cudaMemcpyDeviceToHost));
				CHECK(cudaMemcpy(receipt_query.data(), p_list_k + offset_p + receipt_stride * 2,
				                 static_cast<std::size_t>(lnum) * sizeof(double), cudaMemcpyDeviceToHost));
				for (int receipt_index = 0; receipt_index < lnum; ++receipt_index) {
					const int leaf_id = static_cast<int>(receipt_leaf[receipt_index]);
					const int query_id = static_cast<int>(receipt_query[receipt_index]);
					if (leaf_id < 0 || query_id < 0 || query_id >= qnum) {
						safe_c1_traversal_fail_stop("invalid GTS KNN leaf receipt");
					}
					safe_c1_last_visited_leaf_pairs.push_back({query_id, leaf_id});
				}
			}

			// Processing data in leaf node.
			block_num = lnum;
			dataProcessKnnVec<<<block_num, THREAD_NUM>>>(node_list, disk, data_d, query_data, data_info, data_s, size_s, p_list_k,
								   offset_p, id_list, cur_level, size_list, nnum_up);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "dataProcessKnnVec");

			// Sort by distances.
			thrust::sort_by_key(thrust::device, p_list_k + offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * (3 + 2 * MAX_SIZE),
										p_list_k + offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * (3 + 2 * MAX_SIZE) + lnum * MAX_SIZE,
										p_list_k + offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * 3);
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "sort_by_key");

			// Merge result.
			block_num = (le - ls + THREAD_NUM - 1) / THREAD_NUM;
			mergeResKnnIds<<<block_num, THREAD_NUM>>>(ls, le, p_list_k, offset_p, size_list, cur_level, nnum_up, res_ids, res_dis, k);
			safe_c1_traversal_require_cuda_success(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
			cudaStatus = cudaGetLastError();
			safe_c1_traversal_require_cuda_success(cudaStatus, "mergeResKnnIds");
		}

		// Update the query and storage space information and of lower layer.
		if (cur_level < tree_h)
		{
			int cur_level_low = cur_level + 1;
			int size_avg_low = size_a / (tree_h - cur_level);
			int offset_n_low = offset_n + nnum_l;

			if (cur_level_low < tree_h)
			{
				int nnum_l_low = pow(TREE_ORDER, cur_level_low);
				int qnum_l_low = min(size_avg_low / (nnum_l_low + nnum_l_low / TREE_ORDER * 3), qnum_up);
				size_list[cur_level_low] = qnum_l_low * (nnum_l_low + nnum_l_low / TREE_ORDER * 3);
				for (int i = 0; i < qnum_l; i += qnum_l_low)
				{
					int end = min(i + qs + qnum_l_low - 1, qe);
					st.push(i + qs);
					st.push(end);
					st.push(cur_level_low);
					st.push(qnum_l);
					st.push(offset_n_low);
					st.push(qs);
					st.push(size_a - size_list[cur_level_low]);
				}
			}
			else
			{
				unsigned long size_l = min((unsigned long)size_avg_low / (MAX_SIZE * 3 + 3), (unsigned long)qnum_l * nnum_l);
				size_list[cur_level_low] = size_l * (MAX_SIZE * 3 + 3);
				unsigned long qnum_l_low = size_l / (nnum_l);
				printf("qnum_l_low: %lu\n", static_cast<unsigned long>(qnum_l_low));
				for (int i = 0; i < qnum_l; i += qnum_l_low)
				{
					int end = min((int)(i + qnum_l_low), qnum_l);
					st.push(i);
					st.push(end);
					st.push(cur_level_low);
					st.push(qnum_l);
					st.push(offset_n);
					st.push(qs);
					st.push(size_a - size_list[cur_level_low]);
				}
			}
		}
	}

	// Release memory
	safe_c1_traversal_require_cuda_success(cudaFree(p_list_k), "cudaFree p_list_k");
	safe_c1_traversal_require_cuda_success(cudaFree(size_list), "cudaFree size_list");
	safe_c1_traversal_require_cuda_success(cudaFree(disk), "cudaFree disk");
}