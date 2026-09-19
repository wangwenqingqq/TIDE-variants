// Device-side semantic closure for TIDE C1.
// Scope: frozen Base + live Delta + delete overlay + stable merge + rebuild.
// This is intentionally a minimal lifecycle harness.  It does NOT yet replace
// the native GTS tree traversal; its Base producer is an exact frozen-base scan.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kMaxIds = 2048;
constexpr int kMaxBase = 512;
constexpr int kMaxDelta = 512;
constexpr int kMaxK = 8;
constexpr float kInfinity = 3.402823466e+38F;

struct Record {
    int id;
    float x;
    float y;
};

struct Candidate {
    int id;
    float d2;
};

// Delta slot state: 0 = reusable, 1 = payload being written, 2 = published/live.
// Queries read only state 2. This is the publish-after-payload contract.
struct DeviceState {
    Record base[kMaxBase];
    Record next_base[kMaxBase];
    Record delta[kMaxDelta];
    int delta_state[kMaxDelta];
    int deleted[kMaxIds];
    int base_count;
    int delta_capacity;
    int failure_code;
};

struct QueryOutput {
    int count;
    Candidate best[kMaxK];
};

struct Config {
    int seeds = 8;
    int ops = 100;
    int delta_capacity = 64;
    int k = 3;
    std::string receipt_path;
};

inline void Check(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        std::ostringstream os;
        os << where << ": " << cudaGetErrorString(status);
        throw std::runtime_error(os.str());
    }
}

__device__ inline bool Better(const Candidate& lhs, const Candidate& rhs) {
    return (lhs.d2 < rhs.d2) || (lhs.d2 == rhs.d2 && lhs.id < rhs.id);
}

__device__ inline void InsertTopK(Candidate* top, int* count, int k, Candidate candidate) {
    int pos = 0;
    while (pos < *count && !Better(candidate, top[pos])) {
        ++pos;
    }
    if (pos >= k) {
        return;
    }
    if (*count < k) {
        ++(*count);
    }
    for (int j = *count - 1; j > pos; --j) {
        top[j] = top[j - 1];
    }
    top[pos] = candidate;
}

__device__ inline void Consider(DeviceState* state, Candidate* top, int* count, int k,
                                const Record& record, float qx, float qy) {
    if (record.id < 0 || record.id >= kMaxIds) {
        state->failure_code = 10;
        return;
    }
    if (state->deleted[record.id] != 0) {
        return;
    }
    const float dx = record.x - qx;
    const float dy = record.y - qy;
    InsertTopK(top, count, k, Candidate{record.id, dx * dx + dy * dy});
}

__global__ void InsertDelta(DeviceState* state, Record record) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (record.id < 0 || record.id >= kMaxIds) {
        state->failure_code = 11;
        return;
    }
    int slot = -1;
    for (int i = 0; i < state->delta_capacity; ++i) {
        if (atomicCAS(&state->delta_state[i], 0, 1) == 0) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        state->failure_code = 12;  // caller must rebuild before the live Delta is full.
        return;
    }
    state->delta[slot] = record;  // complete payload first
    __threadfence();
    state->delta_state[slot] = 2;  // then atomically publish as live
}

__global__ void DeleteId(DeviceState* state, int id) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (id < 0 || id >= kMaxIds) {
        state->failure_code = 13;
        return;
    }
    state->deleted[id] = 1;
    // A deleted delta entry is not live and its slot becomes reusable. Base data
    // remains physically immutable and is filtered through the same overlay.
    for (int i = 0; i < state->delta_capacity; ++i) {
        if (state->delta_state[i] == 2 && state->delta[i].id == id) {
            atomicCAS(&state->delta_state[i], 2, 0);
        }
    }
}

__global__ void QueryFrozenBasePlusLiveDelta(DeviceState* state, float qx, float qy, int k,
                                             QueryOutput* output) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    Candidate top[kMaxK];
    int count = 0;
    for (int i = 0; i < kMaxK; ++i) {
        top[i] = Candidate{2147483647, kInfinity};
    }

    // Base adapter: exact frozen-base candidate production.  The next closure
    // replaces this loop with the native GTS traversal without changing Delta,
    // overlay, or stable-merge semantics below.
    for (int i = 0; i < state->base_count; ++i) {
        Consider(state, top, &count, k, state->base[i], qx, qy);
    }
    // Every physical Delta slot is inspected; only published/live slots enter
    // the merge. This is exhaustive ScanLiveDelta, not route-local visibility.
    for (int i = 0; i < state->delta_capacity; ++i) {
        if (state->delta_state[i] == 2) {
            Consider(state, top, &count, k, state->delta[i], qx, qy);
        }
    }

    output->count = count;
    for (int i = 0; i < kMaxK; ++i) {
        output->best[i] = top[i];
    }
}

__global__ void RebuildFrozenBase(DeviceState* state) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int next_count = 0;
    for (int i = 0; i < state->base_count; ++i) {
        const Record record = state->base[i];
        if (record.id < 0 || record.id >= kMaxIds) {
            state->failure_code = 14;
            return;
        }
        if (state->deleted[record.id] == 0) {
            if (next_count >= kMaxBase) {
                state->failure_code = 15;
                return;
            }
            state->next_base[next_count++] = record;
        }
    }
    for (int i = 0; i < state->delta_capacity; ++i) {
        if (state->delta_state[i] != 2) {
            continue;
        }
        const Record record = state->delta[i];
        if (record.id < 0 || record.id >= kMaxIds) {
            state->failure_code = 16;
            return;
        }
        if (state->deleted[record.id] == 0) {
            if (next_count >= kMaxBase) {
                state->failure_code = 17;
                return;
            }
            state->next_base[next_count++] = record;
        }
    }
    for (int i = 0; i < next_count; ++i) {
        state->base[i] = state->next_base[i];
    }
    state->base_count = next_count;
    for (int i = 0; i < state->delta_capacity; ++i) {
        state->delta_state[i] = 0;
    }
    for (int id = 0; id < kMaxIds; ++id) {
        state->deleted[id] = 0;
    }
}

bool BetterHost(const Candidate& lhs, const Candidate& rhs) {
    return (lhs.d2 < rhs.d2) || (lhs.d2 == rhs.d2 && lhs.id < rhs.id);
}

class HostModel {
public:
    void Reset(std::vector<Record> base) {
        base_ = std::move(base);
        delta_.clear();
        deleted_.fill(0);
    }

    void Insert(Record record) { delta_.push_back(record); }

    void Delete(int id) {
        if (id < 0 || id >= kMaxIds) {
            throw std::runtime_error("host delete id out of range");
        }
        deleted_[id] = 1;
    }

    std::vector<Record> Active() const {
        std::vector<Record> active;
        for (const Record& record : base_) {
            if (deleted_.at(record.id) == 0) {
                active.push_back(record);
            }
        }
        for (const Record& record : delta_) {
            if (deleted_.at(record.id) == 0) {
                active.push_back(record);
            }
        }
        return active;
    }

    std::vector<Candidate> TopK(float qx, float qy, int k) const {
        std::vector<Candidate> result;
        for (const Record& record : Active()) {
            const float dx = record.x - qx;
            const float dy = record.y - qy;
            result.push_back(Candidate{record.id, dx * dx + dy * dy});
        }
        std::sort(result.begin(), result.end(), BetterHost);
        if (static_cast<int>(result.size()) > k) {
            result.resize(k);
        }
        return result;
    }

    int LiveDeltaCount() const {
        int count = 0;
        for (const Record& record : delta_) {
            if (deleted_.at(record.id) == 0) {
                ++count;
            }
        }
        return count;
    }

    void Rebuild() {
        base_ = Active();
        delta_.clear();
        deleted_.fill(0);
    }

private:
    std::vector<Record> base_;
    std::vector<Record> delta_;
    std::array<int, kMaxIds> deleted_{};
};

class ClosureRunner {
public:
    explicit ClosureRunner(int k) : k_(k) {
        Check(cudaGetDevice(&device_), "cudaGetDevice");
        Check(cudaMallocManaged(reinterpret_cast<void**>(&state_), sizeof(DeviceState)), "cudaMallocManaged(state)");
        Check(cudaMallocManaged(reinterpret_cast<void**>(&output_), sizeof(QueryOutput)), "cudaMallocManaged(output)");
    }

    ~ClosureRunner() {
        if (output_ != nullptr) {
            cudaFree(output_);
        }
        if (state_ != nullptr) {
            cudaFree(state_);
        }
    }

    void Reset(const std::vector<Record>& initial_base, int delta_capacity) {
        if (initial_base.size() > static_cast<size_t>(kMaxBase)) {
            throw std::runtime_error("initial Base exceeds fixed test capacity");
        }
        if (delta_capacity <= 0 || delta_capacity > kMaxDelta) {
            throw std::runtime_error("invalid Delta capacity");
        }
        Check(cudaMemset(state_, 0, sizeof(DeviceState)), "cudaMemset(state)");
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(reset)");
        state_->base_count = static_cast<int>(initial_base.size());
        state_->delta_capacity = delta_capacity;
        for (size_t i = 0; i < initial_base.size(); ++i) {
            state_->base[i] = initial_base[i];
        }
        model_.Reset(initial_base);
        // CUDA 13 changed cudaMemPrefetchAsync to a location-object API.  The
        // closure does not require prefetching for correctness; managed memory
        // migration is deliberately left to the runtime in this semantic stage.
    }

    void Insert(Record record) {
        InsertDelta<<<1, 1>>>(state_, record);
        SynchronizeAndCheck("InsertDelta");
        model_.Insert(record);
    }

    void Delete(int id) {
        DeleteId<<<1, 1>>>(state_, id);
        SynchronizeAndCheck("DeleteId");
        model_.Delete(id);
    }

    void Rebuild() {
        RebuildFrozenBase<<<1, 1>>>(state_);
        SynchronizeAndCheck("RebuildFrozenBase");
        model_.Rebuild();
    }

    void Query(float qx, float qy, const std::string& label) {
        QueryFrozenBasePlusLiveDelta<<<1, 1>>>(state_, qx, qy, k_, output_);
        SynchronizeAndCheck("QueryFrozenBasePlusLiveDelta");
        const std::vector<Candidate> expected = model_.TopK(qx, qy, k_);
        if (output_->count != static_cast<int>(expected.size())) {
            std::ostringstream os;
            os << "oracle mismatch at " << label << ": count gpu=" << output_->count
               << " cpu=" << expected.size();
            throw std::runtime_error(os.str());
        }
        for (int i = 0; i < output_->count; ++i) {
            if (output_->best[i].id != expected[i].id ||
                std::fabs(output_->best[i].d2 - expected[i].d2) > 1e-4F) {
                std::ostringstream os;
                os << "oracle mismatch at " << label << " rank=" << i
                   << " gpu=(id=" << output_->best[i].id << ",d2=" << output_->best[i].d2 << ")"
                   << " cpu=(id=" << expected[i].id << ",d2=" << expected[i].d2 << ")";
                throw std::runtime_error(os.str());
            }
        }
        ++query_count_;
    }

    int LiveDeltaSlots() const {
        int count = 0;
        for (int i = 0; i < state_->delta_capacity; ++i) {
            if (state_->delta_state[i] == 2) {
                ++count;
            }
        }
        return count;
    }

    int LiveDeltaCountHost() const { return model_.LiveDeltaCount(); }
    int QueryCount() const { return query_count_; }
    const HostModel& Model() const { return model_; }

private:
    void SynchronizeAndCheck(const char* label) {
        Check(cudaGetLastError(), label);
        Check(cudaDeviceSynchronize(), label);
        if (state_->failure_code != 0) {
            std::ostringstream os;
            os << label << " reported DeviceState failure_code=" << state_->failure_code;
            throw std::runtime_error(os.str());
        }
    }

    int k_;
    int device_ = 0;
    DeviceState* state_ = nullptr;
    QueryOutput* output_ = nullptr;
    HostModel model_;
    int query_count_ = 0;
};

Record MakeRecord(int id, int x, int y) {
    return Record{id, static_cast<float>(x), static_cast<float>(y)};
}

void RunDeterministicContract(ClosureRunner& runner) {
    // Capacity one exposes the key deletion rule: a deleted Delta record must
    // release live capacity so a subsequent insert can publish without rebuild.
    const std::vector<Record> initial = {
        MakeRecord(0, 0, 0), MakeRecord(1, 8, 0),
        MakeRecord(2, 0, 8), MakeRecord(3, -8, -8),
    };
    runner.Reset(initial, 1);
    runner.Query(1, 0, "deterministic/base");
    runner.Insert(MakeRecord(4, 1, 1));
    runner.Query(1, 1, "deterministic/live-delta");
    runner.Delete(4);  // delete a Delta record
    if (runner.LiveDeltaSlots() != 0 || runner.LiveDeltaCountHost() != 0) {
        throw std::runtime_error("deleted Delta record still consumes live capacity");
    }
    runner.Insert(MakeRecord(5, 2, 1));  // must reuse the released slot
    runner.Delete(1);                    // delete a frozen Base record
    runner.Query(2, 0, "deterministic/overlay-filter");
    runner.Rebuild();
    if (runner.LiveDeltaSlots() != 0 || runner.LiveDeltaCountHost() != 0) {
        throw std::runtime_error("rebuild did not clear transient Delta state");
    }
    runner.Query(2, 0, "deterministic/post-rebuild");
}

void RunRandomTrace(ClosureRunner& runner, int seed, int ops, int delta_capacity) {
    std::vector<Record> initial;
    for (int id = 0; id < 8; ++id) {
        initial.push_back(MakeRecord(id, id * 7 - 24, (id % 3) * 11 - 8));
    }
    runner.Reset(initial, delta_capacity);
    std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
    std::uniform_int_distribution<int> action(0, 99);
    std::uniform_int_distribution<int> coord(-100, 100);
    int next_id = 8;

    for (int step = 0; step < ops; ++step) {
        const int choice = action(gen);
        if (choice < 42 && next_id < kMaxIds) {
            if (runner.LiveDeltaCountHost() >= delta_capacity) {
                runner.Rebuild();
            }
            runner.Insert(MakeRecord(next_id++, coord(gen), coord(gen)));
        } else if (choice < 62) {
            const std::vector<Record> active = runner.Model().Active();
            if (active.empty()) {
                runner.Query(static_cast<float>(coord(gen)), static_cast<float>(coord(gen)),
                             "random-empty-" + std::to_string(seed) + "-" + std::to_string(step));
            } else {
                std::uniform_int_distribution<size_t> pick(0, active.size() - 1);
                runner.Delete(active[pick(gen)].id);
            }
        } else if (choice < 92) {
            runner.Query(static_cast<float>(coord(gen)), static_cast<float>(coord(gen)),
                         "random-query-" + std::to_string(seed) + "-" + std::to_string(step));
        } else {
            runner.Rebuild();
        }
    }
    runner.Query(0.0F, 0.0F, "random-final-" + std::to_string(seed));
}

int ParsePositive(const char* flag, const char* value) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("invalid value for ") + flag + ": " + value);
    }
    return static_cast<int>(parsed);
}

Config ParseArgs(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string flag(argv[i]);
        if (flag == "--help") {
            std::cout << "Usage: C1MinimalClosure [--seeds N] [--ops N] [--delta-capacity N] [--k N] [--receipt PATH]\n";
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("missing value for " + flag);
        }
        const char* value = argv[++i];
        if (flag == "--seeds") {
            config.seeds = ParsePositive("--seeds", value);
        } else if (flag == "--ops") {
            config.ops = ParsePositive("--ops", value);
        } else if (flag == "--delta-capacity") {
            config.delta_capacity = ParsePositive("--delta-capacity", value);
        } else if (flag == "--k") {
            config.k = ParsePositive("--k", value);
        } else if (flag == "--receipt") {
            config.receipt_path = value;
        } else {
            throw std::runtime_error("unknown argument " + flag);
        }
    }
    if (config.delta_capacity > kMaxDelta || config.k > kMaxK) {
        throw std::runtime_error("capacity or k exceeds compiled harness bound");
    }
    return config;
}

std::string EscapeJson(const std::string& text) {
    std::string out;
    for (const char ch : text) {
        if (ch == '"' || ch == '\\') {
            out.push_back('\\');
        }
        out.push_back(ch);
    }
    return out;
}

void WriteSummary(const Config& config, int queries, const cudaDeviceProp& prop) {
    if (config.receipt_path.empty()) {
        return;
    }
    std::ofstream output(config.receipt_path);
    if (!output) {
        throw std::runtime_error("cannot write summary to " + config.receipt_path);
    }
    output << "{\n"
           << "  \"schema\": \"tide-c1-minimal-gpu-v1\",\n"
           << "  \"status\": \"pass\",\n"
           << "  \"scope\": \"device-side Base+live-Delta+delete-overlay+stable-merge+rebuild semantics; frozen Base uses an exact scan adapter, not native GTS traversal\",\n"
           << "  \"c2_enabled\": false,\n"
           << "  \"seeds\": " << config.seeds << ",\n"
           << "  \"ops_per_seed\": " << config.ops << ",\n"
           << "  \"queries_checked\": " << queries << ",\n"
           << "  \"delta_capacity\": " << config.delta_capacity << ",\n"
           << "  \"k\": " << config.k << ",\n"
           << "  \"gpu_name\": \"" << EscapeJson(prop.name) << "\",\n"
           << "  \"compute_capability\": \"" << prop.major << "." << prop.minor << "\"\n"
           << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config config = ParseArgs(argc, argv);
        int device = 0;
        Check(cudaSetDevice(device), "cudaSetDevice");
        cudaDeviceProp prop{};
        Check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

        ClosureRunner runner(config.k);
        RunDeterministicContract(runner);
        for (int seed = 0; seed < config.seeds; ++seed) {
            RunRandomTrace(runner, seed, config.ops, config.delta_capacity);
        }
        WriteSummary(config, runner.QueryCount(), prop);
        std::cout << "C1 minimal GPU semantic closure PASS: queries=" << runner.QueryCount()
                  << ", seeds=" << config.seeds << ", ops=" << config.ops
                  << ", C2=off, gpu=" << prop.name << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "C1 minimal GPU semantic closure FAIL: " << error.what() << "\n";
        return 1;
    }
}
