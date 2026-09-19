// Compile-only probe. main does not invoke a CUDA API or kernel.
#define RP_DEFINE_CONSTANTS
#include "residual_pruning.cuh"
#include "safe_c1_gts_topk_exporter.cuh"

int main() {
  safe_c1_exporter::FrozenGtsBase base{};
  safe_c1_exporter::FullVectorPoolDevice pool{};
  safe_c1_exporter::StaticBaseTopKGate gate{};
  safe_c1_exporter::ExtraTierIds extras{};
  safe_c1_exporter::BaseDeleteBarrier barrier{};
  (void)base;
  (void)pool;
  (void)gate;
  (void)extras;
  (void)barrier;
  return 0;
}
