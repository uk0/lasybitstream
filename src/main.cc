#include "qwen35.hpp"
#include "device.hpp"
#include <cstdio>
#include <exception>

int main(int argc, char** argv) {
  const char* dir = (argc > 1) ? argv[1] : "/home/neo/models/Qwen3.6-27B-NVFP4";
  printf("lasybitstream loader — Qwen3.6-27B-MTP\nmodel dir: %s\n\n", dir);

  lb::Qwen35Model m;
  try {
    m.load(dir);
  } catch (const std::exception& e) {
    fprintf(stderr, "LOAD FAILED: %s\n", e.what());
    return 2;
  }
  m.print_summary();

  printf("\n== device (GB10 / sm_121) ==\n");
  lb::DeviceInfo di = lb::device_info();
  if (di.ok)
    printf("  GPU: %s  sm_%d%d  %d SMs  %.0f GB unified\n", di.name, di.cc_major,
           di.cc_minor, di.sm_count, di.mem_gb);
  else
    printf("  (no CUDA device visible)\n");

  // Round-trip one real NVFP4 weight tensor to the device.
  const lb::QuantLinear* q = m.first_quant_linear();
  if (di.ok && q) {
    const lb::TensorInfo& ti = m.st.info(q->packed);
    double up = 0, kk = 0, gbps = 0;
    uint64_t cks = lb::device_checksum(m.st.data(q->packed), ti.nbytes(), &up, &kk, &gbps);
    printf("  upload %s [%ldx%ld] %.1f MB -> devsum=%llu  (H2D %.2f ms = %.0f GB/s, kernel %.3f ms)\n",
           q->packed.c_str(), (long)q->out, (long)q->in, ti.nbytes() / 1e6,
           (unsigned long long)cks, up, gbps, kk);
  }

  printf("\n== RESULT: %s ==\n",
         m.report.pass ? "PASS — Qwen3.6-27B-MTP loaded correctly"
                       : "FAIL — see errors above");
  return m.report.pass ? 0 : 1;
}
