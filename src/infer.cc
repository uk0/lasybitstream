// CLI around the native engine: validate the forward pass against the golden
// per-layer dumps, or greedily generate from the reference prompt.
//   ./lbinfer <model_dir> [test_dir] [gen N]
#include "engine.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

using namespace lb;

int main(int argc, char** argv) {
  std::string mdir = argc > 1 ? argv[1] : "/model";
  std::string tdir = argc > 2 ? argv[2] : "test";
  int gen = 0, spec = 0; bool mm = false, bench = false, decbench = false;
  for (int i = 3; i < argc; ++i) {
    if (std::string(argv[i]) == "gen" && i + 1 < argc) gen = atoi(argv[i + 1]);
    if (std::string(argv[i]) == "spec" && i + 1 < argc) spec = atoi(argv[i + 1]);  // k drafts
    if (std::string(argv[i]) == "mm") mm = true;
    if (std::string(argv[i]) == "bench") bench = true;
    if (std::string(argv[i]) == "decbench") decbench = true;
  }

  if (bench) {                                          // aggregate-batching throughput curve (GEMM-only)
    Engine eng; eng.load(mdir, 64);
    printf("=== aggregate throughput (weights read once per forward) ===\n");
    for (int M : {1, 32, 64, 128, 256, 512, 1024}) printf("  batch M=%4d : %.1f tok/s\n", M, eng.bench(M, 3));
    return 0;
  }

  if (decbench) {                                       // real aggregate decode (per-seq KV/GDN/conv state)
    Engine eng; eng.load(mdir, 640);
    printf("=== batched decode throughput (real per-seq state, ctx=256, 32 steps) ===\n");
    for (int B : {1, 8, 16, 32, 48, 64}) printf("  B=%3d : %.1f tok/s\n", B, eng.bench_decode(B, 256, 32));
    return 0;
  }

  if (mm) {  // multimodal: reference image (vis_pixels/grid) + prompt (vis_input_ids)
    std::vector<int> ids; std::vector<float> pix; int g[3] = {1, 0, 0};
    { FILE* f = fopen((tdir + "/vis_input_ids.i32").c_str(), "rb"); int x;
      while (f && fread(&x, 4, 1, f) == 1) ids.push_back(x); if (f) fclose(f); }
    { FILE* f = fopen((tdir + "/vis_grid.i32").c_str(), "rb"); if (f) { fread(g, 4, 3, f); fclose(f); } }
    { FILE* f = fopen((tdir + "/vis_pixels.f32").c_str(), "rb"); if (f) {
        fseek(f, 0, SEEK_END); long nb = ftell(f); fseek(f, 0, SEEK_SET);
        pix.resize(nb / 4); fread(pix.data(), 4, pix.size(), f); fclose(f); } }
    if (ids.empty() || pix.empty()) { printf("need vis_input_ids.i32 + vis_pixels.f32 + vis_grid.i32\n"); return 2; }
    printf("mm: image grid t=%d h=%d w=%d, prompt %zu tokens\n", g[0], g[1], g[2], ids.size());
    Engine eng; eng.load(mdir, ids.size() + 40);
    int merged = eng.encode_image(pix, g[0], g[1], g[2]);
    printf("encoded %d image tokens\n", merged);
    std::vector<int> o = eng.generate_mm(ids, 248056, g[1], g[2], gen > 0 ? gen : 24, 248046);
    std::vector<int> all = ids; for (int t : o) all.push_back(t);
    FILE* f = fopen((tdir + "/mm_gen_ids.i32").c_str(), "wb"); fwrite(o.data(), 4, o.size(), f); fclose(f);
    printf("mm generated:"); for (int t : o) printf(" %d", t); printf("\nwrote %s/mm_gen_ids.i32\n", tdir.c_str());
    return 0;
  }

  std::vector<int> ids;
  { FILE* f = fopen((tdir + "/ref_tokens.i32").c_str(), "rb");
    if (!f) { printf("need %s/ref_tokens.i32\n", tdir.c_str()); return 2; }
    int x; while (fread(&x, 4, 1, f) == 1) ids.push_back(x); fclose(f); }
  printf("prompt T=%zu:", ids.size()); for (int i : ids) printf(" %d", i); printf("\n");

  int want = gen > 0 ? gen : 24;
  Engine eng; eng.load(mdir, ids.size() + want + 12);

  if (spec > 0) {                                        // MTP speculative decode vs greedy
    std::vector<int> g = eng.generate(ids, want, 248046);
    std::vector<int> s = eng.generate_spec(ids, want, 248046, spec);
    size_t common = std::min(g.size(), s.size()); bool match = true;
    for (size_t i = 0; i < common; ++i) if (g[i] != s[i]) { match = false; break; }
    printf("spec k=%d: greedy %zu tok, spec %zu tok, prefix-match=%s\n",
           spec, g.size(), s.size(), match ? "YES (identical output)" : "NO — BUG");
    printf("spec ids:"); for (int t : s) printf(" %d", t); printf("\n");
  } else if (gen > 0) {
    std::vector<int> outs = eng.generate(ids, gen, -1);   // no eos stop; fixed count
    std::vector<int> all = ids; for (int t : outs) all.push_back(t);
    FILE* f = fopen((tdir + "/gen_ids.i32").c_str(), "wb");
    fwrite(all.data(), 4, all.size(), f); fclose(f);
    printf("generated ids:"); for (int t : all) printf(" %d", t);
    printf("\nwrote %s/gen_ids.i32\n", tdir.c_str());
  } else {
    int am = eng.validate(ids, tdir);
    printf("\nargmax = %d\n", am);
  }
  return 0;
}
