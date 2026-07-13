// CLI around the native engine: validate the forward pass against the golden
// per-layer dumps, or greedily generate from the reference prompt.
//   ./lbinfer <model_dir> [test_dir] [gen N]
#include "engine.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace lb;

int main(int argc, char** argv) {
  std::string mdir = argc > 1 ? argv[1] : "/model";
  std::string tdir = argc > 2 ? argv[2] : "test";
  int gen = 0;
  for (int i = 3; i < argc; ++i)
    if (std::string(argv[i]) == "gen" && i + 1 < argc) gen = atoi(argv[i + 1]);

  std::vector<int> ids;
  { FILE* f = fopen((tdir + "/ref_tokens.i32").c_str(), "rb");
    if (!f) { printf("need %s/ref_tokens.i32\n", tdir.c_str()); return 2; }
    int x; while (fread(&x, 4, 1, f) == 1) ids.push_back(x); fclose(f); }
  printf("prompt T=%zu:", ids.size()); for (int i : ids) printf(" %d", i); printf("\n");

  Engine eng; eng.load(mdir, ids.size() + (gen > 0 ? gen : 1) + 2);

  if (gen > 0) {
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
