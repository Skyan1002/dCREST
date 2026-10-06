// Parity harness for dcrest's CRESTPhys kernel: EF5's own
// CRESTPHYSModel::WaterBalanceInt, included VERBATIM through body.inc, which
// scripts/verify_ef5_partition.py extracts from the EF5 source on every run.
// Lumped cell: no routed-in baseflow (the gridded EF5 adds it; dcrest is lumped).
#include <cmath>
#include <cstdio>
#include <vector>
enum { PARAM_CRESTPHYS_WM, PARAM_CRESTPHYS_B, PARAM_CRESTPHYS_IM, PARAM_CRESTPHYS_KE,
       PARAM_CRESTPHYS_FC, PARAM_CRESTPHYS_KSOIL, PARAM_CRESTPHYS_IWU, PARAM_CRESTPHYS_IGW,
       PARAM_CRESTPHYS_HMAXAQ, PARAM_CRESTPHYS_GWC, PARAM_CRESTPHYS_GWE, PARAM_CRESTPHYS_QTY };
enum { STATE_CRESTPHYS_SM, STATE_CRESTPHYS_GW, STATE_CRESTPHYS_QTY };
enum { CRESTPHYS_LAYER_OVERLAND, CRESTPHYS_LAYER_INTERFLOW, CRESTPHYS_LAYER_BASEFLOW, CRESTPHYS_LAYER_QTY };
struct GridNode {};
struct CRESTPHYSGridNode { float params[PARAM_CRESTPHYS_QTY]; float states[STATE_CRESTPHYS_QTY];
  double excess[CRESTPHYS_LAYER_QTY]; double actET; };
struct CRESTPHYSModel { void WaterBalanceInt(GridNode *node, CRESTPHYSGridNode *cNode,
  float stepHours, float precipIn, float petIn, float *fastFlow, float *interFlow, float *baseFlow); };
#define printf(...) ((void)0)
#include "body.inc"
#undef printf
int main(int argc, char **argv) {
  FILE *in = fopen(argv[1], "r"); FILE *out = fopen(argv[2], "w");
  int N, T; float sh; fscanf(in, "%d %d %f", &N, &T, &sh);
  std::vector<CRESTPHYSGridNode> nd(N); CRESTPHYSModel m; GridNode g;
  for (int i = 0; i < N; i++) {
    for (int p = 0; p < PARAM_CRESTPHYS_QTY; p++) fscanf(in, "%f", &nd[i].params[p]);
    fscanf(in, "%f %f", &nd[i].states[0], &nd[i].states[1]);
  }
  for (int t = 0; t < T; t++) for (int i = 0; i < N; i++) {
    float p, e; fscanf(in, "%f %f", &p, &e);
    float ff = 0, itf = 0, bf = 0;   // no routed-in baseflow (lumped cell)
    m.WaterBalanceInt(&g, &nd[i], sh, p, e, &ff, &itf, &bf);
    fprintf(out, "%.9e %.9e %.9e %.9e %.9e\n", ff, itf, bf, nd[i].states[0], nd[i].states[1]);
  }
  fclose(in); fclose(out); return 0;
}
