#include "train_lora.h"

#include <cstdio>

static int expect(bool ok, const char* msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        return 1;
    }
    return 0;
}

int main() {
    int fails = 0;
    ggml_init_params ip = { 1024 * 1024, nullptr, false };
    ggml_context* ctx = ggml_init(ip);
    sa3::GgufModel m;
    ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8);
    ggml_set_name(a, "dit.0.self.qkv.weight");
    float* ad = (float*)a->data;
    for (int i = 0; i < 32; ++i) ad[i] = (float)(i + 1);
    ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_set_name(b, "dit.0.ff.out.bias");
    ggml_tensor* c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 5);
    ggml_set_name(c, "other.weight");
    m.tensors[a->name] = a;
    m.tensors[b->name] = b;
    m.tensors[c->name] = c;
    std::vector<sa3::TrainLoraTarget> targets = sa3::enumerate_train_lora_targets(m);
    fails += expect(targets.size() == 1, "one DiT 2D weight target");
    fails += expect(targets[0].stem == "dit.0.self.qkv", "target stem");
    fails += expect(targets[0].in == 4 && targets[0].out == 8, "target shape");
    sa3::TrainLoraState st;
    std::string err;
    fails += expect(sa3::init_train_lora_state(m, targets, "dora-rows", 2, 4.0f, 7, st, err), "init dora rows");
    fails += expect(st.params.size() == 1, "state target count");
    fails += expect(st.params[0].lora_A.size() == 8, "lora A shape");
    fails += expect(st.params[0].lora_B.size() == 16, "lora B shape");
    fails += expect(st.params[0].magnitude.size() == 8, "dora magnitude shape");
    fails += expect(st.params[0].magnitude[0] > 5.47f && st.params[0].magnitude[0] < 5.48f, "dora row norm");
    fails += expect(sa3::init_train_lora_state(m, targets, "lora-xs", 3, 3.0f, 7, st, err), "init xs");
    fails += expect(st.params[0].U.size() == 24 && st.params[0].V.size() == 12 && st.params[0].M_xs.size() == 9, "xs shapes");
    ggml_tensor* A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_tensor* B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 8);
    ggml_tensor* mag = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor* magc = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    sa3::TrainLoraGraphParam gp;
    gp.lora_A = A;
    gp.lora_B = B;
    gp.magnitude = mag;
    gp.magnitude_r = mag;
    gp.magnitude_c = magc;
    ggml_tensor* ew = sa3::train_lora_effective_weight(ctx, a, gp, "lora", 2, 2.0f);
    fails += expect(ew && ew->ne[0] == 4 && ew->ne[1] == 8, "lora effective graph shape");
    ew = sa3::train_lora_effective_weight(ctx, a, gp, "dora-rows", 2, 2.0f);
    fails += expect(ew && ew->ne[0] == 4 && ew->ne[1] == 8, "dora rows graph shape");
    gp.magnitude = magc;
    ew = sa3::train_lora_effective_weight(ctx, a, gp, "dora-cols", 2, 2.0f);
    fails += expect(ew && ew->ne[0] == 4 && ew->ne[1] == 8, "dora cols graph shape");
    ew = sa3::train_lora_effective_weight(ctx, a, gp, "bora", 2, 2.0f);
    fails += expect(ew && ew->ne[0] == 4 && ew->ne[1] == 8, "bora graph shape");
    ggml_free(ctx);
    if (fails) return 1;
    std::printf("train_lora_test: inventory ok\n");
    return 0;
}
