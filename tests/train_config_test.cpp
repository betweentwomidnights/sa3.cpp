#include "train_config.h"

#include <cstdio>
#include <fstream>
#include <string>

static int expect(bool ok, const char* msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        return 1;
    }
    return 0;
}

static char* arg(char* s) { return s; }

int main() {
    int fails = 0;
    {
        sa3::TrainConfig c;
        std::string err;
        char a0[] = "test";
        char a1[] = "--rank";
        char a2[] = "16";
        char a3[] = "--learning-rate=0.001";
        char a4[] = "--adapter-type";
        char a5[] = "dora-rows";
        char a6[] = "--prompt-mode";
        char a7[] = "caption-lyrics";
        char* argv[] = {arg(a0), arg(a1), arg(a2), arg(a3), arg(a4), arg(a5), arg(a6), arg(a7)};
        fails += expect(sa3::train_parse_args(8, argv, c, err), "CLI parse succeeds");
        fails += expect(c.rank == 16, "rank override");
        fails += expect(c.learning_rate > 0.00099f && c.learning_rate < 0.00101f, "learning rate override");
        fails += expect(c.adapter_type == "dora-rows", "adapter override");
        fails += expect(c.prompt_mode == "caption-lyrics", "prompt mode override");
        fails += expect(c.prompt_config_path.empty(), "prompt_config default empty");
        fails += expect(sa3::validate_train_config(c, err), "validated CLI config");
    }
    {
        const char* path = "train_config_test.json";
        {
            std::ofstream f(path);
            f << "{"
              << "\"dataset_dir\":\"../data\","
              << "\"output_dir\":\"runs/x\","
              << "\"rank\":4,"
              << "\"alpha\":12.5,"
              << "\"batch_size\":2,"
              << "\"adapter_type\":\"bora\""
              << "}";
        }
        sa3::TrainConfig c;
        std::string err;
        char a0[] = "test";
        char a1[] = "--config";
        char a2[] = "train_config_test.json";
        char a3[] = "--rank";
        char a4[] = "6";
        char* argv[] = {arg(a0), arg(a1), arg(a2), arg(a3), arg(a4)};
        fails += expect(sa3::train_parse_args(5, argv, c, err), "JSON plus CLI parse succeeds");
        fails += expect(c.dataset_dir == "../data", "JSON dataset");
        fails += expect(c.output_dir == "runs/x", "JSON output");
        fails += expect(c.rank == 6, "CLI overrides JSON");
        fails += expect(c.alpha > 12.49f && c.alpha < 12.51f, "JSON alpha");
        fails += expect(c.batch_size == 2, "JSON batch size");
        fails += expect(c.adapter_type == "bora", "JSON adapter");
        std::remove(path);
    }
    {
        sa3::TrainConfig c;
        std::string err;
        char a0[] = "test";
        char a1[] = "--unknown";
        char a2[] = "x";
        char* argv[] = {arg(a0), arg(a1), arg(a2)};
        fails += expect(!sa3::train_parse_args(3, argv, c, err), "unknown option rejected");
        fails += expect(err.find("unknown") != std::string::npos, "unknown option error text");
    }
    {
        sa3::TrainConfig c;
        c.adapter_type = "bad";
        std::string err;
        fails += expect(!sa3::validate_train_config(c, err), "bad adapter rejected");
    }
    {
        sa3::TrainConfig c;
        c.prompt_mode = "bad";
        std::string err;
        fails += expect(!sa3::validate_train_config(c, err), "bad prompt mode rejected");
    }
    {
        // Stage 9/10 defaults + parsing: dist-shift canonicalization (case-insensitive) and cfg-dropout.
        sa3::TrainConfig def;
        fails += expect(def.timestep_sampler == "trunc_logit_normal", "default timestep sampler trunc_logit_normal");
        fails += expect(def.dist_shift == "Full", "default dist_shift Full");
        fails += expect(def.dist_shift_effective_length, "default effective-length true");
        fails += expect(def.cfg_dropout_prob > 0.099f && def.cfg_dropout_prob < 0.101f, "default cfg_dropout 0.1");

        sa3::TrainConfig c;
        std::string err;
        char a0[] = "test";
        char a1[] = "--dist-shift";
        char a2[] = "logsnr";
        char a3[] = "--dist-shift-effective-length=false";
        char a4[] = "--cfg-dropout-prob";
        char a5[] = "0.0";
        char* argv[] = {arg(a0), arg(a1), arg(a2), arg(a3), arg(a4), arg(a5)};
        fails += expect(sa3::train_parse_args(6, argv, c, err), "schedule CLI parse succeeds");
        fails += expect(c.dist_shift == "LogSNR", "dist_shift canonicalized to LogSNR");
        fails += expect(!c.dist_shift_effective_length, "effective-length overridden false");
        fails += expect(c.cfg_dropout_prob == 0.0f, "cfg_dropout overridden 0");
        fails += expect(sa3::validate_train_config(c, err), "validated schedule config");
    }
    {
        sa3::TrainConfig c;
        std::string err;
        char a0[] = "test";
        char a1[] = "--dist-shift";
        char a2[] = "bogus";
        char* argv[] = {arg(a0), arg(a1), arg(a2)};
        fails += expect(!sa3::train_parse_args(3, argv, c, err), "bad dist_shift rejected");
    }
    {
        sa3::TrainConfig c;
        c.cfg_dropout_prob = 1.5f;
        std::string err;
        fails += expect(!sa3::validate_train_config(c, err), "out-of-range cfg_dropout rejected");
    }
    {
        // Stage 11: lr scheduler defaults + parsing.
        sa3::TrainConfig def;
        fails += expect(def.lr_scheduler == "constant", "default scheduler constant");
        fails += expect(def.lr_inv_gamma > 999999.0f && def.lr_inv_gamma < 1000001.0f, "default inv_gamma 1e6");
        fails += expect(def.lr_warmup > 0.9949f && def.lr_warmup < 0.9951f, "default warmup 0.995");

        sa3::TrainConfig c;
        std::string err;
        char a0[] = "test";
        char a1[] = "--lr-scheduler";
        char a2[] = "inverse_lr";
        char a3[] = "--lr-power=1.0";
        char a4[] = "--prompt-config";
        char a5[] = "ds.json";
        char* argv[] = {arg(a0), arg(a1), arg(a2), arg(a3), arg(a4), arg(a5)};
        fails += expect(sa3::train_parse_args(6, argv, c, err), "lr scheduler CLI parse");
        fails += expect(c.lr_scheduler == "inverse_lr", "scheduler overridden");
        fails += expect(c.prompt_config_path == "ds.json", "prompt_config path parsed");
        fails += expect(c.lr_power > 0.999f && c.lr_power < 1.001f, "lr_power overridden");
        fails += expect(sa3::validate_train_config(c, err), "validated scheduler config");
    }
    {
        sa3::TrainConfig c;
        c.lr_scheduler = "inverse_lr";
        c.lr_warmup = 1.0f;   // out of [0,1)
        std::string err;
        fails += expect(!sa3::validate_train_config(c, err), "bad lr_warmup rejected");
    }
    if (fails) return 1;
    std::printf("train_config_test: ok\n");
    return 0;
}
