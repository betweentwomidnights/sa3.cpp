sa3.cpp standalone Windows package
==================================

This folder contains sa3-server.exe, sat-server.exe, the training and CLI
tools, the CPU backend variants, CUDA and Vulkan backends, and the CUDA runtime.
Model weights are downloaded separately. BUILD-INFO.json records the source
commit and toolchain used to build the package.

1. Download a model from this folder. Examples:

     models.cmd --variant small-music --encoding q4_k_m
     models.cmd --sat --sat-model foundation-1 --encoding q4_k_m
     models.cmd --sat --sat-model saos --encoding q4_k_m

   models.cmd --help lists the tiers and finetunes. Model licenses and access
   terms remain with their publishers.

2. For the SA3 inference and training browser UI, run:

     studio.cmd

   Open http://127.0.0.1:8006/ . The training page is linked there.

3. For a native SAT HTTP service, run one process per model:

     sat-server.exe --model foundation-1 --models-dir models --encoding Q4_K_M --port 8015
     sat-server.exe --model arc --models-dir models --encoding Q4_K_M --port 8018

   SAT_SERVER.md documents the endpoints. Both servers support --version and
   --props without loading a model.

For direct SA3 commands, see RUNTIME_README.md. CUDA works when the NVIDIA
driver supports this toolkit; Vulkan is available for supported GPUs.
