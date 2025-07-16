DiffuLLaMA/
├── 📁 assets/                                    # 🎨 **Tài nguyên hình ảnh và media**
│   ├── logo.png                                 # Logo chính của project
│   ├── overview.png                             # Sơ đồ tổng quan về phương pháp adaptation
│   └── poster.png                               # Poster của paper tại ICLR 2025
│
├── 📁 DiffuLLaMA-training/                      # 🎯 **Training LLaMA2 → DiffuLLaMA** (Core training cho mô hình lớn)
│   ├── 📁 accelerate_configs/                   # ⚙️ Cấu hình training phân tán và tối ưu hóa
│   │   ├── deepspeed_inference.yaml             # Config DeepSpeed cho inference
│   │   ├── multi_node.yaml                     # Config training đa node (cluster)
│   │   ├── single_node.yaml                    # Config training đơn node
│   │   ├── two_node.yaml                       # Config training 2 node
│   │   ├── zero3_offload_inference.json         # ZeRO Stage 3 + offloading cho inference
│   │   ├── zero3_offload.json                  # ZeRO Stage 3 + CPU offloading
│   │   └── zero3.json                          # ZeRO Stage 3 cơ bản
│   ├── 📁 easy_context/                         # 🔧 **Tối ưu hóa attention cho sequence dài**
│   │   ├── __init__.py                         # Module initialization
│   │   ├── 📁 dist_flash_attn/                 # Flash Attention phân tán
│   │   │   ├── async_communication.py          # Communication bất đồng bộ giữa các GPU
│   │   │   ├── lightseq_async_attn_varlen.py   # LightSeq attention với variable length
│   │   │   ├── lightseq_async_attn.py          # LightSeq attention core
│   │   │   ├── monkey_patch.py                 # Patch các function attention
│   │   │   ├── prepare_input.py                # Chuẩn bị input cho attention
│   │   │   └── README.md                       # Hướng dẫn sử dụng
│   │   ├── 📁 ulysses_attn/                    # Ulysses Attention pattern
│   │   │   ├── monkey_patch.py                 # Patch attention theo Ulysses pattern
│   │   │   └── prepare_inputs.py               # Chuẩn bị input cho Ulysses
│   │   ├── 📁 unsloth_offloaded_gradient_checkpoint/ # Gradient checkpointing tối ưu
│   │   │   └── monkey_patch.py                 # Patch gradient checkpointing
│   │   └── 📁 zigzag_ring_attn/                # ZigZag Ring Attention
│   │       ├── monkey_patch.py                 # Patch ZigZag Ring Attention
│   │       └── prepare_inputs.py               # Chuẩn bị input cho ZigZag
│   ├── hostname.txt                            # Danh sách hostname các node trong cluster
│   ├── model_llama.py                          # 🏗️ **Định nghĩa DiffuLLaMA model architecture**
│   ├── multi_node.sh                           # 🚀 Script chạy training đa node
│   ├── packed_dataset.py                       # 📦 **Xử lý và pack dataset cho training**
│   ├── pip_freeze.txt                          # Danh sách packages và versions
│   ├── README.md                               # Hướng dẫn training LLaMA
│   ├── requirements.txt                        # Dependencies cho training
│   ├── run_distributed.sh                      # 🚀 Script chạy training phân tán
│   └── train.py                                # 🎯 **Script training chính cho LLaMA**
│
├── 📁 evaluation/                               # 📊 **Hệ thống đánh giá và benchmark**
│   ├── attention_patch.py                      # Patch attention cho evaluation
│   ├── 📁 baselines/                           # Các mô hình baseline để so sánh
│   │   ├── Plaid_sample.py                     # Implementation baseline Plaid
│   │   └── SEDD_run_sample_cond.py             # Implementation baseline SEDD
│   ├── eval-diffugpt.py                        # 🧪 **Script đánh giá DiffuGPT**
│   ├── eval-diffullama.py                      # 🧪 **Script đánh giá DiffuLLaMA**
│   ├── eval-llm.py                             # 🧪 **Script đánh giá LLM thông thường**
│   ├── 📁 evaluation/                          # Dataset và data cho evaluation
│   │   ├── cloze_test_val__spring2016.csv      # Dataset Cloze test
│   │   └── lambada_test_plain_text.txt         # Dataset LAMBADA test
│   ├── f1.py                                   # 📈 Tính toán F1 score
│   ├── model.py                                # Wrapper model cho evaluation
│   └── README.md                               # Hướng dẫn evaluation
│
├── 📁 example_output/                           # 📋 **Kết quả mẫu và demo**
│   ├── DiffuGPT-m-GSM-results.jsonl            # Kết quả DiffuGPT-medium trên GSM8K
│   └── DiffuLLaMA-GSM-results.jsonl            # Kết quả DiffuLLaMA trên GSM8K
│
├── 📁 LLaMA-Factory/                            # 🏭 **Framework training GPT2 → DiffuGPT & LoRA fine-tuning**
│   ├── 📁 data/                                # 📊 **Quản lý và xử lý dữ liệu**
│   │   ├── c4_demo.json                        # Demo dataset C4
│   │   ├── data_prepare.py                     # 🔧 **Script chuẩn bị và tiền xử lý dữ liệu**
│   │   ├── dataset_info.json                   # 📋 **Registry tất cả datasets có sẵn**
│   │   ├── gsm_test.json                       # Dataset test GSM8K
│   │   ├── gsm.json                            # Dataset train GSM8K
│   │   ├── README.md                           # Hướng dẫn quản lý data
│   │   └── wiki_demo.txt                       # Demo dataset Wikipedia
│   ├── 📁 evaluation/                          # Data cho đánh giá
│   │   ├── cloze_test_val__spring2016.csv      # Cloze test validation
│   │   ├── lambada_test_plain_text.txt         # LAMBADA test data
│   │   └── 📁 mmlu/                            # MMLU benchmark
│   │       ├── mapping.json                    # Mapping categories MMLU
│   │       ├── mmlu.py                         # Script chạy MMLU evaluation
│   │       └── mmlu.zip                        # Full MMLU dataset
│   ├── 📁 examples/                            # ⚙️ **Tất cả config files cho training/inference**
│   │   ├── 📁 accelerate/                      # Config Accelerate
│   │   │   └── fsdp_config.yaml                # FSDP configuration
│   │   ├── 📁 deepspeed/                       # Config DeepSpeed
│   │   │   ├── ds_z0_config.json               # DeepSpeed ZeRO Stage 0
│   │   │   ├── ds_z2_config.json               # DeepSpeed ZeRO Stage 2
│   │   │   ├── ds_z2_offload_config.json       # ZeRO Stage 2 + offloading
│   │   │   ├── ds_z3_config.json               # DeepSpeed ZeRO Stage 3
│   │   │   └── ds_z3_offload_config.json       # ZeRO Stage 3 + offloading
│   │   ├── 📁 extras/                          # Các phương pháp training đặc biệt
│   │   │   ├── 📁 adam_mini/                   # Adam-mini optimizer
│   │   │   ├── 📁 badam/                       # BAdam optimizer
│   │   │   ├── 📁 fsdp_qlora/                  # FSDP + QLoRA
│   │   │   ├── 📁 galore/                      # GaLore algorithm
│   │   │   ├── 📁 llama_pro/                   # LLaMA Pro expansion
│   │   │   ├── 📁 loraplus/                    # LoRA+ algorithm
│   │   │   ├── 📁 mod/                         # Mixture-of-Depths
│   │   │   └── 📁 pissa/                       # PiSSA algorithm
│   │   ├── 📁 inference/                       # 🎯 **Config inference cho diffusion models**
│   │   │   ├── gpt2_full_ddm-gsm-inf.yaml      # Inference DiffuGPT trên GSM8K
│   │   │   ├── gpt2_full_ddm-inf.yaml          # Inference DiffuGPT unconditional
│   │   │   ├── gpt2_full_ddm-small-inf-512.yaml # Inference DiffuGPT-small
│   │   │   ├── llama2_full_ddm-gsm-inf.yaml    # Inference DiffuLLaMA trên GSM8K
│   │   │   └── llama2_full_ddm-inf-uncon.yaml  # Inference DiffuLLaMA unconditional
│   │   ├── README.md                           # Hướng dẫn sử dụng configs
│   │   ├── 📁 train_full/                      # 🎯 **Config training full diffusion models**
│   │   │   ├── gpt2_full_ddm-sft.yaml          # Fine-tuning DiffuGPT
│   │   │   ├── gpt2_full_ddm-small.yaml        # Training DiffuGPT-small
│   │   │   ├── gpt2_full_ddm.yaml              # 🚀 **Training DiffuGPT chính**
│   │   │   └── gpt2_preprocess.yaml            # Tiền xử lý data cho GPT2
│   │   └── 📁 train_lora/                      # 🎯 **Config LoRA training**
│   │       ├── llama2_lora_ar-sft-gsm.yaml     # LoRA AR fine-tuning trên GSM8K
│   │       └── llama2_lora_ddm-sft.yaml        # 🚀 **LoRA diffusion fine-tuning**
│   ├── LICENSE                                 # Apache 2.0 License
│   ├── Makefile                                # Build automation
│   ├── MANIFEST.in                             # Package manifest
│   ├── pyproject.toml                          # Python project config
│   ├── README.md                               # LLaMA Factory documentation
│   ├── requirements.txt                        # Dependencies
│   ├── run_example.sh                          # Script chạy examples
│   ├── 📁 scripts/                             # Utility scripts
│   │   ├── cal_flops.py                        # Tính toán FLOPs
│   │   ├── cal_lr.py                           # Tính toán learning rate
│   │   ├── cal_ppl.py                          # Tính toán perplexity
│   │   ├── length_cdf.py                       # Phân tích độ dài sequence
│   │   ├── llama_pro.py                        # LLaMA Pro utilities
│   │   ├── llamafy_baichuan2.py                # Convert Baichuan2 to LLaMA format
│   │   ├── llamafy_qwen.py                     # Convert Qwen to LLaMA format
│   │   ├── loftq_init.py                       # LoftQ initialization
│   │   ├── pissa_init.py                       # PiSSA initialization
│   │   └── test_toolcall.py                    # Test tool calling
│   ├── setup.py                                # Package setup
│   ├── 📁 src/                                 # 🧠 **Core source code**
│   │   ├── api.py                              # API entry point
│   │   ├── 📁 llamafactory/                    # 🏭 **Main framework**
│   │   │   ├── __init__.py                     # Package initialization
│   │   │   ├── 📁 api/                         # API components
│   │   │   │   ├── __init__.py
│   │   │   │   ├── app.py                      # FastAPI application
│   │   │   │   ├── chat.py                     # Chat API endpoints
│   │   │   │   ├── common.py                   # Common API utilities
│   │   │   │   └── protocol.py                 # API protocol definitions
│   │   │   ├── attention_patch.py              # 🔧 **Patch attention mask cho diffusion**
│   │   │   ├── 📁 chat/                        # Chat engines
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base_engine.py              # Base chat engine
│   │   │   │   ├── chat_model.py               # Chat model wrapper
│   │   │   │   ├── hf_engine.py                # HuggingFace engine
│   │   │   │   └── vllm_engine.py              # vLLM engine
│   │   │   ├── cli.py                          # Command line interface
│   │   │   ├── 📁 data/                        # 📊 **Data processing pipeline**
│   │   │   │   ├── __init__.py
│   │   │   │   ├── aligner.py                  # Data alignment
│   │   │   │   ├── collator.py                 # Data collation cho batch
│   │   │   │   ├── data_utils.py               # Data utilities
│   │   │   │   ├── formatter.py                # Data formatting
│   │   │   │   ├── loader.py                   # 📥 **Data loading cho training**
│   │   │   │   ├── mm_plugin.py                # Multimodal plugin
│   │   │   │   ├── parser.py                   # Data parsing
│   │   │   │   ├── preprocess.py               # Data preprocessing
│   │   │   │   ├── 📁 processors/              # Data processors cho các loại training
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── feedback.py             # Feedback data processing
│   │   │   │   │   ├── pairwise.py             # Pairwise data processing
│   │   │   │   │   ├── pretrain.py             # Pre-training data processing
│   │   │   │   │   ├── processor_utils.py      # Processor utilities
│   │   │   │   │   ├── supervised.py           # Supervised fine-tuning data
│   │   │   │   │   └── unsupervised.py         # Unsupervised data processing
│   │   │   │   ├── template.py                 # Prompt templates
│   │   │   │   └── tool_utils.py               # Tool utilities
│   │   │   ├── 📁 eval/                        # Evaluation framework
│   │   │   │   ├── __init__.py
│   │   │   │   ├── evaluator.py                # Model evaluator
│   │   │   │   └── template.py                 # Evaluation templates
│   │   │   ├── 📁 extras/                      # Extra utilities
│   │   │   │   ├── __init__.py
│   │   │   │   ├── constants.py                # Constants and configs
│   │   │   │   ├── env.py                      # Environment setup
│   │   │   │   ├── logging.py                  # Logging utilities
│   │   │   │   ├── misc.py                     # Miscellaneous utilities
│   │   │   │   ├── packages.py                 # Package management
│   │   │   │   └── ploting.py                  # Plotting utilities
│   │   │   ├── 📁 hparams/                     # Hyperparameters
│   │   │   │   ├── __init__.py
│   │   │   │   ├── data_args.py                # Data arguments
│   │   │   │   ├── evaluation_args.py          # Evaluation arguments
│   │   │   │   ├── finetuning_args.py          # Fine-tuning arguments
│   │   │   │   ├── generating_args.py          # Generation arguments
│   │   │   │   ├── model_args.py               # Model arguments
│   │   │   │   └── parser.py                   # Argument parser
│   │   │   ├── launcher.py                     # Main launcher
│   │   │   ├── 📁 model/                       # 🏗️ **Model management**
│   │   │   │   ├── __init__.py
│   │   │   │   ├── adapter.py                  # Model adapters
│   │   │   │   ├── loader.py                   # 🔄 **Model loading cho diffusion**
│   │   │   │   ├── 📁 model_utils/             # Model utilities
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── attention.py            # Attention utilities
│   │   │   │   │   ├── checkpointing.py        # Gradient checkpointing
│   │   │   │   │   ├── embedding.py            # Embedding utilities
│   │   │   │   │   ├── liger_kernel.py         # Liger kernel support
│   │   │   │   │   ├── longlora.py             # LongLoRA support
│   │   │   │   │   ├── misc.py                 # Miscellaneous model utils
│   │   │   │   │   ├── mod.py                  # Mixture-of-Depths
│   │   │   │   │   ├── moe.py                  # Mixture-of-Experts
│   │   │   │   │   ├── packing.py              # Sequence packing
│   │   │   │   │   ├── quantization.py         # Quantization support
│   │   │   │   │   ├── rope.py                 # RoPE utilities
│   │   │   │   │   ├── unsloth.py              # Unsloth optimization
│   │   │   │   │   ├── valuehead.py            # Value head for RL
│   │   │   │   │   └── visual.py               # Visual model support
│   │   │   │   └── patcher.py                  # Model patching
│   │   │   ├── 📁 train/                       # 🎯 **Training algorithms**
│   │   │   │   ├── __init__.py
│   │   │   │   ├── callbacks.py                # Training callbacks
│   │   │   │   ├── 📁 ddm/                     # 🎯 **MAIN: Diffusion training module**
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── metric.py               # Diffusion metrics
│   │   │   │   │   ├── model.py                # 🧠 **Diffusion model wrapper**
│   │   │   │   │   ├── trainer.py              # 🏃 **Diffusion trainer chính**
│   │   │   │   │   └── workflow.py             # 🔄 **Quy trình training diffusion**
│   │   │   │   ├── 📁 dpo/                     # DPO training
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── trainer.py              # DPO trainer
│   │   │   │   │   └── workflow.py             # DPO workflow
│   │   │   │   ├── 📁 kto/                     # KTO training
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── trainer.py              # KTO trainer
│   │   │   │   │   └── workflow.py             # KTO workflow
│   │   │   │   ├── 📁 ppo/                     # PPO training
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── ppo_utils.py            # PPO utilities
│   │   │   │   │   ├── trainer.py              # PPO trainer
│   │   │   │   │   └── workflow.py             # PPO workflow
│   │   │   │   ├── 📁 pt/                      # Pre-training
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── trainer.py              # Pre-training trainer
│   │   │   │   │   └── workflow.py             # Pre-training workflow
│   │   │   │   ├── 📁 rm/                      # Reward modeling
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── metric.py               # RM metrics
│   │   │   │   │   ├── trainer.py              # RM trainer
│   │   │   │   │   └── workflow.py             # RM workflow
│   │   │   │   ├── 📁 sft/                     # Supervised fine-tuning
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── metric.py               # SFT metrics
│   │   │   │   │   ├── trainer.py              # SFT trainer
│   │   │   │   │   └── workflow.py             # SFT workflow
│   │   │   │   ├── test_utils.py               # Testing utilities
│   │   │   │   ├── trainer_utils.py            # Trainer utilities
│   │   │   │   └── tuner.py                    # Main tuner
│   │   │   └── 📁 webui/                       # Web interface
│   │   │       ├── __init__.py
│   │   │       ├── chatter.py                  # Chat interface
│   │   │       ├── common.py                   # Common UI components
│   │   │       ├── 📁 components/              # UI components
│   │   │       │   ├── __init__.py
│   │   │       │   ├── chatbot.py              # Chatbot component
│   │   │       │   ├── data.py                 # Data management UI
│   │   │       │   ├── eval.py                 # Evaluation UI
│   │   │       │   ├── export.py               # Export UI
│   │   │       │   ├── infer.py                # Inference UI
│   │   │       │   ├── top.py                  # Top UI components
│   │   │       │   └── train.py                # Training UI
│   │   │       ├── css.py                      # CSS styles
│   │   │       ├── engine.py                   # UI engine
│   │   │       ├── interface.py                # Main interface
│   │   │       ├── locales.py                  # Internationalization
│   │   │       ├── manager.py                  # UI manager
│   │   │       ├── runner.py                   # UI runner
│   │   │       └── utils.py                    # UI utilities
│   │   ├── train.py                            # Training entry point
│   │   └── webui.py                            # Web UI entry point
│   └── 📁 tests/                               # 🧪 **Test suite**
│       ├── 📁 data/                            # Data tests
│       │   ├── 📁 processors/                  # Processor tests
│       │   │   ├── test_feedback.py            # Test feedback processor
│       │   │   ├── test_pairwise.py            # Test pairwise processor
│       │   │   ├── test_processor_utils.py     # Test processor utils
│       │   │   ├── test_supervised.py          # Test supervised processor
│       │   │   └── test_unsupervised.py        # Test unsupervised processor
│       │   ├── test_collator.py                # Test data collator
│       │   ├── test_formatter.py               # Test data formatter
│       │   ├── test_mm_plugin.py               # Test multimodal plugin
│       │   └── test_template.py                # Test templates
│       ├── 📁 eval/                            # Evaluation tests
│       │   └── test_eval_template.py           # Test evaluation template
│       └── 📁 model/                           # Model tests
│           ├── 📁 model_utils/                 # Model utility tests
│           │   ├── test_attention.py           # Test attention utils
│           │   ├── test_checkpointing.py       # Test checkpointing
│           │   └── test_packing.py             # Test sequence packing
│           ├── test_base.py                    # Test base model
│           ├── test_freeze.py                  # Test freeze training
│           ├── test_full.py                    # Test full training
│           ├── test_lora.py                    # Test LoRA training
│           └── test_pissa.py                   # Test PiSSA training
│
├── 🐍 **Scripts inference nhanh**
├── attention_patch.py                          # 🔧 **Patch attention mask cho diffusion** (global)
├── inf_diffugpt.py                            # 🚀 **Quick start inference DiffuGPT**
├── inf_diffullama.py                          # 🚀 **Quick start inference DiffuLLaMA**
├── model.py                                   # 🏗️ **Core DiscreteDiffusionModel definition**
│
├── 📚 **Tài liệu và hướng dẫn**
├── graph-builder.md                           # Hướng dẫn xây dựng graph (có thể cho visualization)
├── how-to-fine-tune.md                        # 📖 **Hướng dẫn chi tiết fine-tuning**
├── project-explain.md                         # 📋 **Giải thích chi tiết project**
└── README.md                                  # 📋 **Documentation chính và quick start**