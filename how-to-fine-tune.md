Nếu bạn chỉ muốn **fine-tune mô hình** mà không cần training từ đầu, thì rất đơn giản! Project này đã cung cấp các mô hình pre-trained sẵn.

Read file: LLaMA-Factory/data/gsm_test.json

Perfect! Để **chỉ fine-tune mô hình** mà không cần training từ đầu, đây là hướng dẫn chi tiết:

## 🚀 **Cách Fine-tune DiffuGPT/DiffuLLaMA**

### **1. Mô hình Pre-trained có sẵn**
- **DiffuGPT**: `diffusionfamily/diffugpt-s`, `diffusionfamily/diffugpt-m`
- **DiffuLLaMA**: `diffusionfamily/diffullama`


### **2. Cài đặt môi trường**
```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install flash-attn==2.6.3 --no-build-isolation
```

### **3. Chuẩn bị dữ liệu**
Dữ liệu cần format JSON như sau:
```json
[
    {
        "instruction": "Giải bài toán này: 2+2 bằng bao nhiêu?",
        "output": "2+2 = 4"
    },
    {
        "instruction": "Câu hỏi khác...",
        "output": "Câu trả lời..."
    }
]
```

**Thêm dataset vào** `LLaMA-Factory/data/dataset_info.json`:
```json
"my_dataset": {
    "file_name": "my_data.json",
    "columns": {
        "prompt": "instruction",
        "response": "output"
    }
}
```

### **4. Fine-tune DiffuLLaMA với LoRA**
**Config file: `my_diffullama_finetune.yaml`**
```yaml
### model
model_name_or_path: diffusionfamily/diffullama
print_param_status: true

### method
stage: ddm-sft
do_train: true
finetuning_type: lora
lora_target: all
additional_target: embed_tokens
lora_rank: 16

### dataset
dataset: my_dataset
template: empty
cutoff_len: 256
streaming: false
overwrite_cache: true

### output
output_dir: output/my-diffullama-finetuned/
logging_steps: 10
save_steps: 2000

### train
per_device_train_batch_size: 8
learning_rate: 1.0e-4
num_train_epochs: 5.0
bf16: true
anneal_steps: 1
shift: true
```

**Chạy fine-tuning:**
```bash
cd LLaMA-Factory
llamafactory-cli train my_diffullama_finetune.yaml
```

### **5. Fine-tune DiffuGPT**
**Config file: `my_diffugpt_finetune.yaml`**
```yaml
### model
model_name_or_path: diffusionfamily/diffugpt-m
checkpoint_dir: # để trống nếu dùng HuggingFace model

### method
stage: ddm-sft
do_train: true
finetuning_type: full

### dataset
dataset: my_dataset
template: empty
cutoff_len: 256

### output
output_dir: output/my-diffugpt-finetuned/

### train
per_device_train_batch_size: 64
learning_rate: 3.0e-4
num_train_epochs: 10.0
bf16: true
anneal_steps: 1
shift: true
```

### **6. Inference (Test mô hình)**
**Config file: `my_inference.yaml`**
```yaml
### model
model_name_or_path: diffusionfamily/diffullama
adapter_name_or_path: output/my-diffullama-finetuned/checkpoint-2000

### method
stage: ddm-sft
do_predict: true
finetuning_type: lora
shift: true

### dataset
dataset: my_test_dataset
template: empty
cutoff_len: 256

### output
output_dir: output/predictions/

### eval
per_device_eval_batch_size: 32
logits_temp: 1.0
diffusion_steps: 64
```

**Chạy inference:**
```bash
llamafactory-cli train my_inference.yaml
```

### **7. Sử dụng Quick Inference**
```bash
# DiffuGPT
python inf_diffugpt.py --model_name output/my-diffugpt-finetuned/ --base_model_name gpt2-medium

# DiffuLLaMA
python inf_diffullama.py --model_name output/my-diffullama-finetuned/ --flash_attn flash_attention_2
```

## 🎯 **Các điểm quan trọng:**

1. **LoRA vs Full Fine-tuning**: LoRA nhanh hơn, ít VRAM hơn, full fine-tuning hiệu quả hơn
2. **Dataset**: Đã có sẵn GSM8K dataset, bạn có thể test với nó
3. **Parameters quan trọng**: 
   - `stage: ddm-sft` (diffusion supervised fine-tuning)
   - `shift: true` (bắt buộc cho diffusion models)
   - `anneal_steps: 1` (cho fine-tuning)

4. **Monitoring**: Theo dõi loss trong quá trình training
5. **Inference**: Điều chỉnh `diffusion_steps` và `logits_temp` để kiểm soát chất lượng generation

Bạn có muốn tôi giúp tạo config cụ thể cho dataset của bạn không?