import transformers
import torch
import time

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Mengatur device ke CUDA tanpa fallback ke CPU
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda"  # Memaksa penggunaan CUDA
)

messages = [
    {"role": "system", "content": "Kamu adalah chatbot bajak laut yang selalu merespons dengan gaya bicara bajak laut!"},
    {"role": "user", "content": "Siapa kamu?"},  # Input user
]

# Mulai pengukuran waktu
start_time = time.time()

outputs = pipeline(
    messages,
    max_new_tokens=5000,
)

# Selesai pengukuran waktu
end_time = time.time()

# Menghitung waktu inferensi dalam detik
inference_time = end_time - start_time

# Konversi waktu inferensi ke format jam, menit, detik
hours, rem = divmod(inference_time, 3600)
minutes, seconds = divmod(rem, 60)

# Mengambil teks dari respons bot tanpa tanda kurung
response = outputs[0]["generated_text"]

# Cetak hasil dengan format yang diinginkan
print(f"Input user: {messages[1]['content']}")
print(f"Output Bot: {response}")

# Cetak waktu inferensi yang diformat
print(f"Waktu inferensi: {int(hours)} jam, {int(minutes)} menit, {seconds:.2f} detik")
