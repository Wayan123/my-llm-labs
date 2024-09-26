import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import sys

# Tentukan ID model yang mendukung bahasa Indonesia
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# Muat tokenizer dan model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Fungsi untuk menghasilkan respons dengan tipewriter effect
def generate_response(prompt, max_new_tokens=256, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

    # Siapkan parameter generate
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id
    }

    # Mulai waktu inferensi
    start_time = time.time()

    # Generate dengan mengumpulkan output token per langkah
    generated_ids = model.generate(
        input_ids,
        **generation_kwargs,
        output_scores=True,
        return_dict_in_generate=True
    )

    # Hitung durasi inferensi
    end_time = time.time()
    duration = end_time - start_time

    # Ambil token yang dihasilkan
    generated_tokens = generated_ids.sequences[0][input_ids.shape[-1]:]

    # Dekode token menjadi teks
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Hitung jumlah token yang dihasilkan
    num_tokens = len(generated_tokens)

    # Hitung tokens per detik
    if duration > 0:
        tps = num_tokens / duration
    else:
        tps = num_tokens  # Jika durasi terlalu kecil

    return response, num_tokens, tps

# Inisialisasi pesan sistem
system_prompt = "Anda adalah asisten virtual yang cerdas dan selalu merespons dalam bahasa Indonesia."

print(" ")
print("Selamat datang di Chatbot Bahasa Indonesia! Ketik 'keluar' untuk mengakhiri percakapan.\n")

while True:
    # Minta input dari pengguna
    user_input = input("Anda: ")
    
    if user_input.lower() == 'keluar':
        print("Chatbot: Sampai jumpa!")
        break

    # Siapkan prompt untuk model
    prompt = f"{system_prompt}\nAnda: {user_input}\nAsisten:"

    # Generate respons
    try:
        response, num_tokens, tps = generate_response(
            prompt,
            max_new_tokens=4096,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
    except Exception as e:
        print(f"Terjadi kesalahan saat menghasilkan respons: {e}")
        continue

    # Tampilkan respons dengan efek tipewriter
    for word in response.split():
        print(word, end=' ', flush=True)
        time.sleep(0.05)  # Penundaan untuk efek tipewriter (sesuaikan sesuai keinginan)
    print()  # Baris baru setelah respons selesai

    # Tampilkan informasi TPS
    print(f"(Tokens dihasilkan: {num_tokens}, Kecepatan: {tps:.2f} token/detik)\n")

