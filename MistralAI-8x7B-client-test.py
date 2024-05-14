from transformers import pipeline
from huggingface_hub import InferenceClient
import time

client1 = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

def model(text):
    generate_kwargs = dict(
        temperature=0.7,
        max_new_tokens=4096,
        top_p=0.95,
        repetition_penalty=1,
        do_sample=True,
        seed=42,
    )
    
    formatted_prompt = text
    start_time = time.time()  # Start time measurement
    stream = client1.text_generation(
        formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""
    for response in stream:
        if not response.token.text == "</s>":
            output += response.token.text
    end_time = time.time()  # End time measurement
    latency = end_time - start_time  # Calculate latency
    print("Latency:", latency, "seconds")
    return output

if __name__ == "__main__":
    while True:
        text = input("Nanya apa kau? ")
        if text.lower() in ["q", "exit"]:  # Check if the input is "q" or "exit"
            break  # Exit the loop if input is "q" or "exit"
        response = model(text)
        print("===================================================")
        print("AI: ", response)
        print("===================================================")
