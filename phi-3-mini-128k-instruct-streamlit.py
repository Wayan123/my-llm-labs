import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time

# Set random seed for reproducibility
torch.random.manual_seed(0)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct", 
    device_map="cpu", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

# Define pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Define default generation arguments
generation_args = {
    "max_new_tokens": 4096,
    "return_full_text": False,
    "temperature": 0.5,  # Default temperature set to 0.5
    "do_sample": True,   # Default do_sample set to True
}

# Streamlit App
st.title("Phi 3 Mini 128k Instruct - Streamlit")

# Sidebar for generation parameters
st.sidebar.subheader("Generation Parameters")

# Slider for max_new_tokens
max_tokens = st.sidebar.slider("Max Tokens", min_value=100, max_value=4096, value=500, step=10)
generation_args["max_new_tokens"] = max_tokens

# Slider for temperature
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
generation_args["temperature"] = temperature

# Radio buttons for return_full_text
return_full_text = st.sidebar.radio("Return Full Text", [False, True], index=0)
generation_args["return_full_text"] = return_full_text

# Radio buttons for do_sample
do_sample = st.sidebar.radio("Do Sample", [False, True], index=0)
generation_args["do_sample"] = do_sample

# Input text area for user messages
st.subheader("Input Your Message")
user_input = st.text_area("Type your message here", "What about solving an 2x + 3 = 7 equation?")

# Button to submit user input
if st.button("Submit"):
    if user_input:
        # Add user input to messages
        messages = [{"role": "user", "content": user_input}]
        
        # Display loading animation
        with st.spinner("Generating text..."):
            # Measure latency
            start_time = time.time()
            
            # Generate text based on user input
            output = pipe(messages, **generation_args)
            generated_text = output[0]['generated_text']
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Convert latency to hours, minutes, and seconds
            hours, rem = divmod(latency, 3600)
            minutes, seconds = divmod(rem, 60)
            
            st.write("Generated Text:", generated_text)
            st.write("Latency:", "{:0>2} Hour {:0>2} Minutes {:05.2f} Seconds".format(int(hours), int(minutes), seconds))
    else:
        st.warning("Please enter a message before submitting.")
