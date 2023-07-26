# Llama 2 - Getting Started Guide

Welcome to Llama 2! This guide will help you install the required packages, import the necessary libraries, load and configure the model, and generate responses using Llama 2. Let's get started!

## 1. Install Required Packages

In order to use Llama 2, we need to install the required packages. Execute the following code to install them:

```python
!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose
!pip install huggingface_hub
```

## 2. Import Required Libraries

After installing the packages, we need to import the necessary libraries to work with Llama 2:

```python
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
```

## 3. Load and Configure the Model

Now, let's load the Llama 2 model and configure it. The model will be automatically downloaded from the Hugging Face Hub:

```python
model_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"

model_path = hf_hub_download(repo_id=model_path, filename=model_basename)

lcpp_llm = Llama(
    model_path=model_path,
    n_threads=4,       # Number of CPU cores
    n_batch=512,       # Batch size, consider your GPU's VRAM
    n_gpu_layers=32    # Adjust based on your model and GPU VRAM
)
```

**Note:** In the `n_gpu_layers` parameter, consider your model's architecture and the available VRAM on your GPU.

## 4. Prompt Template

Before generating a response, let's set up a prompt template that the model will use to generate the answer. In this example, we use a pandas-related prompt:

```python
prompt = "Write a pandas code for below query?\nWhich industry has the highest average number of employees per company?"
prompt_template = f'''Question: {prompt}\nResponse:'''
```

## 5. Generate the Response

Finally, let's generate the response using Llama 2:

```python
response = lcpp_llm(
    prompt=prompt_template,
    max_tokens=256,
    temperature=0.5,
    top_p=0.95,
    repeat_penalty=1.2,
    top_k=150,
    echo=True
)

print(response["choices"][0]["text"])
```

The `temperature`, `top_p`, and `top_k` parameters influence the randomness and diversity of the response. Feel free to experiment with different values to achieve the desired results!

That's it! You are now ready to have interactive conversations with Llama 2 and use it for various tasks. Happy chatting!

For more details about the "llama-cpp-python" library and its functionalities, you can refer to its official documentation and GitHub repository.

Documentation: https://llama.readthedocs.io/en/latest/

---

_By [Renukumar R]