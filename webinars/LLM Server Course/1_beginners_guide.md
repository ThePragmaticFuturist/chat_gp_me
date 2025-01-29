# Building an LLM Server: A Complete Guide for Beginners

## Introduction
This guide will walk you through creating a Language Learning Model (LLM) server from scratch. An LLM server allows you to run AI models locally and create a chat interface similar to ChatGPT. We'll use Python and several supporting libraries to build this server.

## Prerequisites
Before starting, you should have:
- A computer running Ubuntu Linux (or similar distribution)
- Basic knowledge of Python programming
- Basic understanding of command line operations
- An NVIDIA GPU (recommended for better performance)

## Part 1: Setting Up Your Environment

### Step 1: Installing NVIDIA CUDA Toolkit
CUDA allows your computer to use the GPU for running AI models. Here's how to install it:

```bash
# Download and install CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# Install NVIDIA drivers
sudo apt-get install -y nvidia-open

# Set up environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Step 2: Setting Up Python Environment
We'll use Conda to manage our Python environment:

```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Initialize conda
source ~/.bashrc
conda init

# Create and activate environment
conda create -n cuda_env python=3.10
conda activate cuda_env

# Configure conda channels
conda config --add channels pytorch
conda config --add channels nvidia
conda config --set channel_priority flexible

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Step 3: Installing Required Python Packages
```bash
pip install flask flask-cors transformers accelerate langchain-community chromadb
```

## Part 2: Understanding the LLM Server Components

### Key Components Overview
1. **Flask Server**: Handles HTTP requests and responses
2. **Transformers Library**: Manages AI models
3. **Vector Database**: Stores and retrieves document embeddings
4. **Text Processing**: Handles document chunking and embeddings
5. **Chat Session Management**: Maintains conversation history

### Core Functionalities

#### 1. Model Management
The server loads and manages AI models using the Transformers library. Here's how it works:

```python
def initialize_model(model_name):
    """
    Loads an AI model and prepares it for use
    
    Parameters:
        model_name (str): Name/path of the model to load
        
    The function:
    1. Clears GPU memory if needed
    2. Loads the tokenizer for processing text
    3. Loads the model with optimized settings
    4. Configures the model for inference
    """
    global model, tokenizer
    
    # Clear GPU memory if available
    if torch.cuda.is_available():
        if model is not None:
            del model
        torch.cuda.empty_cache()
        gc.collect()

    try:
        # Load tokenizer for processing text
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure proper padding tokens are set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision for memory efficiency
            device_map='auto',          # Automatically manage GPU memory
            low_cpu_mem_usage=True      # Optimize CPU memory usage
        )
        
        # Set model to evaluation mode
        model.eval()
        return True
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        return False
```

#### 2. Document Processing
The server can process uploaded documents for retrieval-augmented generation:

```python
@app.route('/upload', methods=['POST'])
def upload_document():
    """
    Handles document uploads and processing
    
    Steps:
    1. Receives uploaded file
    2. Creates temporary storage
    3. Processes document based on type
    4. Splits into manageable chunks
    5. Stores in vector database
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Create temporary storage
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

        # Select appropriate document loader
        if file.filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(temp_path)
        elif file.filename.lower().endswith('.txt'):
            loader = TextLoader(temp_path)
            
        # Process and store document
        documents = loader.load()
        texts = text_splitter.split_documents(documents)
        ids = vector_store.add_documents(texts)
        vector_store.persist()
        
        return jsonify({
            'message': f'Successfully processed {file.filename}',
            'document_ids': ids
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

#### 3. Chat Session Management
The server maintains chat sessions for continuous conversations:

```python
@app.route('/chat/start', methods=['POST'])
def start_chat():
    """
    Initializes a new chat session
    
    Features:
    1. Creates unique session ID
    2. Stores conversation history
    3. Generates initial response
    4. Maintains context between messages
    """
    try:
        data = request.get_json()
        session_id = data.get('sessionId', str(uuid.uuid4()))
        initial_message = data.get('message', '')
        
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        if initial_message:
            # Format system prompt
            prompt = f"<|system|>You are a helpful AI assistant.</s><|user|>{initial_message}</s><|assistant|>"
            
            # Generate response
            input_tokens = tokenizer(
                [prompt],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True
            ).to(device)
            
            # Generate AI response
            generation_output = model.generate(
                input_tokens['input_ids'],
                max_new_tokens=MAX_TOKENS,
                temperature=0.7,    # Controls randomness
                top_p=0.9,         # Nucleus sampling
                do_sample=True,    # Enable sampling
                num_beams=4        # Beam search for better quality
            )
            
            # Process response
            response = tokenizer.decode(generation_output.sequences[0])
            response = response.split("<|assistant|>")[-1].strip()
            
            # Store conversation
            chat_sessions[session_id].extend([
                {
                    'role': 'user',
                    'content': initial_message,
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().isoformat()
                }
            ])
            
        return jsonify({
            'sessionId': session_id,
            'messages': chat_sessions[session_id]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

## Part 3: Running the Server

### Starting the Server
1. Activate your environment:
```bash
conda activate cuda_env
```

2. Start the server:
```bash
python app.py
```

The server will start on port 5000 by default.

### Testing the Server
You can test the server using curl commands:

```bash
# Test model loading
curl http://localhost:5000/models

# Start a chat session
curl -X POST http://localhost:5000/chat/start \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello, how are you?"}'
```

## Common Issues and Solutions

### Memory Management
If you encounter GPU memory issues:
- Reduce MAX_LENGTH and MAX_TOKENS values
- Use smaller models
- Clear GPU cache regularly

### Model Loading
If models fail to load:
- Check internet connection for downloading
- Verify GPU memory availability
- Ensure correct model names in models.json

### Document Processing
If document processing fails:
- Check file permissions
- Verify supported file types
- Monitor disk space for temporary files

## Best Practices
Here are some [best practices](https://github.com/ThePragmaticFuturist/chat_gp_me/edit/main/webinars/LLM%20Server%20Course/4_llm_server_best_practices.md) for your server:
1. Always monitor GPU memory usage
2. Implement proper error handling
3. Use logging for debugging
4. Regular maintenance of vector store
5. Proper session cleanup

## Next Steps
After setting up the basic server, [add advanced features:](https://github.com/ThePragmaticFuturist/chat_gp_me/edit/main/webinars/LLM%20Server%20Course/2_advanced_llm_server_development.md)
1. Adding authentication
2. Implementing rate limiting
3. Adding more document types
4. Optimizing response generation
5. Implementing proper error handling

Remember to always monitor server performance and adjust settings as needed.
