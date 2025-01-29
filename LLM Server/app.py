# LLM server 
# file structure
# cuda_env
# cuda_env/llmserver/models
# cuda_env/llmserver/vectordb
# cuda_env/llmserver/app.py
# cuda_env/llmserver/models.json

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.serving import WSGIRequestHandler
from werkzeug.utils import secure_filename
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from typing import List, Union
import torch
import logging
import json
import gc
import os
import tempfile
import shutil
import atexit
# Add these imports and globals for threaded session
from typing import Dict, List
import uuid
from datetime import datetime

# Add this after other global variables
chat_sessions: Dict[str, List[Dict]] = {}

SESSIONS_FILE = "chat_sessions.json"

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Print CUDA information
print("Is CUDA available:", torch.cuda.is_available())
print("CUDA devices count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Configuration
VECTOR_DB_PATH = "vectordb"
MAX_LENGTH = 2048
MAX_TOKENS = 512
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
WSGIRequestHandler.protocol_version = "HTTP/1.1"

# Initialize components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
	model_name="sentence-transformers/all-mpnet-base-v2",
	model_kwargs={'device': device}
)

# Initialize vector store
if not os.path.exists(VECTOR_DB_PATH):
	vector_store = Chroma(
		persist_directory=VECTOR_DB_PATH,
		embedding_function=embeddings
	)
else:
	vector_store = Chroma(
		persist_directory=VECTOR_DB_PATH,
		embedding_function=embeddings
	)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
	chunk_size=CHUNK_SIZE,
	chunk_overlap=CHUNK_OVERLAP,
	length_function=len
)

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_models():
	try:
		with open('models.json', 'r') as f:
			return json.load(f)
	except FileNotFoundError:
		# Creates default models file if missing
		
		logger.error("models.json not found. Creating default file.")
		default_models = [
			{
				"name": "TinyLlama Chat 1.1B",
				"url": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
			}
		]
		with open('models.json', 'w') as f:
			json.dump(default_models, f, indent=2)
		return default_models
	except json.JSONDecodeError:
		logger.error("Invalid JSON in models.json")
		return []

available_models = load_models()

def load_chat_sessions():
    """Load chat sessions from disk"""
    global chat_sessions
    try:
        if os.path.exists(SESSIONS_FILE):
            with open(SESSIONS_FILE, 'r') as f:
                loaded_sessions = json.load(f)
                chat_sessions = loaded_sessions
                logger.info(f"Loaded {len(chat_sessions)} chat sessions from disk")
        else:
            chat_sessions = {}
            logger.info("No saved chat sessions found")
    except Exception as e:
        logger.error(f"Error loading chat sessions: {str(e)}")
        chat_sessions = {}

def save_chat_sessions():
    """Save chat sessions to disk"""
    try:
        with open(SESSIONS_FILE, 'w') as f:
            json.dump(chat_sessions, f)
        logger.info(f"Saved {len(chat_sessions)} chat sessions to disk")
    except Exception as e:
        logger.error(f"Error saving chat sessions: {str(e)}")

# Add this near the top of app.py, before the route definitions
def initialize_vector_store():
	"""Initialize or load the vector store with proper error handling"""
	global vector_store
	
	try:
		if not os.path.exists(VECTOR_DB_PATH):
			os.makedirs(VECTOR_DB_PATH)
			logger.info(f"Created vector store directory at {VECTOR_DB_PATH}")
			
		vector_store = Chroma(
			persist_directory=VECTOR_DB_PATH,
			embedding_function=embeddings
		)
		logger.info("Vector store initialized successfully")
		
		# Verify the store is working
		try:
			vector_store._collection.count()
			logger.info("Vector store connection verified")
		except Exception as e:
			logger.error(f"Vector store verification failed: {str(e)}")
			raise
			
	except Exception as e:
		logger.error(f"Failed to initialize vector store: {str(e)}")
		raise

# Call this during app initialization
initialize_vector_store()
load_chat_sessions()

# Add a cleanup function
@atexit.register
def cleanup():
	"""Ensure vector store is properly persisted on shutdown"""
	try:
		if vector_store:
			vector_store.persist()
			logger.info("Vector store persisted successfully")
		save_chat_sessions()
	except Exception as e:
		logger.error(f"Error persisting vector store: {str(e)}")
        
def initialize_model(model_name):
	"""Initialize or reinitialize the model with new settings"""
	global model, tokenizer

	# Clear GPU memory if available
	if torch.cuda.is_available():
		if model is not None:
			del model
		torch.cuda.empty_cache()
		gc.collect()

	logger.info(f"Initializing model: {model_name}")
	cache_dir = "/home/ken/.cache/huggingface/hub"

	try:
		# Initialize tokenizer and model
		tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
		
		# Set padding token if missing
		if tokenizer.pad_token is None:
			tokenizer.pad_token = tokenizer.eos_token
			tokenizer.pad_token_id = tokenizer.eos_token_id
		
		# Load model with optimizations and automatic device mapping
		max_memory = {0: "15GB", "cpu": "30GB"}  # Adjust values based on your system
		
		try:
			model = AutoModelForCausalLM.from_pretrained(
				model_name,
				torch_dtype=torch.float16,
				low_cpu_mem_usage=True,
				cache_dir=cache_dir,
				device_map='auto',
				# max_memory=max_memory  # Add this parameter
			)
		except RuntimeError as e:
			if "out of memory" in str(e):
				logger.error("GPU out of memory. Try a smaller model or increase system memory")
			raise
		
		# Set padding token ID if missing
		if model.config.pad_token_id is None:
			model.config.pad_token_id = tokenizer.pad_token_id

		# Set evaluation mode without moving the model
		model.eval()
		logger.info(f"Model {model_name} initialized")
		return True

	except Exception as e:
		logger.error(f"Error initializing model: {str(e)}")
		return False

# Initialize model on startup
initialize_model(MODEL_NAME)

@app.route('/clear_db', methods=['GET'])
def clear_database():
    try:
        # Get all document IDs from the collection
        all_ids = vector_store._collection.get()['ids']
        
        if not all_ids:
            return jsonify({'message': 'Database is already empty'}), 200
            
        # Delete all documents
        vector_store._collection.delete(ids=all_ids)
        
        # Persist the changes
        vector_store.persist()
        
        logger.info(f"Successfully cleared database, removed {len(all_ids)} documents")
        return jsonify({
            'message': f'Successfully cleared database',
            'removed_count': len(all_ids)
        })
        
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def get_models():
	# Returns list of available models from models.json
	try:
		return jsonify(available_models)
	except Exception as e:
		logger.error(f"Error getting models: {str(e)}")
		return jsonify({'error': str(e)}), 500

@app.route('/settings', methods=['POST'])
def update_settings():
	global MAX_LENGTH, MAX_TOKENS, MODEL_NAME
	
	try:
		# Validate and update settings
		data = request.get_json()
		new_max_length = int(float(data.get('maxLength', MAX_LENGTH)))
		new_max_tokens = int(float(data.get('maxTokens', MAX_TOKENS)))
		new_model_name = data.get('model', MODEL_NAME)
		
		print(f"Model: {new_model_name}")

		# Input validation
		if not isinstance(new_max_length, int) or new_max_length <= 0:
			return jsonify({'error': 'Invalid maxLength value'}), 400
		if not isinstance(new_max_tokens, int) or new_max_tokens <= 0:
			return jsonify({'error': 'Invalid maxTokens value'}), 400
		if not isinstance(new_model_name, str) or not new_model_name.strip():
			return jsonify({'error': 'Invalid model name'}), 400

		# Update global settings
		MAX_LENGTH = new_max_length
		MAX_TOKENS = new_max_tokens
		
		print(f"Model: {new_model_name} {MAX_LENGTH} {MAX_TOKENS}")

		# Reinitialize model if changed
		if new_model_name != MODEL_NAME:
			MODEL_NAME = new_model_name
			if not initialize_model(MODEL_NAME):
				return jsonify({'error': 'Failed to initialize new model'}), 500

		return jsonify({
			'status': 'success',
			'settings': {
				'maxLength': MAX_LENGTH,
				'maxTokens': MAX_TOKENS,
				'model': MODEL_NAME
			}
		})

	except Exception as e:
		logger.error(f"Error updating settings: {str(e)}")
		return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_document():
	try:
		# Validate file presence
		print("Received upload request")
		if 'file' not in request.files:
			print("No file in request")
			return jsonify({'error': 'No file provided'}), 400
		
		file = request.files['file']
		if file.filename == '':
			print("No filename")
			return jsonify({'error': 'No file selected'}), 400

		# Create temporary file
		print(f"Processing file: {file.filename}")
		
		with tempfile.NamedTemporaryFile(delete=False) as temp_file:
			file.save(temp_file.name)
			temp_path = temp_file.name

		# Select appropriate loader based on file type
		if file.filename.lower().endswith('.pdf'):
			loader = PyPDFLoader(temp_path)
		elif file.filename.lower().endswith('.txt'):
			loader = TextLoader(temp_path)
		else:
			os.unlink(temp_path)
			return jsonify({'error': 'Unsupported file type'}), 400

		# Process document
		print("Loading document")
		documents = loader.load()
		print("Splitting text")
		texts = text_splitter.split_documents(documents)
		print("Adding to vector store")
		ids = vector_store.add_documents(texts)
		vector_store.persist()

		# Cleanup and return
		os.unlink(temp_path)
		print(f"Document IDs: {file.filename} : {ids}")
		return jsonify({
			'message': f'Successfully processed {file.filename}',
			'document_ids': ids
		})

	except Exception as e:
		logger.error(f"Error uploading document: {str(e)}")
		return jsonify({'error': str(e)}), 500

@app.route('/delete_chunk/<chunk_id>', methods=['DELETE'])
def delete_chunk(chunk_id):
	try:
		if not chunk_id:
			return jsonify({'error': 'No chunk ID provided'}), 400
			
		# Remove single chunk from vector store
		vector_store._collection.delete(ids=[chunk_id])
		vector_store.persist()
		
		return jsonify({
			'message': f'Successfully deleted chunk {chunk_id}',
			'deleted_id': chunk_id
		})
		
	except Exception as e:
		logger.error(f"Error deleting chunk: {str(e)}")
		return jsonify({'error': str(e)}), 500

@app.route('/delete_document', methods=['DELETE'])
def delete_document():
	try:
		data = request.get_json()
		if not data:
			logger.error("No JSON data received")
			return jsonify({'error': 'No data provided'}), 400
			
		ids_to_remove = data.get('ids', [])
		logger.info(f"Attempting to delete documents with IDs: {ids_to_remove}")
		
		if not ids_to_remove or not isinstance(ids_to_remove, list):
			logger.error(f"Invalid or missing IDs: {ids_to_remove}")
			return jsonify({'error': 'Invalid or missing document IDs'}), 400

		# Verify documents exist before deletion
		existing_docs = vector_store._collection.get(
			ids=ids_to_remove,
			include=['documents']
		)
		
		if not existing_docs or not existing_docs['documents']:
			logger.error("No documents found to delete")
			return jsonify({'error': 'Documents not found'}), 404

		# Remove documents from vector store
		vector_store._collection.delete(ids=ids_to_remove)
		vector_store.persist()
		
		logger.info(f"Successfully deleted {len(ids_to_remove)} documents")
		return jsonify({
			'message': f'Successfully deleted {len(ids_to_remove)} documents',
			'deleted_ids': ids_to_remove
		})
		
	except Exception as e:
		logger.error(f"Error deleting documents: {str(e)}")
		return jsonify({'error': str(e)}), 500

@app.route('/get_texts', methods=['POST'])
def get_texts():
	try:
		data = request.get_json()
		if not data:
			logger.error("No JSON data received")
			return jsonify({'error': 'No data provided'}), 400
			
		ids = data.get('ids', [])
		logger.info(f"Received request for document IDs: {ids}")
		
		if not ids or not isinstance(ids, list):
			logger.error(f"Invalid or missing IDs: {ids}")
			return jsonify({'error': 'Invalid or missing document IDs'}), 400

		# Get documents from Chroma
		results = vector_store._collection.get(
			ids=ids,
			include=['documents', 'metadatas']
		)
		
		if not results or 'documents' not in results:
			logger.error("No documents found in Chroma")
			return jsonify({'error': 'No documents found'}), 404

		# Format response
		documents = []
		for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
			if doc is not None:  # Only include non-null documents
				documents.append({
					'id': ids[i],
					'content': doc,
					'metadata': metadata
				})

		logger.info(f"Successfully retrieved {len(documents)} documents")
		return jsonify({
			'documents': documents,
			'count': len(documents)
		})

	except Exception as e:
		logger.error(f"Error retrieving texts: {str(e)}")
		return jsonify({'error': str(e)}), 500
		
@app.route('/remove', methods=['POST'])
def remove_document():
	try:
		data = request.get_json()
		query = data.get('query', '')
		ids_to_remove = data.get('ids', [])
		
		if not query and not ids_to_remove:
			return jsonify({'error': 'No search query provided'}), 400
			
		if not ids_to_remove:
			# Get documents to remove
			retriever = vector_store.as_retriever(search_kwargs={"k": 5})
			docs = retriever.get_relevant_documents(query)
			
			# Get IDs of documents to remove
			ids_to_remove = [doc.metadata.get('id') for doc in docs]
		
		# Remove from vector store
		vector_store._collection.delete(ids=ids_to_remove)
		vector_store.persist()
		
		return jsonify({
			'message': f'Removed {len(ids_to_remove)} documents',
			'removed_ids': ids_to_remove
		})
		
	except Exception as e:
		logger.error(f"Error removing documents: {str(e)}")
		return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    logger.info("Received generate request")
    
    try:
        # Get input data
        data = request.get_json()
        input_text = data.get('text', '')
        use_rag = data.get('use_rag', False)

        if not input_text:
            return jsonify({'error': 'No input text provided'}), 400

        # If RAG is enabled, retrieve relevant context
        context = ""
        if use_rag:
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(input_text)
            context = "\n".join([doc.page_content for doc in docs])

        # Format the prompt with context if RAG is enabled
        if context:
            prompt = f"<|system|>Use the following context to answer the question.\n{context}</s><|user|>{input_text}</s><|assistant|>"
        else:
            prompt = f"<|system|>Provide a clear, direct response.</s><|user|>{input_text}</s><|assistant|>"

        # Generate response
        input_tokens = tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            add_special_tokens=True
        ).to(device)

        generation_output = model.generate(
            input_tokens['input_ids'],
            max_new_tokens=MAX_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_beams=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )

        # Clean up the response
        full_response = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        response_parts = full_response.split("<|assistant|>")
        if len(response_parts) > 1:
            response = response_parts[-1].strip()
            # Remove any remaining system or user messages
            response = response.split("<|system|>")[0].strip()
            response = response.split("<|user|>")[0].strip()
        else:
            response = full_response.strip()

        if torch.cuda.empty_cache():
            torch.cuda.empty_cache()
            gc.collect()

        return jsonify({
            'response': response,
            'context_used': bool(context),
            'status': 'success'
        })

    except Exception as e:
        logger.error(f"Error in generate: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

@app.route('/chat/start', methods=['POST'])
def start_chat():
    try:
        data = request.get_json()
        session_id = data.get('sessionId', str(uuid.uuid4()))
        initial_message = data.get('message', '')
        
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        if initial_message:
            # Format initial system message
            prompt = f"<|system|>You are a helpful AI assistant. Please maintain context throughout our conversation.</s><|user|>{initial_message}</s><|assistant|>"
            
            # Generate response using existing model
            input_tokens = tokenizer(
                [prompt],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                add_special_tokens=True
            ).to(device)
            
            generation_output = model.generate(
                input_tokens['input_ids'],
                max_new_tokens=MAX_TOKENS,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_beams=4,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True
            )
            
            response = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
            response = response.split("<|assistant|>")[-1].strip()
            
            # Store messages in session
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
            
            save_chat_sessions()
        
        return jsonify({
            'sessionId': session_id,
            'messages': chat_sessions[session_id]
        })
        
    except Exception as e:
        logger.error(f"Error starting chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat/message', methods=['POST'])
def send_message():
    try:
        data = request.get_json()
        session_id = data.get('sessionId')
        message = data.get('message')
        
        if not session_id or session_id not in chat_sessions:
            return jsonify({'error': 'Invalid session ID'}), 400
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
            
        # Build context from previous messages
        context = ""
        for msg in chat_sessions[session_id][-4:]:  # Get last 4 messages for context
            if msg['role'] == 'user':
                context += f"<|user|>{msg['content']}</s>"
            else:
                context += f"<|assistant|>{msg['content']}</s>"
                
        # Format prompt with context
        prompt = f"<|system|>You are a helpful AI assistant.</s>{context}<|user|>{message}</s><|assistant|>"
        
        # Generate response
        input_tokens = tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            add_special_tokens=True
        ).to(device)
        
        generation_output = model.generate(
            input_tokens['input_ids'],
            max_new_tokens=MAX_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_beams=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )
        
        # Clean up the response
        full_response = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        response_parts = full_response.split("<|assistant|>")
        if len(response_parts) > 1:
            response = response_parts[-1].strip()
            # Remove any remaining system or user messages
            response = response.split("<|system|>")[0].strip()
            response = response.split("<|user|>")[0].strip()
        else:
            response = full_response.strip()
        
        # Store messages in session
        chat_sessions[session_id].extend([
            {
                'role': 'user',
                'content': message,
                'timestamp': datetime.now().isoformat()
            },
            {
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            }
        ])
        
        save_chat_sessions()
        
        return jsonify({
            'sessionId': session_id,
            'messages': chat_sessions[session_id]
        })
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat/history/<session_id>', methods=['GET'])
def get_chat_history(session_id):
    try:
        if session_id not in chat_sessions:
            return jsonify({'error': 'Session not found'}), 404
            
        return jsonify({
            'sessionId': session_id,
            'messages': chat_sessions[session_id]
        })
        
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat/session/<session_id>', methods=['DELETE'])
def delete_chat_session(session_id):
    try:
        if session_id not in chat_sessions:
            return jsonify({'error': 'Session not found'}), 404
            
        del chat_sessions[session_id]
        
        save_chat_sessions()
        
        return jsonify({
            'message': 'Session deleted successfully',
            'sessionId': session_id
        })
        
    except Exception as e:
        logger.error(f"Error deleting chat session: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
	logger.info("Starting server on 0.0.0.0:5000")
	app.run(host='0.0.0.0', port=5000, debug=True)
