# Advanced LLM Server Development: A Comprehensive Guide

## Lesson 1: Adding Authentication
### Understanding Authentication in API Servers

Authentication is like checking an ID card before letting someone into a building. In our LLM server, we want to make sure only authorized users can access our API. Let's implement a simple but secure authentication system using JSON Web Tokens (JWT).

First, we'll need to install the required package:
```bash
pip install flask-jwt-extended
```

Here's how we implement authentication:

```python
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
import datetime

# Add this to your app configuration
app.config['JWT_SECRET_KEY'] = 'your-secret-key'  # In production, use a secure secret
jwt = JWTManager(app)

# Create a user model (in practice, use a database)
users = {
    "demo@example.com": {
        "password": "hashed_password_here",
        "name": "Demo User"
    }
}

@app.route('/login', methods=['POST'])
def login():
    """
    Handle user login and token generation
    
    This function:
    1. Validates user credentials
    2. Creates a JWT token
    3. Returns the token for future API calls
    """
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    # In practice, verify against a database and use proper password hashing
    if email in users and users[email]['password'] == password:
        access_token = create_access_token(
            identity=email,
            expires_delta=datetime.timedelta(days=1)
        )
        return jsonify({'token': access_token})
    
    return jsonify({'error': 'Invalid credentials'}), 401

# Now protect your routes with @jwt_required
@app.route('/generate', methods=['POST'])
@jwt_required()
def generate():
    # Your existing generate function code here
    pass
```

When using authentication, remember to:
1. Never store passwords in plain text
2. Use HTTPS in production
3. Implement token refresh mechanisms
4. Add rate limiting per user
5. Monitor for suspicious activity

## Lesson 2: Implementing Rate Limiting
### Protecting Your Server from Overuse

Rate limiting prevents users from overwhelming your server with too many requests. Think of it like a traffic light that controls the flow of cars. Let's implement rate limiting using Flask-Limiter:

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Apply specific limits to endpoints
@app.route('/generate', methods=['POST'])
@limiter.limit("10 per minute")  # Custom limit for this endpoint
@jwt_required()
def generate():
    """
    Generate LLM responses with rate limiting
    
    The limiter:
    1. Tracks requests per IP address
    2. Enforces limits per minute/hour/day
    3. Returns 429 Too Many Requests when exceeded
    """
    try:
        # Your existing generation code here
        pass
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add storage for rate limiting data
from flask_limiter.storage import MemoryStorage
limiter.storage = MemoryStorage()  # In production, use Redis or similar
```

Best practices for rate limiting:
1. Set different limits for different endpoints
2. Use Redis or a similar storage backend in production
3. Implement graduated rate limiting
4. Add clear error messages when limits are reached
5. Monitor rate limit hits to adjust as needed

## Lesson 3: Adding More Document Types
### Expanding Your Server's Document Processing Capabilities

Let's add support for more document types like Word documents, HTML, and Markdown. We'll use different document loaders based on file type:

```python
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredEmailLoader
)

def get_document_loader(file_path, file_type):
    """
    Select appropriate document loader based on file type
    
    This function:
    1. Identifies file type
    2. Creates appropriate loader
    3. Handles file-specific parsing options
    """
    loaders = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.doc': UnstructuredWordDocumentLoader,
        '.html': UnstructuredHTMLLoader,
        '.htm': UnstructuredHTMLLoader,
        '.md': UnstructuredMarkdownLoader,
        '.csv': CSVLoader,
        '.eml': UnstructuredEmailLoader
    }
    
    loader_class = loaders.get(file_type.lower())
    if not loader_class:
        raise ValueError(f"Unsupported file type: {file_type}")
        
    return loader_class(file_path)

@app.route('/upload', methods=['POST'])
@jwt_required()
def upload_document():
    """
    Enhanced document upload handler
    
    Process:
    1. Validates file type
    2. Selects appropriate loader
    3. Processes document with type-specific settings
    4. Stores in vector database with metadata
    """
    try:
        file = request.files['file']
        file_type = os.path.splitext(file.filename)[1]
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_type) as temp_file:
            file.save(temp_file.name)
            
            # Get appropriate loader
            loader = get_document_loader(temp_file.name, file_type)
            
            # Process document
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'file_type': file_type,
                    'filename': file.filename,
                    'upload_date': datetime.now().isoformat()
                })
            
            # Split and store
            texts = text_splitter.split_documents(documents)
            ids = vector_store.add_documents(texts)
            
            return jsonify({
                'message': 'Document processed successfully',
                'document_ids': ids
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if 'temp_file' in locals():
            os.unlink(temp_file.name)
```

## Lesson 4: Optimizing Response Generation
### Improving Speed and Quality of LLM Responses

Let's optimize how we generate responses to be faster and more efficient:

```python
def optimize_generation_settings(input_length):
    """
    Dynamically adjust generation parameters based on input
    
    This function:
    1. Calculates optimal settings based on input length
    2. Adjusts beam search parameters
    3. Manages memory usage
    """
    # Base settings
    settings = {
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 50,
        'num_beams': 4
    }
    
    # Adjust based on input length
    if input_length > 1000:
        settings.update({
            'num_beams': 2,
            'temperature': 0.8
        })
    elif input_length < 100:
        settings.update({
            'num_beams': 6,
            'temperature': 0.6
        })
        
    return settings

@app.route('/generate', methods=['POST'])
@jwt_required()
@limiter.limit("10 per minute")
def generate():
    """
    Optimized response generation
    
    Features:
    1. Dynamic parameter adjustment
    2. Efficient memory management
    3. Response quality monitoring
    4. Caching for similar queries
    """
    try:
        data = request.get_json()
        input_text = data.get('text', '')
        
        # Get optimal settings
        settings = optimize_generation_settings(len(input_text))
        
        # Prepare input
        input_tokens = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        ).to(device)
        
        # Generate with optimized settings
        with torch.no_grad():  # Disable gradient calculation for inference
            generation_output = model.generate(
                input_tokens['input_ids'],
                max_new_tokens=MAX_TOKENS,
                **settings,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Clean up memory
        torch.cuda.empty_cache()
        
        return jsonify({
            'response': tokenizer.decode(generation_output[0])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

## Lesson 5: Implementing Proper Error Handling
### Creating a Robust Error Management System

Let's implement comprehensive error handling to make our server more reliable:

```python
from functools import wraps
import traceback

# Custom exception classes
class ModelNotReadyError(Exception):
    pass

class TokenLimitError(Exception):
    pass

class DocumentProcessingError(Exception):
    pass

def handle_errors(f):
    """
    Decorator for consistent error handling
    
    This decorator:
    1. Catches and categorizes errors
    2. Logs error details
    3. Returns appropriate error responses
    4. Maintains error statistics
    """
    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ModelNotReadyError:
            logger.error("Model not initialized")
            return jsonify({
                'error': 'Model not ready',
                'code': 'MODEL_NOT_READY'
            }), 503
        except TokenLimitError:
            logger.warning("Token limit exceeded")
            return jsonify({
                'error': 'Input too long',
                'code': 'TOKEN_LIMIT_EXCEEDED'
            }), 400
        except DocumentProcessingError as e:
            logger.error(f"Document processing error: {str(e)}")
            return jsonify({
                'error': 'Failed to process document',
                'code': 'DOCUMENT_PROCESSING_ERROR',
                'details': str(e)
            }), 400
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': 'Internal server error',
                'code': 'INTERNAL_ERROR'
            }), 500
    return wrapped

@app.route('/generate', methods=['POST'])
@jwt_required()
@limiter.limit("10 per minute")
@handle_errors
def generate():
    """
    Generate responses with proper error handling
    
    Error handling:
    1. Validates input
    2. Checks model status
    3. Monitors token limits
    4. Handles generation errors
    """
    data = request.get_json()
    
    # Input validation
    if not data or 'text' not in data:
        raise ValueError("Missing required 'text' field")
        
    # Check model status
    if model is None or tokenizer is None:
        raise ModelNotReadyError()
        
    # Check token limit
    input_length = len(tokenizer.encode(data['text']))
    if input_length > MAX_LENGTH:
        raise TokenLimitError()
        
    # Your existing generation code here
    pass

# Add error monitoring
error_stats = {
    'total_errors': 0,
    'error_types': {}
}

def update_error_stats(error_type):
    """Track error statistics for monitoring"""
    error_stats['total_errors'] += 1
    error_stats['error_types'][error_type] = error_stats['error_types'].get(error_type, 0) + 1

@app.route('/stats/errors', methods=['GET'])
@jwt_required()
def get_error_stats():
    """Endpoint to monitor error statistics"""
    return jsonify(error_stats)
```

Remember to:
1. Log all errors appropriately
2. Monitor error patterns
3. Implement retry mechanisms where appropriate
4. Provide clear error messages to users
5. Set up alerts for critical errors

These lessons provide a foundation for building a more robust and feature-complete LLM server. Each enhancement adds important functionality while maintaining security and reliability.

Now that you have a basic LLM server running with some advanced features, you are ready to go a little deeper including adding a vector database and additional LLM model selection. Click here for [the advanced topics course](https://github.com/ThePragmaticFuturist/chat_gp_me/blob/main/webinars/LLM%20Server%20Course/3_advanced_llm_topics.md).
