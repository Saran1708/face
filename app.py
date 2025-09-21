from flask import Flask, render_template, request, jsonify, redirect, url_for
import base64
import os
import mysql.connector
from datetime import datetime
import uuid
from werkzeug.security import generate_password_hash, check_password_hash
from pathlib import Path
import shutil

app = Flask(__name__)

# Configuration - Use environment variables for production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEMP_FOLDER'] = 'temp'
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', 'localhost')
app.config['MYSQL_PORT'] = int(os.getenv('MYSQL_PORT', 3306))
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', 'password')
app.config['MYSQL_DATABASE'] = os.getenv('MYSQL_DATABASE', 'user_registration')

# Create folders if they don't exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['TEMP_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Setup DeepFace
def setup_deepface():
    try:
        # Create DeepFace weights directory if it doesn't exist
        weights_dir = Path.home() / ".deepface" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean up any existing pickle files to avoid cache issues
        pickle_file = os.path.join(app.config['UPLOAD_FOLDER'], 'representations_vgg_face.pkl')
        if os.path.exists(pickle_file):
            os.remove(pickle_file)
            print("Removed existing representations file for fresh start")
        
        return True
    except Exception as e:
        print(f"DeepFace setup error: {e}")
        return False

# Database connection with SSL for production
def get_db_connection():
    try:
        # Check if we're using a cloud database (Railway/PlanetScale)
        ssl_config = {}
        if 'railway' in app.config['MYSQL_HOST'].lower() or 'planetscale' in app.config['MYSQL_HOST'].lower():
            ssl_config = {
                'ssl_disabled': False,
                'ssl_verify_cert': True,
                'ssl_verify_identity': True
            }
        
        return mysql.connector.connect(
            host=app.config['MYSQL_HOST'],
            port=app.config['MYSQL_PORT'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            database=app.config['MYSQL_DATABASE'],
            **ssl_config
        )
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        raise

# Initialize database
def init_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create users table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                image_path VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization error: {e}")

@app.route('/')
def index():
    return render_template('registration.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/face_login', methods=['POST'])
def face_login():
    try:
        # Get captured image data
        image_data = request.form['image']
        
        # Remove data URL prefix
        image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        
        # Save temp image
        temp_filename = f"temp_login_{uuid.uuid4().hex}.jpg"
        temp_filepath = os.path.join(app.config['TEMP_FOLDER'], temp_filename)
        
        with open(temp_filepath, 'wb') as f:
            f.write(img_bytes)
        
        # Import DeepFace here to avoid startup issues
        try:
            from deepface import DeepFace
        except ImportError:
            return jsonify({'success': False, 'message': 'DeepFace not installed. Please install: pip install deepface'})
        
        # Delete the pickle file to force refresh of representations
        pickle_file = os.path.join(app.config['UPLOAD_FOLDER'], 'representations_vgg_face.pkl')
        if os.path.exists(pickle_file):
            os.remove(pickle_file)
            print("Removed old representations file for fresh matching")
        
        # Search for matching face in uploads folder
        try:
            print(f"Searching for face match in: {app.config['UPLOAD_FOLDER']}")
            
            results = DeepFace.find(
                img_path=temp_filepath,
                db_path=app.config['UPLOAD_FOLDER'],
                distance_metric='cosine',
                enforce_detection=False,
                silent=True
            )
            
            print(f"DeepFace results type: {type(results)}")
            print(f"DeepFace results length: {len(results) if results else 'None'}")
            
            # Clean up temp file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            
            # Check if any matches found
            match_found = False
            user_info = None
            
            if results is not None and len(results) > 0:
                # results is a list of DataFrames
                df = results[0] if isinstance(results, list) else results
                print(f"DataFrame shape: {df.shape}")
                print(f"DataFrame columns: {df.columns.tolist()}")
                
                if len(df) > 0:
                    # Get the best match (first row, lowest distance)
                    best_match = df.iloc[0]
                    distance = best_match.get('VGG-Face_cosine', 1.0)
                    
                    print(f"Best match distance: {distance}")
                    
                    # Set a threshold for face matching (lower is better for cosine)
                    if distance < 0.4:  # Adjust this threshold as needed
                        match_found = True
                        matched_image_path = best_match['identity']
                        
                        print(f"Match found: {matched_image_path}")
                        
                        # Extract filename to find user in database
                        filename = os.path.basename(matched_image_path)
                        
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute('SELECT name, email FROM users WHERE image_path LIKE %s', (f'%{filename}%',))
                        user = cursor.fetchone()
                        cursor.close()
                        conn.close()
                        
                        if user:
                            user_info = {'name': user[0], 'email': user[1]}
                            print(f"User found: {user_info}")
                    else:
                        print(f"Distance too high: {distance} >= 0.4")
            
            if match_found:
                return jsonify({
                    'success': True, 
                    'message': 'Face authentication successful!',
                    'user': user_info
                })
            else:
                return jsonify({
                    'success': False, 
                    'message': 'Face not recognized. Please register first or try again with better lighting.'
                })
                
        except Exception as e:
            # Clean up temp file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            print(f"DeepFace error: {e}")
            return jsonify({
                'success': False, 
                'message': 'Face recognition failed. Please ensure good lighting and try again.'
            })
            
    except Exception as e:
        print(f"Face login error: {e}")
        return jsonify({
            'success': False, 
            'message': 'Login failed. Please try again.'
        })

@app.route('/register', methods=['POST'])
def register():
    try:
        # Get form data
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        image_data = request.form['image']
        
        # Validate required fields
        if not all([name, email, password, image_data]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        # Process and save image
        image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        
        # Generate unique filename
        filename = f"{uuid.uuid4().hex}_{email.replace('@', '_').replace('.', '_')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save image
        with open(filepath, 'wb') as f:
            f.write(img_bytes)
        
        # Hash password
        password_hash = generate_password_hash(password)
        
        # Save to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO users (name, email, password_hash, image_path)
            VALUES (%s, %s, %s, %s)
        ''', (name, email, password_hash, filepath))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Registration successful!'})
        
    except mysql.connector.IntegrityError:
        return jsonify({'success': False, 'message': 'Email already exists'})
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'success': False, 'message': f'Registration failed: {str(e)}'})

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Health check endpoint for Render
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Face Recognition App is running!'})

if __name__ == '__main__':
    setup_deepface()
    init_db()
    
    # Use PORT environment variable for deployment
    port = int(os.environ.get('PORT', 5000))
    
    # Run in production mode if deployed
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)