import os
import logging

from flask import Flask, send_from_directory

# Configure basic logging for the Flask app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Determine the absolute path to the src directory
# This is important if main.py is not in the project root but, for example, in src/ itself.
# However, standard Flask practice is to run the app from the project root where main.py resides.
# If main.py is in PROJECT_ROOT:
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
# If main.py is in PROJECT_ROOT/src (and you want to serve from PROJECT_ROOT/src/static or similar):
# SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure the SRC_DIR is correctly pointing to where index.html is located.
# Based on previous context, main.py is in the project root, and index.html is in src/.

@app.route('/')
def serve_index():
    # Log the request to the index page
    app.logger.info(f'Serving index.html from {SRC_DIR}')
    try:
        return send_from_directory(SRC_DIR, 'index.html')
    except Exception as e:
        app.logger.error(f"Error serving index.html: {e}")
        return "Error loading page. Check server logs.", 500

def main():
    # Get Flask debug mode from environment variable
    # Common practice: FLASK_DEBUG=1 for True, FLASK_ENV=development also implies debug=True
    # Using a custom variable for clarity here.
    flask_debug_mode = os.environ.get('APP_FLASK_DEBUG_MODE', 'False').lower() in ('true', '1', 't')

    app.logger.info(f"Starting Flask development server (Debug Mode: {flask_debug_mode})...")
    app.logger.info(f"Attempting to serve index.html from directory: {SRC_DIR}")
    if not os.path.exists(os.path.join(SRC_DIR, 'index.html')):
        app.logger.warning(f"WARNING: index.html not found at {os.path.join(SRC_DIR, 'index.html')}. Requests to '/' will fail.")
    
    app.run(debug=flask_debug_mode, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()
