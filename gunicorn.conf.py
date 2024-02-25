# gunicorn.conf.py

# Bind to the specified host and port
bind = '127.0.0.1:8000'

# Number of worker processes (adjust based on your server's CPU cores)
workers = 3

# Specify the Python path to your Flask app
pythonpath = 'app.py'

# Specify the WSGI entry point for your Flask app
app = 'app:app'

# Set the log level (debug, info, warning, error, critical)
loglevel = 'info'

# Configure logging to a file
errorlog = '/path/to/gunicorn/error.log'
accesslog = '/path/to
bfh-cxgz-szy
