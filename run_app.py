import os, subprocess

os.environ['FLASK_APP'] = "./vis_app/app.py"
os.environ['FLASK_DEBUG'] = "1"
subprocess.call(['flask', 'run'])
