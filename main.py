# main.py
from flask import flash
app=Flask(__name__)

@app.route('/')
def hello_World():
    return 'Hello, World!'

if __name__ == '__main__':
    
    app.run()