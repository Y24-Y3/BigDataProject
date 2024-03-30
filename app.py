from flask import Flask, request, jsonify
from main import main_py
app = Flask(__name__)

store = {
    'name': 'My Store',
    'items': [{
        'name': 'chair',
        'price': 15.99
    }]
}

app.register_blueprint(main_py)

if __name__ == '__main__':
    app.run()
    