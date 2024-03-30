from flask import Blueprint, request, jsonify

main_py = Blueprint('main', __name__)

@main_py.route('/', methods=['GET'])
def home():
    return "Test from main blueprint"

@main_py.route('/api/', methods=['POST'])
def api():
    data = request.get_json()
    return jsonify(data)