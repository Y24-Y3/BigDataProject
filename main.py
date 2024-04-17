from flask import Blueprint, request, jsonify, render_template

main_py = Blueprint('main', __name__)

@main_py.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@main_py.route('/api/', methods=['POST'])
def api():
    data = request.get_json()
    return jsonify(data)