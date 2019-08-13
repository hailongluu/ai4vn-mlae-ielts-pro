from flask import Flask, request
import ai_adapter, data_collector

app = Flask(__name__)


@app.route("/home")
def welcome_home():
    return "welcome api main board"


@app.route("/score/")
def get_score():
    text = request.args.get("text")
    response = ai_adapter.get_score_reports(text)
    return response


@app.route("/exam/get_full")
def get_full_exam():
    id = request.args.get("id")
    response = ai_adapter.get_all_exam(id)
    return response


@app.route("/exam/get_writing")
def get_writing_exam():
    id = request.args.get("id", 0)
    return data_collector.get_writing_exam(int(id))


app.run(host="192.168.10.150", port=5000, debug=True)
