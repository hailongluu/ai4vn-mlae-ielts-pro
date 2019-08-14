from flask import Flask, request, jsonify
import data_collector
from ai_adapter import APIAdapter

# import modules.grammar.score_grammar

app = Flask(__name__)
api_adapter = APIAdapter()


# run_with_ngrok(app)


@app.route("/home")
def welcome_home():
    return "welcome api main board"


@app.route("/score/")
def get_score():
    topic = request.args.get("topic")
    text = request.args.get("text")
    response = api_adapter.get_score_reports(topic, text)
    return jsonify(response)


@app.route("/exam/get_full")
def get_full_exam():
    id = request.args.get("id")
    response = api_adapter.get_all_exam(id)
    return response


@app.route("/exam/get_topic")
def get_topic():
    topics = data_collector.get_topic(20)
    return jsonify(topics)


@app.route("/exam/get_writing")
def get_writing_exam():
    id = request.args.get("id", 0)
    return data_collector.get_writing_exam(int(id))


if __name__ == "__main__":
    app.run(port=5000, debug=True, use_reloader=False)
    # app.run()
