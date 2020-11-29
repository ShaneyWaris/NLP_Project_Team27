from flask import Flask, render_template, request, url_for, redirect
import os
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def hello_world():
    if request.method == 'POST':
        headline = request.form.get('headline')
        body = request.form.get('body')

        # func


    return render_template('index.html')







port = int(os.getenv('PORT', 8000))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
