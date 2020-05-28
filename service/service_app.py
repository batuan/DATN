from flask import Flask
from service.service_classify_content import classify
from service.service_classify_url import classify_url


app = Flask(__name__)

app.register_blueprint(classify)
app.register_blueprint(classify_url)
