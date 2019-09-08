from flask import Flask
from flask import redirect, url_for, request, Response, jsonify
from flask_cors import CORS
import json
import base64

#Files Import
from mnist import *
from iris import *
from predictnumber import *


app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def home():
    return "App is running on port 4000! :D"

@app.route('/tfpy', methods=['GET'])
def test():
    return "App is running on port 4000! :D"


#Get IRIS
@app.route('/worker/iris', methods=['POST'])
def irisPost():
    data = request.get_json()
    id = data["id"]
    data = data["vinnslItem"]["definition"]
    runIris(id, data)
    return "got Iris"


#Get Mnist
@app.route('/worker/mnist', methods=['POST'])
def mnistPost():
    data = request.get_json()
    id = data["id"]
    data = data["vinnslItem"]["definition"]
    runMnist(id, data)
    return "got Mnist"


#Get Wine
@app.route('/worker/wine', methods=['POST'])
def winePost():
    data = request.get_json()
    id = data["id"]
    data = data["vinnslItem"]["definition"]
    #runWine(id, data)
    return "got Wine"


#Create LSTM
@app.route('/worker/lstm', methods=['POST'])
def lstmPost():
    data = request.get_json()
    id = data["id"]
    data = data["vinnslItem"]["definition"]
    #createLSTM(id, data)
    return "created LSTM"


#Generate Text
@app.route('/worker/lstm/generate/text', methods=['POST'])
def postGenerateText():
    data = request.get_json()
    id = data["id"]
    data = data["vinnslItem"]["definition"]
    #generateText(id, data)
    return "Text generated"


#Save text in DB
@app.route('/worker/save/text/lstm', methods=['POST'])
def postSaveText():
    data = request.get_json()
    id = data["id"]
    data = data["vinnslItem"]["definition"]
    #saveTextInDB(id, data)
    return "Text saved in DB"


#PredictNumber
@app.route('/mnist/predictnumber', methods=['Post'])
def predictNumberFromImage():
    data = request.data
    data = json.loads(data)
    base64String = str(data["imageData"])
    base64String += '=' * (-len(base64String) % 4)    
    base64String = base64String[22:]
    
    base64String = base64String.encode()

    with open("image.png", "wb") as fh:
        fh.write(base64.decodebytes(base64String))

    pred = predictNumber()
    print("Bild ist: " + str(pred))

    #Send result back to server
    data = {'number': pred}
    return jsonify(data)


#Starte App
#app.run(host='0.0.0.0', port=4000)
app.run(host='vinnsl-nn-worker-python', port=4000)