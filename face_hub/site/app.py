from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import base64
from prediction import Model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow import Session, get_default_graph
from tensorflow.keras.backend import set_session

sess = Session()
graph = get_default_graph()

set_session(sess)
model = Model()


def create_app():
  app = Flask(__name__)
  Bootstrap(app)

  return app

app = create_app()


@app.route('/', methods = ['GET'])
def home():
    return render_template(
        template_name_or_list = 'index.html'
    )

@app.route('/predict', methods=['GET', 'POST'])
def predictor():
    if request.method == 'GET':
        return render_template('predict.html', predictor='active')
    file = request.files["Portrait"]
    bytes = file.read()
    b64_string = base64.b64encode(bytes)
    b64_data = "data:" + file.content_type + ";base64," + str(b64_string)[2:-1]

    img = image.img_to_array(image.load_img(file, target_size=(64,64)))
    img = img.reshape(1,64,64,3)

    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        features = model.model.predict(img)
    columns = []
    for i, col in enumerate(model.columns):
        for prediction in features:
            if prediction[i] == 1:
                columns.append(model.columns[i])
    columns = ", ".join(columns)

    return render_template('predict.html', img_b64=b64_data, columns=columns)


if __name__ == '__main__':
    app.run()