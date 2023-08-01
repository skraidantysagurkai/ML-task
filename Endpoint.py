"""
    Endpoint created to serve the GDP per capita trained model.
"""
from flask import Flask, jsonify, request
import Final_model as Fm

app = Flask(__name__)


# Function to perform setup tasks before the first request
@app.before_first_request
def setup():
    """
        This function is the setup for the model before first request. It will be deprecated in Flask 2.3,
        but for now it will do.

        Note: The first request will take about 15s to load the model
            and then subsequent requests will be carried out quickly.
    """
    print('Setup started')
    model = Fm.create_model()
    app.config['my_model'] = model
    print('Setup Complete')


@app.route("/", methods=['POST'])
def get_input_give_result():
    """
        This function the app route for receiving and sending back json messages to and from the client.
        We use a XGBRegressor model to do predictions that are sent in a json string back to the client
        Note : the function returns a json string with a list inside consisting of status_code and result
        Example: {'GPD_per_capita': [{'status_code': 200}, {'result': '1234.56'}]}
    """

    model = app.config.get('my_model')
    if request.is_json:
        json_string = request.get_json()
    else:
        return jsonify({'error': 'JSON data not found'}), 400

    predictions = Fm.make_predictions(json_string, model)
    pred_list = predictions.tolist()

    return jsonify(GPD_per_capita=pred_list)


if __name__ == '__main__':
    app.run(debug=True)
