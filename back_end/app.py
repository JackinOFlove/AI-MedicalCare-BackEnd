from flask import Flask, request, jsonify
from ards_model_generate import Logic_Model
import torch

app = Flask(__name__)

# 加载预训练模型
logic_model = Logic_Model()
logic_model.load_state_dict(torch.load('ards_model.pth'))
logic_model.eval()

# 完整路由是 http://0.0.0.0:5000/predict
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    body_time_sample = torch.tensor(data['body_predicates_time'], dtype=torch.float64)

    with torch.no_grad():
        final_intensity, predicted_time = logic_model.predict_head_predicate_time(body_time_sample)

    return jsonify({
        'predicted_time': predicted_time
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)