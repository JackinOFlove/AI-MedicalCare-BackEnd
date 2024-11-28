from flask import Flask, request, jsonify
from aki.aki_model_generate import Logic_Model as AKI_Logic_Model
from ards.ards_model_generate import Logic_Model as ARDS_Logic_Model
import torch

app = Flask(__name__)
import time

# 加载预训练模型

# 加载aki模型
start_time_aki = time.time()

aki_logic_model = AKI_Logic_Model()
aki_checkpoint = torch.load('aki/aki_model.pt', map_location=torch.device('cpu'))
aki_logic_model.relation = aki_checkpoint['relation']
aki_logic_model.model_parameter = aki_checkpoint['model_parameter']
aki_logic_model.eval()

end_time_aki = time.time()
print(f"加载aki模型耗时: {end_time_aki - start_time_aki} 秒")

# 加载ards模型
start_time_ards = time.time()

ards_logic_model = ARDS_Logic_Model()
ards_checkpoint = torch.load('ards/ards_model.pt', map_location=torch.device('cpu'))
ards_logic_model.relation = ards_checkpoint['relation']
ards_logic_model.model_parameter = ards_checkpoint['model_parameter']
ards_logic_model.eval()

end_time_ards = time.time()
print(f"加载ards模型耗时: {end_time_ards - start_time_ards} 秒")

# 完整路由是 http://0.0.0.0:5000/predict
# 请求体是json，包含body_predicates_time和item
# body_predicates_time是numpy数组，表示体谓词时间
# item是字符串，表示模型类型，取值为aki或ards
# 返回json，包含predicted_time, final_intensity, base, weight
# 返回的predicted_time表示预测的头部谓词时间
# ards:body_predicates_time为一个47维的numpy数组，aki:body_predicates_time为一个37维的numpy数组
# ards和aki的体谓词对应关系在ards/ards_model_generate.py和aki/aki_model_generate.py中
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    body_time_sample = torch.tensor(data['body_predicates_time'], dtype=torch.float64)
    item = data.get('item')

    with torch.no_grad():
        if item == 'aki':
            final_intensity, predicted_time, base, weight = aki_logic_model.predict_head_predicate_time(body_time_sample)
        elif item == 'ards':
            final_intensity, predicted_time, base, weight = ards_logic_model.predict_head_predicate_time(body_time_sample)
        else:
            return jsonify({'error': 'Invalid item type'}), 400
        
    return jsonify({
        'predicted_time': predicted_time,
        'final_intensity': final_intensity.tolist(),
        'base': base.tolist(),
        'weight': weight.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)