from flask import Flask, request, jsonify
from ards_model_generate import Logic_Model
import torch

app = Flask(__name__)
import time

# 加载预训练模型
# 加载并查看保存的模型文件内容
#saved_data = torch.load('ards_model.pt')
#print(type(saved_data))  # 检查保存的数据类型

# 打印保存的内容，如果是模型参数，应该是一个字典
#print(saved_data)

start_time = time.time()
logic_model = Logic_Model()
# 加载模型
# 从文件中加载模型状态
checkpoint = torch.load('ards_model.pt',map_location=torch.device('cpu'))

# 恢复 relation 和 model_parameter
logic_model.relation = checkpoint['relation']  # 恢复 relation 字典
logic_model.model_parameter = checkpoint['model_parameter']  # 恢复 model_parameter 字典
print(logic_model.model_parameter)
logic_model.eval()

print(logic_model.relation)

end_time = time.time()
print(f"加载预训练模型耗时: {end_time - start_time} 秒")


# 完整路由是 http://0.0.0.0:5000/predict
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    body_time_sample = torch.tensor(data['body_predicates_time'], dtype=torch.float64)

    with torch.no_grad():
        final_intensity, predicted_time,base,wight = logic_model.predict_head_predicate_time(body_time_sample)
        
    return jsonify({
        'predicted_time': predicted_time,
        'final_intensity': final_intensity.tolist(),
        'base':base.tolist(),
        'wight':wight.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)