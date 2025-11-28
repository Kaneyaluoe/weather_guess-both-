import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
import random

# 忽略警告，保持控制台清洁
warnings.filterwarnings("ignore")

# 1. 初始化 Flask 应用
app = Flask(__name__)
# 2. 启用 CORS，允许前端网页访问
CORS(app)

# --- 定义 LSTM 模型 ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, output_size=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- 旅游推荐数据模拟 ---
def get_travel_recommendations(city_name):
    # 简单的模拟数据库，实际可连接 Google Places API 或 Yelp
    city_db = {
        "Beijing": {
            "food": [
                {"name": "北京烤鸭", "desc": "酥脆外皮，传统果木挂炉技艺。", "icon": "🦆"},
                {"name": "炸酱面", "desc": "老北京地道面食，酱香浓郁。", "icon": "🍜"}
            ],
            "spots": [
                {"name": "故宫博物院", "desc": "世界最大宫殿建筑群。", "tag": "历史"},
                {"name": "798艺术区", "desc": "现代艺术与工业风的完美结合。", "tag": "艺术"}
            ]
        },
        "Shanghai": {
            "food": [
                {"name": "小笼包", "desc": "皮薄汁多，南翔特色。", "icon": "🥟"},
                {"name": "生煎馒头", "desc": "底脆肉鲜，撒上葱花芝麻。", "icon": "🥠"}
            ],
            "spots": [
                {"name": "外滩", "desc": "万国建筑博览群，夜景迷人。", "tag": "地标"},
                {"name": "豫园", "desc": "江南古典园林，精致典雅。", "tag": "园林"}
            ]
        },
        "Tokyo": {
            "food": [
                {"name": "寿司 (Sushi)", "desc": "筑地市场新鲜直供。", "icon": "🍣"},
                {"name": "豚骨拉面", "desc": "浓郁骨汤，弹牙面条。", "icon": "🍜"}
            ],
            "spots": [
                {"name": "浅草寺", "desc": "东京最古老的寺庙。", "tag": "文化"},
                {"name": "涩谷路口", "desc": "世界最繁忙的十字路口。", "tag": "都市"}
            ]
        },
        "Paris": {
            "food": [
                {"name": "法式牛角包", "desc": "层层酥脆，黄油香气。", "icon": "🥐"},
                {"name": "马卡龙", "desc": "少女的酥胸，甜点中的贵族。", "icon": "🍪"}
            ],
            "spots": [
                {"name": "埃菲尔铁塔", "desc": "巴黎铁娘子，浪漫象征。", "tag": "地标"},
                {"name": "卢浮宫", "desc": "蒙娜丽莎的微笑所在地。", "tag": "艺术"}
            ]
        }
    }

    # 模糊匹配或返回默认值
    key = None
    for k in city_db:
        if k.lower() in city_name.lower():
            key = k
            break
    
    if key:
        return city_db[key]
    else:
        # 通用兜底数据
        return {
            "food": [
                {"name": "当地特色小吃", "desc": "探索街头巷尾的隐藏美味。", "icon": "🍢"},
                {"name": "精选料理", "desc": "主厨推荐的时令佳肴。", "icon": "🍽️"}
            ],
            "spots": [
                {"name": "城市中心公园", "desc": "感受当地的生活节奏。", "tag": "休闲"},
                {"name": "历史博物馆", "desc": "了解这座城市的过去。", "tag": "文化"}
            ]
        }

# --- 核心业务逻辑 ---
def run_ai_analysis(city_name):
    # 模拟生成数据 (实际会连接数据库)
    dates = pd.date_range('2025-11-01', periods=10)
    # 稍微添加一点随机性
    base_temp = 20 + random.randint(-10, 10)
    temp_data = [base_temp + random.randint(-3, 3) for _ in range(10)]
    precip_data = [random.choice([0, 0, 0, 5, 15]) for _ in range(10)]
    
    df = pd.DataFrame({'temp': temp_data, 'precip': precip_data}, index=dates)

    # 1. ARIMA 预测
    try:
        model_arima = ARIMA(df['temp'], order=(1,1,1))
        model_fit = model_arima.fit()
        forecast_arima = model_fit.forecast(steps=5).tolist()
    except:
        forecast_arima = [base_temp] * 5

    # 2. LSTM 预测
    lstm_temps = []
    lstm_rain = []
    try:
        data = df.values.astype(np.float32)
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        X, y = [], []
        for i in range(len(data_scaled) - 3):
            X.append(data_scaled[i:i+3])
            y.append(data_scaled[i+3])
        
        if len(X) > 0:
            X_tensor = torch.from_numpy(np.array(X))
            y_tensor = torch.from_numpy(np.array(y))

            model = LSTMModel()
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            for _ in range(30): # 减少训练轮数加快响应
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()

            last_seq = torch.from_numpy(data_scaled[-3:]).unsqueeze(0)
            forecast_scaled = []
            for _ in range(5):
                pred = model(last_seq)
                forecast_scaled.append(pred.detach().numpy())
                pred_expanded = pred.unsqueeze(1)
                last_seq = torch.cat((last_seq[:, 1:, :], pred_expanded), dim=1)

            forecast_lstm = scaler.inverse_transform(np.concatenate(forecast_scaled, axis=0))
            lstm_temps = forecast_lstm[:, 0].tolist()
            lstm_rain = forecast_lstm[:, 1].tolist()
        else:
            raise Exception("Data too short")
            
    except Exception as e:
        print(f"LSTM Fallback: {e}")
        lstm_temps = [base_temp] * 5
        lstm_rain = [0] * 5

    # 3. 穿衣建议
    avg_temp = sum(lstm_temps) / len(lstm_temps)
    if avg_temp > 25:
        suggestion = "AI 建议: 热浪来袭，建议穿着清凉透气的衣物。"
    elif avg_temp > 15:
        suggestion = "AI 建议: 气候舒适，T恤搭配薄外套即可。"
    elif avg_temp > 5:
        suggestion = "AI 建议: 天气转凉，请穿着风衣或夹克。"
    else:
        suggestion = "AI 建议: 严寒预警，请务必穿着羽绒服保暖。"

    # 4. 获取旅游推荐
    travel_data = get_travel_recommendations(city_name)

    return {
        "status": "success",
        "arima_forecast": forecast_arima,
        "lstm_forecast": {
            "temp": [round(x, 1) for x in lstm_temps],
            "rain": [round(x, 1) for x in lstm_rain]
        },
        "advice": suggestion,
        "travel": travel_data
    }

# --- 路由 ---
@app.route('/', methods=['GET'])
def home():
    return "OmniWeather 后端服务器正在运行！"

@app.route('/api/analyze', methods=['GET'])
def api_analyze():
    city = request.args.get('city', 'Beijing')
    print(f"收到前端请求: 分析城市 {city}...")
    data = run_ai_analysis(city)
    return jsonify(data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)