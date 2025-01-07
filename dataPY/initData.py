import json

test_data = [
    {
        "text": "我想了解产品的功能介绍",
        "label": 0  # 0代表产品咨询类
    },
    {
        "text": "我要投诉你们的售后服务",
        "label": 1  # 1代表售后投诉类
    },
    {
        "text": "怎么申请退款？",
        "label": 2  # 2代表退款相关类
    },
    {
        "text": "产品的价格能优惠吗？",
        "label": 3  # 3代表价格咨询类
    }
]

# 将测试数据保存为JSON文件
with open('huggingface_customer_service_test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)