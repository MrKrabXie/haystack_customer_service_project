import json

test_data = [
    {
        "question": "产品的保修期是多久？",
        "answer": "我们产品的保修期为一年，自购买之日起计算。在保修期内，若产品出现非人为质量问题，我们将提供免费维修服务。"
    },
    {
        "question": "如何联系客服？",
        "answer": "您可以通过拨打客服热线400-123-4567联系我们，也可以在我们的官方网站上提交在线客服咨询表单。"
    },
    {
        "question": "产品支持哪些支付方式？",
        "answer": "我们支持多种支付方式，包括微信支付、支付宝支付、银行卡支付以及信用卡支付。"
    }
]

# 将测试数据保存为JSON文件
with open('../data/haystack_customer_service_test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)