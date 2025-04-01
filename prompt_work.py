from openai import OpenAI
import llm_api

client = OpenAI(api_key="sk-645c13592ea34fc38c776779a0c01b77", base_url="https://api.deepseek.com")


class QueryStandardizer:  # 问题标准化类
    # """
    # 问题标准化类，用于将问题中的信息标准化。从而更好的匹配到数据库中的问诊记录。
    # """
    # global field_list
    # field_list = [
    #     "患者性别",  # 患者性别，男或女
    #     "患者年龄",  # 患者年龄，具体年龄或年龄段
    #     "症状表现",  # 具体症状，如体重增加、食量大、懒惰、不爱运动等
    #     "患儿体型",  # 体型描述，如肥胖、超重等
    #     "病史",  # 相关的病史信息，如是否有其他健康问题
    #     "父母观察",  # 父母或照顾者的观察结果
    #     "治疗需求",  # 需要的治疗方式，如饮食控制、运动、药物治疗等
    #     "饮食习惯",  # 孩子的饮食习惯描述，如暴饮暴食、不规律等
    #     "运动习惯",  # 孩子的运动习惯，是否积极参与运动
    #     "环境因素",  # 环境影响，如是否有促使肥胖的因素（如家庭饮食环境等）
    #     "期望治疗效果",  # 对治疗的期望，如减轻体重、改善生活质量等
    #     "是否已尝试治疗",  # 是否已经尝试过任何治疗方式及其效果
    #     "注意事项",  # 治疗过程中的注意事项或生活指导
    #     "医学领域",  # 相关的医学领域，如儿童营养、内分泌、肥胖症等
    #     "治疗方法推荐",  # 根据问题推荐的治疗方法（如药物、运动、饮食调节等）
    #     "治疗效果",  # 治疗效果的预期和反馈（例如治疗后体重变化等）
    #     "家长态度",  # 家长对治疗的态度和配合度
    #     "长期管理",  # 长期管理建议，如如何保持健康体重、预防复发等
    #     "问题描述",  # 用户提问的详细描述
    # ]

    def standardize_query(self, input_data):  # 从问题中提取字段列表对应的信息，标准化后输出。
        """
        :param input_data:
        :param field_list: 字段列表。
        :return: 标准化后的医学问题
        """
        # all_fields_str = ", ".join(field_list)
        extract_template = f"""
        给你一个医学问题，这个问题可能不标准，请你按照参考格式对问题进行标准化。
        问题：{input_data}
        用中文回复以及中文字符，标准化时时参考以下格式，
        参考1：【女宝宝，刚7岁，这一年，察觉到，我家孩子身上肉很多，而且，食量非常的大，平时都不喜欢吃去玩，请问：小儿肥胖超重该如何治疗。】
        参考2：【男孩子，刚4岁，最近，发现，我家孩子体重要比别的孩子重很多，而且，最近越来越能吃了，还特别的懒，请问：小儿肥胖超重该怎样医治。】
        参考3：【我家的孩子是男宝宝，7岁，这一年，觉得，孩子身上越来越肉乎了，另外，一顿可以吃好几碗饭，喜欢吃完饭就躺着，请问：小儿肥胖懒怎样诊治。】
        用上述参考格式输出标准化后的问题，注意，对于不存在的信息忽略即可，不要说“未提及”等信息，不要产生幻觉，“请问”之后的内容可以是对前面内容的总结。
        """

        """
        我家小孩吃了三个橘子之后喝了一袋牛奶，现在肚子疼，怎么办
        我儿子看了一天电视，现在眼睛很痛，看不清东西，怎么办
        我闺女老是吃辣条，然后喝凉水，今天吃完特别辣的辣条后又喝很多凉水，然后一直拉稀，怎么办
        """

        extract_template = extract_template.replace("\n", "。")
        # answer = llm_api.ollama_generate_by_api(extract_template)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": extract_template},
            ],
            stream=False,
            temperature=0.1,
        )
        standardized_question = response.choices[0].message.content
        return standardized_question


def judge_question_type(input_data):  # 判断问题类型
    # 检查用户输入的问题是否与医学相关。
    """
    判断用户输入的问题是否与药品或药学相关。

    :param input_data: 用户输入的问题字符串。
    :return: "good"（医学相关问题）或 "bad"（非医学相关问题）。
    """
    judge_template = f"""你是医生，请判断以下问题是否与医学相关：
            {input_data}
            如果这是一个需要医生来解决的问题（如提问生病症状或身体不舒服等），返回 "good"；
            如果不是医生能解决的问题，返回 "bad"。
            只能返回“good”或者“bad”
            """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": judge_template},
        ],
        stream=False,
        temperature=0.05,
    )
    query_type = response.choices[0].message.content
    # 确保只返回 "good" 或 "bad"
    if query_type not in ["good", "bad"]:
        return "未知类型"
    return query_type

def genertate_answer(question, document):  # 生成答案
    # 根据问题和数据库中的问诊记录生成答案。
    """
    根据用户输入的问题和数据库中的问诊记录生成答案。

    :param input_data: 用户输入的问题字符串。
    :param document: 问诊记录。
    :return: 答案字符串。
    """



if __name__ == '__main__':
    standardizer = QueryStandardizer()
    while True:
        input_data = input("请输入问题：")
        query_type = judge_question_type(input_data)
        print("问题类型：" + query_type)
        final_output = standardizer.standardize_query(input_data)
        print(final_output)
