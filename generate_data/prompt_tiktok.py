import time

import openai
import sys
import random
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

openai.api_key = openai_key   # you must provide your OpenAI API key before crawling
if not openai.api_key:
  raise ValueError("OpenAI API key not provided. Please set the 'openai.api_key' variable.")


def return_random_prompt(context_str):
  system_prompt = "下面给出的上下文是抖音带货的运营、主播话术和直播间选品等相关的教学内容脚本，你需要尽可能给出多样化的任务指令、上下文和对应的回答。我们将用于人工评估ChatGPT模型对指令的完成情况。" \
                  "-------------------------------------------------\n"\
                  "%s\n"\
                  "-------------------------------------------------\n" \
                  "要求:\n"


  system_prompt += "1. 主题多样化，涵盖各个方面，从学员角度提问，问题都是跟实际问题相关的。\n"
  task_list = ["开放式生成", "分类", "问答", "编辑", "摘要", "写作", "分析",  "抽取", "推荐", "直播话术生成"]
  system_prompt += "2. 表述多样化，结合真实问题；指令类型多样化，例如：" + "、".join(task_list) + "等。\n"

  # other requirements
  system_prompt += "3. 如果遇到无法处理的指令（只靠文本无法回答），给出无法处理的回复。\n"
  system_prompt += "4. 所有回答尽可能的一点一点的罗列清晰，有教学意义，回答给出例子详细说明。\n"
  system_prompt += "5. 所有回答都要解释。\n"
  system_prompt += "7. 除非特别要求，请使用中文，指令可以是命令句、疑问句、或其他合适的类型。\n"
  system_prompt += "8. 为指令生成一个适当且涉及真实情况的<instruction>，不应该只包含简单的占位符。<instruction>应提供实质性的内容，具有挑战性。\n"
  system_prompt += "9. <input>为空。\n"
  system_prompt += "10. <output>应该是对指令的适当且真实的回应，<input>是指令所引用的上下文都为空，不能只回复答应或拒绝请求。如果需要额外信息才能回复时，请努力预测用户意图并尝试回复。<output>的内容应少于" + str(random.randint(400, 512)) + "字。\n\n"

  system_prompt += "请给出满足条件的10条JSON格式数据，输出格式和样例如下，对于输出结果请检查json格式的准确性，并修正结果：\n"
  system_prompt += "例如："

  system_prompt += """ [
  {
    "instruction": "我们如何在日常生活中减少用水？",
    "input": "",
    "output": "1. 使用节水装置，如节水淋浴喷头和水龙头。 \\n2. 使用水箱或水桶收集家庭废水，例如洗碗和洗浴。 \\n3. 在社区中提高节水意识。 \\n4. 检查水管和灌溉系统的漏水情况，并及时修复它们。 \\n5. 洗澡时间缩短，使用低流量淋浴头节约用水。 \\n6. 收集雨水，用于园艺或其他非饮用目的。 \\n7. 刷牙或擦手时关掉水龙头。 \\n8. 减少浇水草坪的时间。 \\n9. 尽可能多地重复使用灰水（来自洗衣机、浴室水槽和淋浴的水）。 \\n10. 只购买能源效率高的洗碗机和洗衣机。"
  },
  {
    "instruction": "编辑文章，使其更吸引读者。",
    "input": "",
    "output": "自主机器人是计算机控制的机器，被编程执行特定任务而不需要任何人类输入，从而实现了新的效率、精确度和可靠性水平。自主机器人在各个行业中被越来越广泛地应用，从制造业，它们可以使用精度和一致的质量组装复杂的组件，到医疗保健，可以协助进行医疗测试和处理，再到安全，可以监控大面积地区，保障人们和财产的安全。自主机器人还可以减少在危险或有害环境中的错误和增加安全，在工业流程的检查或维修期间等。由于其多样性，自主机器人将彻底改变我们工作方式的方式，使任务变得更加简单、快速，最终更加愉悦。"
  },
  {
    "instruction": "政府可以采取哪些策略来减少空气污染？",
    "input": "",
    "output": "1. 实施强制的车辆排放标准和基于激励的计划，以降低车辆的碳足迹。\\n2. 增加公共交通工具，减少公众对车辆的依赖。\\n3. 增加对空气污染的影响的认识，鼓励市民减少污染物的生成。\\n4. 投资于可再生能源的研究和开发，如太阳能和风能。\\n5. 在工厂和发电厂安装空气污染控制装置，例如洗涤器。\\n6. 对车辆和工厂使用清洁燃料。\\n7. 实施更好的城市规划和控制拓展。\\n8. 改善农业效率，减少化肥和杀虫剂的使用。\\n9. 种植更多的树木以减少空气污染。\\n10. 减少木材、煤炭和生物质的燃烧。"
  }
  ]

 """
  print(system_prompt)
  return system_prompt%context_str


def generate_data(directory_path):
  loader = DirectoryLoader(directory_path, glob='*.pdf', loader_cls=PyPDFLoader)
  documents = loader.load()
  context_str = ""
  file_index = 0
  max_content = 8000
  for i in range(0, len(documents)):
      try:
          context_str_ = context_str + documents[i].page_content

          if len(context_str_) > max_content:
            response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo-16k",  # here we use `gpt-3.5-turbo` model, while Stanford-Alpaca uses `text-davinci-003`
              messages=[
                {"role": "user", "content": return_random_prompt(context_str)},
              ]
            )
            output_file = open("data/tiktok_data_%d.json" % file_index, 'w', encoding="utf8")
            print(response["choices"][0]["message"]["content"])
            output_file.write(response["choices"][0]["message"]["content"] + '\n')
            output_file.close()
            file_index+=1
            context_str = documents[i].page_content

          else:
            context_str = context_str_
      except Exception as e:
        import traceback
        print( traceback.format_exc())
        print("error", " document :", i)
        time.sleep(60)

      print("document :", i)


if __name__ == "__main__":

  directory_path = '../douyin/zhubo'
  generate_data(directory_path)
