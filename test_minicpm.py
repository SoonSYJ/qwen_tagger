import torch
import time
from PIL import Image
# from modelscope import AutoModel, AutoTokenizer
# from auto_gptq import AutoGPTQForCausalLM
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
# export VLLM_WORKER_MULTIPROC_METHOD=spawn

# model = AutoModel.from_pretrained(
#     'OpenBMB/MiniCPM-o-2_6',
#     trust_remote_code=True,
#     attn_implementation='sdpa', # sdpa or flash_attention_2
#     torch_dtype=torch.bfloat16,
#     init_vision=True,
#     init_audio=False,
#     init_tts=False
# )

# model = model.eval().cuda()
# tokenizer = AutoTokenizer.from_pretrained(
#     'OpenBMB/MiniCPM-o-2_6', trust_remote_code=True)

# model = AutoGPTQForCausalLM.from_quantized(
#     'OpenBMB/MiniCPM-o-2_6-int4',
#     torch_dtype=torch.bfloat16,
#     device="cuda:1",
#     trust_remote_code=True,
#     disable_exllama=True,
#     disable_exllamav2=True,
#     init_audio=False,
#     init_tts=False
# )
# model = model.eval().cuda()
# tokenizer = AutoTokenizer.from_pretrained(
#     'OpenBMB/MiniCPM-o-2_6-int4',
#     trust_remote_code=True
# )

# model = AutoModel.from_pretrained(
#     'OpenBMB/MiniCPM-V-2_6', 
#     trust_remote_code=True,
#     attn_implementation='sdpa', 
#     torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
# model = model.eval().cuda()
# tokenizer = AutoTokenizer.from_pretrained('OpenBMB/MiniCPM-V-2_6', trust_remote_code=True)

if __name__ == '__main__':
    # ===== vllm
    # MODEL_NAME = "/home/sunyujia/.cache/modelscope/hub/models/OpenBMB/MiniCPM-V-2_6"
    MODEL_NAME = "/home/sunyujia/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    llm = LLM(model=MODEL_NAME,
            gpu_memory_utilization=1,  # 使用全部GPU内存
            trust_remote_code=True,
            tensor_parallel_size=4,
            max_model_len=2048)  # 根据内存状况可调整此值

    image = Image.open('/home/sunyujia/python_ws/adas_test_data/car_in/1725287267033674000.png').convert('RGB')
    usr_prompt = '''你是一个智能驾驶机器人。根据输入图像描述当前驾驶场景。尽可能简洁。

    ## Workflow:
    1. **判断天气：** 从以下标签中选取一个天气标签，[晴天，阴天，雨天，雪天，雾霾天]。
    2. **判断道路类型：** 从以下标签中选取一个道路类型标签，[城市道路，高速公路，乡村道路]。
    3. **判断交通信号灯：** 从以下标签中选取一个交通信号灯标签，[红灯，绿灯，黄灯]。
    4. **判断行人：** 从以下标签中选取一个行人标签，[有行人，无行人]。
    5. **判断非机动车：** 从以下标签中选取一个非机动车标签，[有非机动车，无非机动车]。
    6. **判断车辆：** 从以下标签中选取一个车辆标签，[有车辆，无车辆]。
    7. **判断路口：** 从以下标签中选取一个路口标签，[有路口，无路口]。
    8. **判断交通标志：** 从以下标签中选取一个交通标志标签，[有交通标志，无交通标志]。
    9. **判断车道线：** 从以下标签中选取一个车道线标签，[有车道线，无车道线]。
    10. **判断障碍物：** 从以下标签中选取一个障碍物标签，[有障碍物，无障碍物]。
    11. **判断行驶方向：** 从以下标签中选取一个行驶方向标签，[左转，右转，直行]。
    12. **判断车速：** 从以下标签中选取一个车速标签，[慢速，中速，快速]。
    13. **判断路面状况：** 从以下标签中选取一个路面状况标签，[良好，较好，一般，较差，差]。
    14. **判断路面材质：** 从以下标签中选取一个路面材质标签，[沥青，水泥，土路，砖石]。

    ## Output Format:
    1. 天气标签: *
    2. 道路类型标签: *
    3. 交通信号灯标签: *
    4. 行人标签: *
    5. 非机动车标签: *
    6. 车辆标签: *
    7. 路口标签: *
    8. 交通标志标签: *
    9. 车道线标签: *
    10. 障碍物标签: *
    11. 行驶方向标签: *
    12. 车速标签: *
    13. 路面状况标签: *
    14. 路面材质标签: *'''

    # msgs = [{'role': 'user', 'content': [image, usr_prompt]}]
    # for i in range (10):
    #     t0 = time.time()
    #     res = model.chat(
    #         image=None,
    #         msgs=msgs,
    #         tokenizer=tokenizer
    #     )
    #     print(res)
        
    #     print(i, time.time() - t0)

    messages = [{'role': 'user', 'content': '(<image>./</image>)\n' + usr_prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    stop_tokens = ['<|im_end|>', '<|endoftext|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    # 设置生成参数
    sampling_params = SamplingParams(
        stop_token_ids=stop_token_ids,
        # temperature=0.7,
        # top_p=0.8,
        # top_k=100,
        # seed=3472,
        max_tokens=1024,
        # min_tokens=150,
        temperature=0,
        # use_beam_search=True,
        # length_penalty=1.2,
        best_of=1)

    # 获取模型输出
    for i in range (10):
        t0 = time.time()
        outputs = llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": image}},
             {"prompt": prompt, "multi_modal_data": {"image": image}}], 
            sampling_params=sampling_params)
        print(outputs[0].outputs[0].text)
        print(i, time.time() - t0)