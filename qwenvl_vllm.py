import os
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from modelscope import snapshot_download
# import multiprocessing
# multiprocessing.set_start_method('spawn')
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

if __name__ == '__main__':
    # MODEL_PATH = "/home/sunyujia/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"
    MODEL_PATH = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
    llm = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"image": 10, "video": 10},
        tensor_parallel_size=4,
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=256,
        stop_token_ids=[],
    )

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
    
    image_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/home/sunyujia/python_ws/adas_test_data/car_in/1725287267033674000.png",
                    "min_pixels": 224 * 224,
                    "max_pixels": 1280 * 28 * 28,
                },
                {"type": "text", "text": usr_prompt},
            ],
        },
    ]

    # Here we use video messages as a demonstration
    messages = image_messages

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }

    for i in range(10):
        outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text

        print(generated_text)