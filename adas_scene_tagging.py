import time
import torch
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


scene_tags_l1 = {
    "urban road": "00",
    "country road": "01",
    "highway": "02",
}

scene_tags_l2 = {
    "normal road": "00",
    "road under construction": "03",
    "accident section": "04",
    "intersection": "020",
    "bend": "030",
    "slope": "040",
    "ramp": "050",
    "roundabout": "060",
    "pedestrian overpass": "070",
    "bridge": "080",
    "tunnel": "090",
    "parking area": "100",
    "internal road": "110",
    "harbour": "120",
    "gas station": "130",
    "toll station": "140",
    "service area": "150",
}

scene_tags_l3 = {
    "drive on the main road": "000",
    "drive on the side road": "001",
    "drive up a slope": "040",
    "drive down a slope": "041",
    "enter the ramp": "050",
    "exit the ramp": "051",
    "driving on the ramp": "052",
    "enter the tunnel": "090",
    "exit the tunnel": "091",
    "drive in the tunnel": "092"
}

weather_tags = {
    "sunny": "00",
    "cloudy": "01",
    "overcast": "02",
    "rainy": "03",
    "snowy": "04",
    "foggy": "05",
    "extreme weather": "06",
    "night": "07",
}

tl1 = ['日间', '夜间']
tl2 = ['晴天', '雨天', '大雾', '雪天', '阴天']
tl3 = ['城市道路', '乡村道路', '高速道路']
tl4 = ['主路', '辅路', '停车场', '加油站', '收费站', '服务区', '港口', '内部路']
tl5 = ['环岛', '隧道', '匝道', '坡道', '过街天桥', '桥梁']
tl6 = ['路口', '事故路段', '工程路段']
tl7 = ['转弯', '上坡', '下坡']

tl1_en = ['daytime', 'night']
tl2_en = ['sunny', 'rainy', 'foggy', 'snowy', 'overcast']
tl3_en = ['urban road', 'country road', 'highway']
tl4_en = ['main road', 'side road','parking area', 
          'gas station', 'toll station', 'service area', 'harbour', 'internal road']
tl5_en = ['normal', 'intersection', 'roundabout', 'tunnel', 'ramp', 'slope', 'pedestrian overpass', 'bridge']
tl6_en = ['normal', 'accident section', 'road under construction']
tl7_en = ['normal', 'drive on a bend', 'drive up a slope', 'drive down a slope']


class Tagger:
    def __init__(self, vlm, max_token_gen=1024) -> None:
        if vlm == "7b_q4":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit", 
                torch_dtype="auto", device_map="auto"
            )
        
            self.processor = AutoProcessor.from_pretrained("unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit")
    
        elif vlm == "7b":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        elif vlm == "3b":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )

            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
            
        self.device = self.model.device
        self.max_token_gen = max_token_gen
    
    def get_message(self, prompt, image=None):
        message = [
            {"role": "user", 
             "content": [
                {"type": "image", 
                 "image": image},
                {"type": "text", 
                 "text": prompt}
            ]}
        ]   
        
        return message

    def scene_description(self, image):
        prompt = f"""You are a autonomous driving labeller. \
            You have access to these front-view camera images of a car taken at a 0.5 second interval over the past 5 seconds. \
            Imagine you are driving the car. \
            Describe the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings."""

        result = self.inference(text=prompt, image=image)
        return result

    def objects_description(self, image):
        prompt = f"""You are a autonomous driving labeller. \
            You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. \
            Imagine you are driving the car. \
            What other road users should you pay attention to in the driving scene? \
            List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you."""

        result = self.inference(text=prompt, image=image)

        return result

    def scene_tagging_by_level(self, image, tag_l="l2"):
        if tag_l == "l2":
            tags = ",".join(t for t in scene_tags_l2.keys())
        elif tag_l == "l3":
            tags = ",".join(t for t in scene_tags_l3.keys())
        elif tag_l == "l1":
            tags = ",".join(t for t in scene_tags_l1.keys())

        prompt = f"""You are a autonomous driving labeller. \
            You have access to a front-view camera images of a vehicle. \
            Describe the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings.\
            Select a class from the following list {tags}."""

        result = self.inference(text=prompt, image=image)

        return result
    
    def scene_tagging_if_bend(self, image):
        prompt = f"""You are a autonomous driving labeller.\
            You have access to a front-view camera images of a vehicle.\
            Detect the lane marking.\
            Answer if the road is a bend or not.\
            """

        result = self.inference(text=prompt, image=image)

        return result
    
    def scene_tagging_zh(self, image):
        prompt = f"""你是一个自动驾驶机器人。\
            根据当前输入的图像，描述当前道路情况，具体步骤如下。\
            第一步，从（{"，".join(tl1)}）中选择一个标签。\
            第二步，从（{"，".join(tl2)}）中选择一个标签。\
            第三步，根据道路标志牌，车辆情况，道路周边建筑情况，从（{"，".join(tl3)}）中选择一个或多个标签。\
            第四步，根据道路标志牌，车辆情况，道路周边建筑情况，从（{"，".join(tl4)}）中选择一个或多个标签。\
            第五步，根据道路标志牌，车辆情况，周边障碍物情况，从（{"，".join(tl5)}）中选择一个或多个标签，没有则输出无。\
            第六步，根据道路标志牌，车辆情况，周边障碍物情况，从（{"，".join(tl6)}）中选择一个或多个标签，没有则输出无。\
            第七步，根据可行驶区域的弯曲情况判断是否是弯道。\
            输出每一步的判断结果。"""
        result = self.inference(text=prompt, image=image)

        return result
    
    def scene_tagging_en(self, image):
        prompt = f"""You are a autonomous driving labeller.\
            You have access to a camera images of a vehicle.\
            Describe the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings.\
            Answer the following 7 questions step by step.\
            Step 1, select a class from the following list ({",".join(tl1_en)}).\
            Step 2, select a class from the following list ({",".join(tl2_en)}).\
            Step 3, select a class from the following list ({",".join(tl3_en)}).\
            Step 4, select a class from the following list ({",".join(tl4_en)}).\
            Step 5, select a class from the following list ({",".join(tl5_en)}).\
            Step 6, select a class from the following list ({",".join(tl6_en)}).\
            Step 7, detect the surface of the read. Answer if the road show an obvious curvature or not. Ignore the position of the car.\
            """
        result = self.inference(text=prompt, image=image)
        
        return result

    def scene_tagging_en_json(self, image):
        prompt = f"""You are a autonomous driving labeller.\
            You have access to a camera images of a vehicle.\
            Describe the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings.\
            Answer the following 7 questions step by step. 
            Format the final answer to json with the following keys ("time_of_day","weather","road_type","location","structure","traffic_control", "road_surface")\
            Step 1, select a class from the following list ({",".join(tl1_en)}).\
            Step 2, select a class from the following list ({",".join(tl2_en)}).\
            Step 3, select a class from the following list ({",".join(tl3_en)}).\
            Step 4, select a class from the following list ({",".join(tl4_en)}).\
            Step 5, select a class from the following list ({",".join(tl5_en)}).\
            Step 6, select a class from the following list ({",".join(tl6_en)}).\
            Step 7, detect the surface of the read. Answer if the road show an obvious curvature or not. Ignore the position of the car.\
            """
        result = self.inference(text=prompt, image=image)

        return result
    
    def scene_tagging_en_json_v2(self, image):
        prompt = f"""You are a autonomous driving labeller.\
            You have access to a camera images of a vehicle.\
            Describe the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings.\
            Answer the following questions step by step. 
            Format the final answer to json with the following keys ("time_of_day","weather","road_type","location","structure","road_event", "road_curvature")\
            Step 1, answer the time of day, select a class from the following list ({",".join(tl1_en)}).\
            Step 2, answer the weather, select a class from the following list ({",".join(tl2_en)}).\
            Step 3, answer the type of the road, select a class from the following list ({",".join(tl3_en)}).\
            Step 4, answer the location, select a class from the following list ({",".join(tl4_en)}).\
            Step 5, answer the structure, select a class from the following list ({",".join(tl5_en)}).\
            Step 6, answer the road event, select a class from the following list ({",".join(tl6_en)}).\
            Step 7, detect the surface of the read. Answer if the road show an obvious curvature or not. Ignore the position of the car.\
            """
        result = self.inference(text=prompt, image=image)

        return result
    
    def scene_tagging_all_level(self, image):
        tags_l1 = ",".join(t for t in scene_tags_l1.keys())
        tags_l2 = ",".join(t for t in scene_tags_l2.keys())
        tags_l3 = ",".join(t for t in scene_tags_l3.keys())
        tags_w = ",".join(t for t in weather_tags.keys())

        prompt = f"""You are a autonomous driving labeller. \
            You have access to a camera images of a vehicle. \
            Describe the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings.\
            Take four steps to classify the image. \
            Fist step, select a class from the following list {tags_l1}.\
            Second step, select a class from the following list {tags_l2}.\
            Third step, select a class from the following list {tags_l3}.\
            Fourth step, select a class from the following list {tags_w}."""

        result = self.inference(text=prompt, image=image)

        return result
       
    def inference(self, text=None, image=None):
        message = self.get_message(text, image=image)
        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_token_gen)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    
    
def run_gradio(tagger, host=None, port=None):
    import gradio as gr

    demo = gr.Interface(
        description=f"""
            # 标签自动化
            第一级，从（{"，".join(tl1)}）中选择一个标签。\n
            第二级，从（{"，".join(tl2)}）中选择一个标签。\n
            第三级，从（{"，".join(tl3)}）中选择一个或多个标签。\n
            第四级，从（{"，".join(tl4)}）中选择一个或多个标签。\n
            第五级，从（{"，".join(tl5)}）中选择一个或多个标签，没有则输出无。\n
            第六级，从（{"，".join(tl6)}）中选择一个或多个标签，没有则输出无。\n
            第七级，根据可行驶区域的弯曲情况判断是否是弯道。。
            """,
        fn=tagger.scene_tagging_en_json_v2,
        inputs=gr.Image(type="filepath"),
        outputs=["text"],
    )
    
    demo.launch(server_name="10.78.4.131", server_port=7860)
    
if __name__ == "__main__":
    tagger = Tagger("7b")
    
    # image_path = "/home/sunyujia/python_ws/OpenEMMA/testset/1723599486425456000.png"
    # ans = tagger.scene_tagging_zh(image_path)
    # print(ans)
    
    # t0 = time.time()
    # ans = tagger.scene_tagging_by_level(image_path, "l2")
    # print("======= l2 =======\n" + ans)
    # t1 = time.time()
    # ans = tagger.scene_tagging_all_level(image_path)
    # print("======= full =======\n" + ans)
    # t2 = time.time()
    # print(f"time escaped warmup {t1 - t0} first infer {t2 - t1}")
    
    run_gradio(tagger)