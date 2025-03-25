import os
import time
import torch
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from collections import OrderedDict


tag_dict = OrderedDict({
    "time_of_day": ['daytime', 'night'],
    "weather_condition": ['sunny', 'rainy', 'foggy', 'snowy', 'overcast'],
    "road_type": ['urban road', 'country road', 'highway'],
    "location": ['main road', 'side road','parking area', 
          'gas station', 'toll station', 'service area', 'harbour', 'internal road'],
    "road_feature": ['no', 'intersection', 'roundabout', 'tunnel', 'ramp', 'slope', 'pedestrian overpass', 'bridge'],
    "road_status": ['no', 'accident section', 'road under construction'],
    "road_curvature": ['no', 'the road show an obvious curvature'],
})

tag_dyn_dcit = OrderedDict({
    "time_of_day": ['daytime', 'night'],
    "weather_condition": ['sunny', 'rainy', 'foggy', 'snowy', 'overcast'],
    "road_type": ['urban road', 'country road', 'highway'],
    "location": ['main road', 'side road','parking area', 
          'gas station', 'toll station', 'service area', 'harbour', 'internal road'],
    "road_feature": ['no', 'intersection', 'roundabout', 'tunnel', 'ramp', 'slope', 'pedestrian overpass', 'bridge'],
    "road_status": ['no', 'accident section', 'road under construction'],
    "road_curvature": ['no', 'the road show an obvious curvature'],
})

class Tagger:
    def __init__(self, vlm, max_token_gen=1024) -> None:
        if vlm == "7b_q4":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit", 
                torch_dtype="auto", device_map="auto"
            )
        
            self.processor = AutoProcessor.from_pretrained(
                "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit")
    
        elif vlm == "7b":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct")

        elif vlm == "3b":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )

            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct")
            
        self.device = self.model.device
        self.max_token_gen = max_token_gen
    
    def get_message(self, prompt, image=None):
        if isinstance(image, (list, tuple)):
            message = [
                {"role": "user", 
                "content": [
                    {"type": "video", 
                    "video": image,
                    "resized_height": 320,
                    "resized_width": 800},
                    {"type": "text", 
                    "text": prompt}
                ]}
            ]              
        else:
            message = [
                {"role": "user", 
                "content": [
                    {"type": "image", 
                    "image": image,
                    "resized_height": 320,
                    "resized_width": 800
                    },
                    {"type": "text", 
                    "text": prompt}
                ]}
            ]   
        
        return message

    def get_tag(self, image, if_reasoning):
        sence_l = list(tag_dict.keys())
        if isinstance(image, (list, tuple)):
            pre_prompt = "You have access to a video of a vehicle. "
        else:
            pre_prompt = "You have access to a camera image of a vehicle. "
            
        if if_reasoning:
            prompt = f"""You are an autonomous driving labeller. """ + \
                f"""{pre_prompt}""" + \
                f"""Describe the driving scene according to traffic lights, movements of other cars, buildings or pedestrians and lane markings. """ + \
                f"""Answer the following questions step by step, strictly. And give a final description in the end. """ + \
                f"""Step 1, select a value of time_of_day from the following list ({",".join(tag_dict[sence_l[0]])}). Explain why. And show the class name in this format #Class: time_of_day). """ + \
                f"""Step 2, select a class from the following list ({",".join(tag_dict[sence_l[1]])}). """ + \
                f"""Step 3, select a class from the following list ({",".join(tag_dict[sence_l[2]])}). """ + \
                f"""Step 4, select a class from the following list ({",".join(tag_dict[sence_l[3]])}). """ + \
                f"""Step 5, select a class from the following list ({",".join(tag_dict[sence_l[4]])}). """ + \
                f"""Step 6, select a class from the following list ({",".join(tag_dict[sence_l[5]])}). """ + \
                f"""Step 7, detect the surface of the road. Answer if the road show an obvious curvature or not. Ignore the position of the car. Select a class from the following list ({",".join(tag_dict[sence_l[6]])}). """
        else:
            prompt = f"You are an autonomous driving labeller. " + \
            f"{pre_prompt}" + \
            f"Describe the driving scene according to traffic lights, movements of other cars, buildings or pedestrians and lane markings. " + \
            f"Given a dictionary of road scene tags {tag_dict}. " + \
            f"Select a value from the items according to each key of the dictionary. " + \
            "Answer with a json file."
        result = self.inference(text=prompt, image=image)

        return result
       
    def get_tag_v(self, image, video, if_reasoning):
        if image is not None:
            print("image tagging", image)
            pre_prompt = "You have access to a camera image of a vehicle. "
            prompt = f"You are an autonomous driving labeller. " + \
            f"{pre_prompt}" + \
            f"Describe the driving scene according to traffic lights, movements of other cars, buildings or pedestrians and lane markings. " + \
            f"Given a dictionary of road scene tags {tag_dict}. " + \
            f"Select a value from the items according to each key of the dictionary. " + \
            "Answer with a json file."
            result_image = self.inference(text=prompt, image=image)
        else:
            result_image = "No image."
        if video is not None:
            video = [v[0] for v in video]
            video.sort()
            print("video tagging", video)
            pre_prompt = "You have access to a video of a vehicle. "
            prompt = f"You are an autonomous driving robot. " + \
                f"{pre_prompt}" + \
                f"Describe the driving scene according to traffic lights, movements of other cars, buildings or pedestrians and lane markings. " + \
                f"Answer the following questions step by step. " + \
                f"step 1, any intersection scene in the video. Did the car stop? " + \
                f"step 2, any car drive into our road suddently? " + \
                f"step 3, did the car drive into a tunnel or leave a tunnel? " + \
                f"step 4, did the car show a turn in the road? " + \
                f"step 5, any contruction in the road? "            

            result_video = self.inference(text=prompt, image=video)
        else:
            result_video = "No video."

        return result_image, result_video
    
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
    sence_l = list(tag_dict.keys())
    demo = gr.Interface(
        description=f"""
            # 标签自动化
            第一级，从（{",".join(tag_dict[sence_l[0]])}）中选择一个标签。\n
            第二级，从（{",".join(tag_dict[sence_l[1]])}）中选择一个标签。\n
            第三级，从（{",".join(tag_dict[sence_l[2]])}）中选择一个或多个标签。\n
            第四级，从（{",".join(tag_dict[sence_l[3]])}）中选择一个或多个标签。\n
            第五级，从（{",".join(tag_dict[sence_l[4]])}）中选择一个或多个标签，没有则输出无。\n
            第六级，从（{",".join(tag_dict[sence_l[5]])}）中选择一个或多个标签，没有则输出无。\n
            第七级，根据可行驶区域的弯曲情况判断是否是弯道。
            """,
        fn=tagger.get_tag_v,
        inputs=[gr.Image(type="filepath"),
                gr.Gallery(),
                gr.Checkbox(value=True, label='推理模式')],
        outputs=["text", 
                 "text"],
    )
    demo.launch(server_name="10.78.4.131", server_port=7860)
    
if __name__ == "__main__":
    tagger = Tagger("7b")
    
    run_gradio(tagger)
    
    # data_root = "/home/sunyujia/python_ws/adas_test_data/under_construction"
    # video_full = [os.path.join(data_root, p) for p in os.listdir(data_root)]
    # video_full.sort()
    # for i in range(len(video_full) // 10):
    #     print(f"====================== {i} ======================")
    #     videos = video_full[i*10:(i+1)*10]
    #     print(f"=== {videos[0]} ===")
    #     ans = tagger.get_tag_v(videos, False)
    #     print(ans)