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

    def get_tag_v(self, image, video, usr_token, usr_prompt):
        if usr_token is None:
            return "请输入用户token", "请输入用户token"
        else:
            if usr_token not in ["syj1616"]:
                return "用户token错误", "用户token错误"
            
        hist_root = os.path.join("/home/sunyujia/python_ws/qwen_tagger/infer_hist", usr_token)
        if not os.path.exists(hist_root):
            os.makedirs(hist_root)
        
        hist = [usr_token, usr_prompt]
        # user available
        if image is not None:
            print("image tagging", image)
            hist.append(image)
            pre_prompt = "You have access to a camera image of a vehicle. "
            prompt = f"You are an autonomous driving labeller. " + \
            f"{pre_prompt}" + \
            f"Describe the driving scene according to traffic lights, movements of other cars, buildings or pedestrians and lane markings. " + \
            f"Given a dictionary of road scene tags {tag_dict}. " + \
            f"Select a value from the items according to each key of the dictionary. " + \
            "Answer with a json file."
            
            if usr_prompt:
                # replace system prompt with user prompt
                prompt = usr_prompt
            result_image = self.inference(text=prompt, image=image)
            hist.append(result_image)
        else:
            result_image = "No image."
        if video is not None:
            video = [v[0] for v in video]
            video.sort()
            for p in video:
                hist.append(p)
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

            if usr_prompt:
                # replace system prompt with user prompt
                prompt = usr_prompt
            result_video = self.inference(text=prompt, image=video)
            hist.append(result_video)
        else:
            result_video = "No video."
      
        with open(os.path.join(hist_root, f"{usr_token}_{time.strftime('%Y%m%d%H%M%S')}.txt"), "w") as f:
            f.write("\n".join(hist))
            
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
            # 视觉语言大模型应用  🌈动-静态场景理解🌈
            1️⃣ 静态场景理解：输入单帧图像，大模型返回静态场景描述 \n
            2️⃣ 动态场景理解：输入连续帧图像，大模型返回动态场景描述
            """,
        fn=tagger.get_tag_v,
        inputs=[gr.Image(type="filepath", label="单帧图像"),
                gr.Gallery(label="连续帧图像"),
                gr.Textbox(label="用户token"),
                gr.Textbox("你是一个智能驾驶机器人。输入是一段道路行驶的视频。请描述视频中车辆和行人情况，交通信号灯情况，车道线情况，交通标志牌情况。", label="用户提示词", type="text")],
        outputs=[gr.Textbox(label="静态场景理解", type="text", lines=10), 
                 gr.Textbox(label="动态场景理解", type="text", lines=10)],
    )
    demo.launch(server_name="10.78.4.131", server_port=7860)
    
if __name__ == "__main__":
    tagger = Tagger("3b")
    
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