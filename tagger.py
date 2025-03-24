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
        if isinstance(image, (list, tuple)):
            message = [
                {"role": "user", 
                "content": [
                    {"type": "video", 
                    "video": image},
                    {"type": "text", 
                    "text": prompt}
                ]}
            ]              
        else:
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

    def get_tag(self, image, if_reasoning):
        print(image)
        sence_l = list(tag_dict.keys())
        if isinstance(image, (list, tuple)):
            pre_prompt = "You have access to a video of a vehicle."
        else:
            pre_prompt = "You have access to a camera image of a vehicle."
            
        if if_reasoning:
            prompt = f"""You are an autonomous driving labeller.\
                {pre_prompt}\
                Describe the driving scene according to traffic lights, movements of other cars, buildings or pedestrians and lane markings.\
                Answer the following questions step by step, strictly. And give a final description in the end.\
                Step 1, select a value of time_of_day from the following list ({",".join(tag_dict[sence_l[0]])}). Explain why. And show the class name in this format #Class: time_of_day)\
                Step 2, select a class from the following list ({",".join(tag_dict[sence_l[1]])}).\
                Step 3, select a class from the following list ({",".join(tag_dict[sence_l[2]])}).\
                Step 4, select a class from the following list ({",".join(tag_dict[sence_l[3]])}).\
                Step 5, select a class from the following list ({",".join(tag_dict[sence_l[4]])}).\
                Step 6, select a class from the following list ({",".join(tag_dict[sence_l[5]])}).\
                Step 7, detect the surface of the read. Answer if the road show an obvious curvature or not. Ignore the position of the car. Select a class from the following list ({",".join(tag_dict[sence_l[6]])}).\
                """
        else:
            prompt = f"""You are an autonomous driving labeller.\
                {pre_prompt}\
                Describe the driving scene according to traffic lights, movements of other cars, buildings or pedestrians and lane markings.\
                Given a dictionary of road scene tags {tag_dict}. Select a value from the items according to each key of the dictionary.\
                Answer with a json file.\
                    """
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
        fn=tagger.get_tag,
        inputs=[gr.Image(type="filepath"), gr.Checkbox(value=True, label='推理模式')],
        outputs=["text"],
    )
    
    demo.launch(server_name="10.78.4.131", server_port=7860)
    
if __name__ == "__main__":
    tagger = Tagger("7b")
    
    run_gradio(tagger)
    
    # data_root = "/home/sunyujia/python_ws/OpenEMMA/rear"
    # video = [os.path.join(data_root, p) for p in os.listdir(data_root)]
    # for i in range(len(video) // 10):
    #     ans = tagger.get_tag(video[i*10:(i+1)*10], False)
    #     print(ans)