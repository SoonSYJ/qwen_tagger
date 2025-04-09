from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from .ad_tags import tag_dict


class Tagger:
    def __init__(self, model_path, n_node, max_token_gen=256, height=320, width=800) -> None:
        self.model = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 10, "video": 10},
            tensor_parallel_size=n_node,
            # gpu_memory_utilization=1,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
            
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=max_token_gen,
            stop_token_ids=[],
        )
        
        self.resized_height = height
        self.resized_width = width
                
    def get_message(self, prompt, model_inp, inp_type="image"):
        message = [
            {"role": "user", 
            "content": [
                {"type": inp_type, 
                inp_type: model_inp,
                "resized_height": self.resized_height,
                "resized_width": self.resized_width,
                "min_pixels": 224 * 224,
                "max_pixels": 1280 * 28 * 28,
                },
                {"type": "text", 
                "text": prompt}
            ]}
        ]   
        
        return message

    def get_tag(self, images, videos):
        """_summary_

        Args:
            images (list): input list of image [img1, img2, ...]
            videos (list): input list of video [[frame1, frame2, ...], [frame1, frame2, ...], ...]

        Returns:
            list: text results for images input
            list: text results for videos input
        """
        result_image = ""
        if images is not None:
            pre_prompt = "You have access to a camera image of a vehicle. "
            prompt = f"You are an autonomous driving labeller. " + \
            f"{pre_prompt}" + \
            f"Describe the driving scene according to traffic lights, movements of other cars, buildings or pedestrians and lane markings. " + \
            f"Given a dictionary of road scene tags {tag_dict}. " + \
            f"Select a value from the items according to each key of the dictionary. " + \
            "Answer with a json file."

            result_image = self.inference(text=prompt, model_inps=images, inp_type='image')

        result_video = ""
        if videos is not None:
            videos = [video.sort() for video in videos]
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

            result_video = self.inference(text=prompt, model_inps=videos, inp_type='video')
            
        return result_image, result_video
    
    def inference(self, text, model_inps, inp_type='image'):
        llm_inputs = []
        for model_inp in model_inps:
            message = self.get_message(text, model_inp=model_inp, inp_type=inp_type)
            prompt = self.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(message)
            
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            llm_input = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
            }
            llm_inputs.append(llm_input)
            
        outputs = self.model.generate(llm_inputs, sampling_params=self.sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]
            
        return generated_texts