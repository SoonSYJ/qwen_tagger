import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.fawai_tagger.tagger import Tagger


if __name__ == "__main__":
    tagger = Tagger(
        "/home/sunyujia/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        4)

    images = ['/home/sunyujia/python_ws/adas_test_data/testset_static/1719286164.147_rear.jpg',
        '/home/sunyujia/python_ws/adas_test_data/testset_static/1725180546518535000.png']
    res_images, res_videos = tagger.get_tag(images, None)

    for res_image in res_images:
        print(res_image)