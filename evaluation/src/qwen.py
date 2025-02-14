from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, set_seed
from qwen_vl_utils import process_vision_info
import torch
import pandas as pd
from PIL import Image
import re


def load_model_processor(model_path, fp32=False, multi_gpu=False):
    MIN_PX = 200 * 200
    MAX_PX = 1600 * 1200

    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=MIN_PX, max_pixels=MAX_PX
    )
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if not fp32 else torch.float32,
        attn_implementation="flash_attention_2",
        device_map="auto" if multi_gpu else "cuda:0",
    )
    return model, processor


def eval_instance(model, processor, image_file, query, seed, icl, icl_img):
    set_seed(seed)
    if icl > 0:

        icl_data = pd.read_csv("icl/icl-subset-twitediens.latvian.tsv", sep='\t', header=0)

        chat = []
        images = []
        count = 1

        # Show one example of Yes and one of No
        if "Is the image adding to the text meaning" in query:
            icl_cats = [1,2]
        else:
            icl_cats = [0,3]

        for rel in icl_cats:
            selected_rows = icl_data[icl_data['relation'] == rel]
            ic = 1
            for index,row in selected_rows.iterrows():

                if icl_img:
                    icl_path = "icl/"+str(row["tweet_id"])+".jpg"
                    # Load the image using PIL
                    icl_file = Image.open(icl_path)

                translation = re.sub(r'^https?:\/\/.*[\r\n]*', '', row["English"], flags=re.MULTILINE)
                given = 'Given the following text, extracted from a tweet in English: \n'+translation+'\n'
                answer = "Reply “Yes” or “No”."

                query1 = given + ' Is the image adding to the text meaning? ' + answer
                query2 = given + ' Is the text represented in the image? ' + answer

                reply1 = "Yes" if row["relation"] in [0,1] else "No"
                reply2 = "Yes" if row["relation"] in [0,2] else "No"

                if "Is the image adding to the text meaning" in query:
                    if icl_img:
                        convo1 = [
                            {   
                                "role": "user", 
                                "content": [
                                    {"type": "image", "image": icl_file},
                                    {"type": "text", "text": query1},
                                ]
                            }
                        ]
                        convo2 = [
                            {
                                "role": "assistant", 
                                "content": [
                                    {"type": "text", "text": reply1 },
                                ]
                            },
                        ]
                    else:
                        convo1 = [
                            {   
                                "role": "user", 
                                "content": [
                                    {"type": "text", "text": query1},
                                ]
                            }
                        ]
                        convo2 = [
                            {
                                "role": "assistant", 
                                "content": [
                                    {"type": "text", "text": reply1 },
                                ]
                            },
                        ]
                else:
                    if icl_img:
                        convo1 = [
                            {   
                                "role": "user", 
                                "content": [
                                    {"type": "image", "image": icl_file},
                                    {"type": "text", "text": query2},
                                ]
                            }
                        ]
                        convo2 = [
                            {
                                "role": "assistant", 
                                "content": [
                                    {"type": "text", "text": reply2 },
                                ]
                            },
                        ]
                    else:
                        convo1 = [
                            {   
                                "role": "user", 
                                "content": [
                                    {"type": "text", "text": query2},
                                ]
                            }
                        ]
                        convo2 = [
                            {
                                "role": "assistant", 
                                "content": [
                                    {"type": "text", "text": reply2 },
                                ]
                            },
                        ]
                        
                chat+=convo1
                chat+=convo2

                ic+=1
                if ic > icl:
                    break

        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_file},
                    {"type": "text", "text": query},
                ],
            }
        ]
        chat+=msg

        text = processor.apply_chat_template(chat, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(chat)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to("cuda")

        output = model.generate(**inputs, 
            max_new_tokens=512, 
            do_sample=True
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)
        ]
        out = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_file},
                    {"type": "text", "text": query},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to("cuda")

        output = model.generate(**inputs, 
            max_new_tokens=512, 
            do_sample=True
        )

        out_with_template = processor.batch_decode(output, skip_special_tokens=True)[0]
        out = out_with_template[
            out_with_template.index("\nassistant\n") + len("\nassistant\n") :
        ]

    return out
