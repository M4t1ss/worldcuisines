from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig, set_seed
import torch
import pandas as pd
from PIL import Image
import re


def load_model_processor(model_path, fp32=False, multi_gpu=False):
    processor = LlavaNextProcessor.from_pretrained(model_path)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        torch_dtype=(torch.float16 if not fp32 else torch.float32),
        low_cpu_mem_usage=True,
        device_map="auto" if multi_gpu else "cuda:0",
    )
    processor.patch_size = 14
    processor.vision_feature_select_strategy = "default"
    processor.tokenizer.padding_side = "left"
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return model, processor


def eval_instance(model, processor, image_file, query, seed, icl, icl_img):

    set_seed(seed)
    if icl > 0:

        icl_data = pd.read_csv("icl/icl-subset-twitediens.latvian.tsv", sep='\t', header=0)

        chat = []
        images = []

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
                    images.append(icl_file)

                translation = re.sub(r'^https?:\/\/.*[\r\n]*', '', row["English"], flags=re.MULTILINE)
                given = 'Given the following text, extracted from a tweet in English: \n'+translation.strip()+'\n'
                answer = "Reply “Yes” or “No”.\n"

                query1 = given + 'Is the image adding to the text meaning? ' + answer
                query2 = given + 'Is the text represented in the image? ' + answer

                reply1 = "Yes\n" if row["relation"] in [0,1] else "No\n"
                reply2 = "Yes\n" if row["relation"] in [0,2] else "No\n"

                if "Is the image adding to the text meaning" in query:

                    if icl_img:
                        convo = [
                            {
                                "role": "user", 
                                "content": [
                                    {"type": "text", "text": query1}, 
                                    {"type": "image"}
                                ]
                            },
                            {
                                "role": "assistant", 
                                "content": [{"type": "text", "text": reply1}],
                            },
                        ]
                    else:
                        convo = [
                            {
                                "role": "user", 
                                "content": [
                                    {"type": "text", "text": query1}, 
                                ]
                            },
                            {   
                                "role": "assistant", 
                                "content": [{"type": "text", "text": reply1}],
                            },
                        ]

                else:
                    if icl_img:
                        convo = [
                            {
                                "role": "user", 
                                "content": [
                                    {"type": "text", "text": query2}, 
                                    {"type": "image"}
                                ]
                            },
                            {
                                "role": "assistant", 
                                "content": [{"type": "text", "text": reply2}],
                            },
                        ]
                    else:
                        convo = [
                            {
                                "role": "user", 
                                "content": [
                                    {"type": "text", "text": query2}, 
                                ]
                            },
                            {   
                                "role": "assistant", 
                                "content": [{"type": "text", "text": reply2}],
                            },
                        ]

                chat+=convo
                ic+=1
                if ic > icl:
                    break

        if icl_img:
            images.append(image_file)
        else:
            images=[image_file]

        msg = [
            {
                "role": "user",
                "content": [{"type": "text", "text": query}, {"type": "image"}],
            },
        ]
        chat+=msg

        prompt = processor.apply_chat_template(chat, add_generation_prompt=True)
        inputs = processor(images=images, text=prompt, return_tensors="pt").to("cuda:0")


        output = model.generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=True,
            temperature=0.2,
            min_new_tokens=1,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)
        ]
        out = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    else:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image_file, text=prompt, return_tensors="pt").to("cuda:0")

        output = model.generate(
            **inputs,
            max_new_tokens=256, 
            do_sample=True,
            temperature=0.2,
            min_new_tokens=1,
        )

        out_with_template = processor.decode(output[0], skip_special_tokens=True)
        if "ASSISTANT: " in out_with_template:
            out = out_with_template[
                out_with_template.index("ASSISTANT: ") + len("ASSISTANT: ") :
            ]
        elif "assistant\n" in out_with_template:
            out = out_with_template[
                out_with_template.index("assistant\n") + len("assistant\n") :
            ]
        else:
            out = out_with_template

    return out
