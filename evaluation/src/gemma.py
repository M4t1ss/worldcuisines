from transformers import AutoModelForCausalLM, AutoProcessor, set_seed, BitsAndBytesConfig
import torch
import pandas as pd
from PIL import Image
import re


def load_model_processor(model_path, fp32=False, multi_gpu=False):
    processor = AutoProcessor.from_pretrained(model_path)
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # quantization_config=quantization_config,
        device_map="auto" if multi_gpu else "cuda:0",
        trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if not fp32 else torch.float32),
    )

    return model, processor


def eval_instance(model, processor, image_file, query, seed, icl, icl_img):
    # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
    
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
                    images.append(icl_file)

                translation = re.sub(r'^https?:\/\/.*[\r\n]*', '', row["English"], flags=re.MULTILINE)
                given = 'Given the following text, extracted from a tweet in English: \n'+translation+'\n'
                answer = "Reply “Yes” or “No”."

                query1 = given + ' Is the image adding to the text meaning? ' + answer
                query2 = given + ' Is the text represented in the image? ' + answer

                reply1 = "Yes" if row["relation"] in [0,1] else "No"
                reply2 = "Yes" if row["relation"] in [0,2] else "No"

                if "Is the image adding to the text meaning" in query:

                    if icl_img:
                        convo = [
                            {"role": "user", "content": f"<|image_{count}|>\n" + query1},
                            {"role": "assistant", "content": reply1},
                        ]
                        count+=1
                    else:
                        convo = [
                            {"role": "user", "content": query1},
                            {"role": "assistant", "content": reply1},
                        ]

                else:
                    if icl_img:
                        convo = [
                            {"role": "user", "content": f"<|image_{count}|>\n" + query2},
                            {"role": "assistant", "content": reply2},
                        ]
                        count+=1
                    else:
                        convo = [
                            {"role": "user", "content": query2},
                            {"role": "assistant", "content": reply2},
                        ]

                chat+=convo
                ic+=1
                if ic > icl:
                    break

        images.append(image_file)

        msg = [
            {"role": "user", "content": f"<|image_{count}|>\n" + query},
        ]
        messages = chat+msg
    else:
        images = [image_file]

        messages = [
            {"role": "user", "content": f"<|image_{1}|>\n" + query},
        ]


    inputs = processor.tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    set_seed(seed)


    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=True)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)




    # inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")

    # generation_args = {
    #     "max_new_tokens": 512,
    #     "temperature": 0.2,
    #     "do_sample": True,
    #     "num_logits_to_keep": 1,
    # }

    # generate_ids = model.generate(
    #     **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
    # )

    # # remove input tokens
    # generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    # response = processor.batch_decode(
    #     generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )[0]

    return decoded
