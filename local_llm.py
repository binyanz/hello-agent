import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen1.5-0.5B-Chat"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    print("模型和分词器加载完成！")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "英伟达最新的GPU型号是什么"}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    print("编码后的输入文本:")
    print(model_inputs)

    # max_new_tokens 控制了模型最多能生成多少个新的Token
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )

    # 截取掉输入部分，只解码模型新生成的部分
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("\n模型的回答:")
    print(response)


if __name__ == "__main__":
    main()
