from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="./tokenizer_custom/tokenizer.json"
)

tokenizer.add_special_tokens({
    "pad_token": "<pad>",
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "unk_token": "<unk>",
    "sep_token": "<sep>",
})

tokenizer.save_pretrained("./model_output")