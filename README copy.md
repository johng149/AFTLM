
Tokenizer special tokens:
special_tokens = [
    # mask token, used for masked language modeling
    "<mask>",
    # start of prompt, special instruction for the model
    "<sop>",
    # end of prompt, special instruction for the model
    "<eop>",
    # start of memory, indicates section of data retrieved from external memory
    "<som>",
    # end of memory, indicates section of data retrieved from external memory
    "<eom>",
    # start of reference text, indicates text that model can imitate if need be
    "<sor>",
    # end of reference text
    "<eor>",
    # new line character, sentencepiece assumes all training data is one liner,>
    "<nel>"
]
