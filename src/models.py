import openai

HF_MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl"
]

GPT3_MODELS = [
    "text-davinci-003", 
    "text-davinci-002", 
    "text-curie-001", 
    "text-babbage-001", 
    "text-ada-001"
]

def get_label_probs(prompt,label,model="text-davinci-003"):
    if model in HF_MODELS:
        return get_label_probs_hf(prompt,label,model)
    if model in GPT3_MODELS:
        return get_label_probs_gpt3(prompt,label,model)
    raise ValueError(f"Model {model} not supported.")


def get_label_probs_hf(prompt,label,model="gpt2"):
    """
    Return the probability of the label given the prompt for the given model.

    Parameters
    ----------
    prompt : str
        The prompt to be completed.
    label : str
        The label to be predicted.
    model : str, optional
        The model to be used. The default is "gpt2".

    Returns
    -------
    float
        The probability P(label|prompt) of the label given the prompt.

    """
    
    ## TODO: Sumar las logprobs de cada token que conforma el label.
    # Es decir, si el label es "politics", calcular logP(" politics"|prompt) = logP(" po"|prompt) + logP("litics"|prompt," po")
    logprob = 0.
    return logprob


def get_label_probs_gpt3(prompt,label,model="text-davinci-003"):
    """
    Return the probability of the label given the prompt for the given model.

    Parameters
    ----------
    prompt : str
        The prompt to be completed.
    label : str
        The label to be predicted.
    model : str, optional
        The model to be used. The default is "text-davinci-003".

    Returns
    -------
    float
        The probability P(label|prompt) of the label given the prompt.

    Comments
    --------
    con estos parámetros, completition["choices"][0] devuelve un diccionario:
    {
        "finish_reason": "length",
        "index": 0,
        "logprobs": {
            "text_offset": [
                ... # lista de offsets de cada token de entrada
            ],
            "token_logprobs": [
                ... # lista de [None, P(w2|w1), ..., P(wn|w1,...,wn-1)]
            ],
            "tokens": [
                ... # lista de [w1, w2, ..., wn]
            ],
            "top_logprobs": None # Lista de los logprobs tokens más probables para [(w1), (w1,w2), ..., (w1,...,wn-1)]
        },
        "text": ...,
    }
    """
    prompt = prompt + " " + label
    completiton = openai.Completion.create(
        model=model,
        prompt=prompt,
        logprobs=0,
        temperature=0,
        max_tokens=0,
        top_p=1,
        echo=True
    )
    logprob = completiton["choices"][0]["logprobs"]["token_logprobs"][-1]
    return logprob