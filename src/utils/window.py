from transformers import AutoTokenizer

def build_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    return tok

def make_windows(text, tokenizer, window_size=1024, overlap=256):
    """
    Metni token ID'lerine çevirir ve sliding-window ile parçalar.
    
    Returns:
        List[List[int]]: Her pencere bir token ID listesi (special token'sız)
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    step = max(1, window_size - overlap)
    out = []
    for i in range(0, len(ids), step):
        chunk = ids[i:i+window_size]
        if not chunk: break
        out.append(chunk)
        if i + window_size >= len(ids): break
    return out



