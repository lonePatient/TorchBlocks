def build_ngrams(input, minn=3, maxn=3, start='<', end='>'):
    input = start + input + end
    len_ = len(input)
    ngrams = []
    for ngram in reversed(range(minn, maxn + 1)):
        for i in range(0, len_ - ngram + 1):
            ngrams.append(input[i:i + ngram])
    return ngrams

if __name__ == "__main__":
    input = '人工智能大赛'
    print(build_ngrams(input))