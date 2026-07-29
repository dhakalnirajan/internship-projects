def softmax(x, axis=-1):
    max_x = x - x.max(axis=axis, keepdims=True)
    e = max_x.exp()
    return e / e.sum(axis=axis, keepdims=True)
