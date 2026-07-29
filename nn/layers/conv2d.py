import numpy as np
from ..autograd import Function, Tensor
from .base import Layer


class _Conv2DFunction(Function):
    @staticmethod
    def forward(ctx, x, kernel, bias, strides, padding):
        # x, kernel, bias are numpy arrays (data of Tensors)
        kh, kw, C, F = kernel.shape
        sy, sx = strides if isinstance(strides, tuple) else (strides, strides)

        # Handle different padding configurations ('valid', 'same', or explicit integer/tuple)
        if padding == 'valid':
            ph = pw = 0
        elif padding == 'same':
            out_h = int(np.ceil(x.shape[1] / sy))
            out_w = int(np.ceil(x.shape[2] / sx))
            total_ph = max(0, (out_h - 1) * sy + kh - x.shape[1])
            total_pw = max(0, (out_w - 1) * sx + kw - x.shape[2])
            ph = total_ph // 2
            pw = total_pw // 2
        else:
            if isinstance(padding, int):
                ph = pw = padding
            else:
                ph, pw = padding

        # Store context variables for the backward pass
        ctx.save_for_backward(x, kernel, bias)
        ctx.strides = (sy, sx)
        ctx.padding = (ph, pw)
        ctx.input_shape = x.shape
        ctx.kernel_shape = kernel.shape

        N, H, W, C = x.shape
        # Pad the input image matrices along spatial dimensions (height and width)
        x_pad = np.pad(x, ((0,0), (ph,ph), (pw,pw), (0,0)), mode='constant')
        out_h = (H + 2*ph - kh) // sy + 1
        out_w = (W + 2*pw - kw) // sx + 1

        # im2col: Extract sliding local regions (patches) for efficient matrix multiplication
        patches = np.zeros((N, kh*kw*C, out_h*out_w), dtype=x.dtype)
        for n in range(N):
            idx = 0
            for i in range(0, out_h):
                for j in range(0, out_w):
                    patch = x_pad[n, i*sy:i*sy+kh, j*sx:j*sx+kw, :]
                    patches[n, :, idx] = patch.ravel()
                    idx += 1
        ctx.patches = patches

        # Convolve: Apply cross-correlation via matrix multiplication across multiple input channels & filters
        # y_i = B_i + sum_{j=1}^{n} x_j * K_{ij}
        kernel_flat = kernel.reshape(F, -1)          # (F, kh*kw*C)
        out_flat = np.einsum('fk,nkw->nfw', kernel_flat, patches)  # (N, F, L)
        out_flat += bias.reshape(1, F, 1)            # Add bias terms per filter
        out = out_flat.reshape(N, F, out_h, out_w).transpose(0, 2, 3, 1)  # (N, out_h, out_w, F)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is a numpy array containing gradients from subsequent layers
        x, kernel, bias = ctx.saved_tensors
        sy, sx = ctx.strides
        ph, pw = ctx.padding
        kh, kw, C, F = ctx.kernel_shape
        N, H, W, C = ctx.input_shape
        patches = ctx.patches
        out_h, out_w = grad_output.shape[1:3]

        # Gradient with respect to bias: sum over batch and spatial dimensions
        dbias = grad_output.sum(axis=(0, 1, 2))

        # Gradient with respect to kernel weights (dkernel)
        grad_flat = grad_output.transpose(0, 3, 1, 2).reshape(N, F, -1)  # (N, F, L)
        dkernel = np.zeros((F, kh * kw * C), dtype=grad_output.dtype)
        for n in range(N):
            dkernel += grad_flat[n] @ patches[n].T
        dkernel = dkernel.reshape(kh, kw, C, F)

        # Gradient with respect to input (col2im): backpropagate gradients into patches
        kernel_flat = kernel.reshape(F, -1)  # (F, K)
        grad_patches = np.zeros((N, kh * kw * C, out_h * out_w), dtype=grad_output.dtype)
        for n in range(N):
            grad_patches[n] = kernel_flat.T @ grad_flat[n]  # (K, L)

        # Scatter patch gradients back into the full input gradient tensor (accounting for stride and padding)
        grad_input = np.zeros((N, H + 2*ph, W + 2*pw, C), dtype=grad_output.dtype)
        for n in range(N):
            idx = 0
            for i in range(0, out_h):
                for j in range(0, out_w):
                    patch = grad_patches[n, :, idx].reshape(kh, kw, C)
                    grad_input[n, i*sy:i*sy+kh, j*sx:j*sx+kw, :] += patch
                    idx += 1
        if ph > 0 or pw > 0:
            grad_input = grad_input[:, ph:-ph if ph>0 else None, pw:-pw if pw>0 else None, :]

        return grad_input, dkernel, dbias, None, None


class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1,1), padding='valid'):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding
        self.kernel = None
        self.bias = None
        self.built = False
        self._ctx = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        kh, kw = self.kernel_size
        k_shape = (kh, kw, in_channels, self.filters)
        # Xavier / Glorot uniform weight initialization
        limit = np.sqrt(6 / (in_channels + self.filters))
        self.kernel = Tensor(np.random.uniform(-limit, limit, k_shape), requires_grad=True)
        self.bias = Tensor(np.zeros(self.filters), requires_grad=True)
        self.built = True

    def forward(self, inputs):
        if not self.built:
            self.build(inputs.shape)
        out = _Conv2DFunction.apply(inputs, self.kernel, self.bias, self.strides, self.padding)
        if out._ctx is not None:
            self._ctx = out._ctx
        return out

    def backward(self, grad_output):
        if self._ctx is None:
            raise RuntimeError("Forward pass must be called before backward.")
        grad_input, dkernel, dbias, _, _ = _Conv2DFunction.backward(
            self._ctx, 
            grad_output.data if hasattr(grad_output, 'data') else grad_output
        )
        if self.kernel.grad is None:
            self.kernel.grad = dkernel
        else:
            self.kernel.grad += dkernel
        if self.bias.grad is None:
            self.bias.grad = dbias
        else:
            self.bias.grad += dbias
        return Tensor(grad_input, requires_grad=False)

    def parameters(self):
        return [('kernel', self.kernel), ('bias', self.bias)]