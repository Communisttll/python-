import numpy as np

from scipy.ndimage import zoom

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

class Embeddings:
    def __init__(self, weights):
        """
        NumPy 实现的 Dinov2 Embeddings 层。

        参数：
        - weights: 权重字典，包含：
            - 'cls_token': 形状为 (1, 1, hidden_size)
            - 'position_embeddings': 形状为 (1, num_patches + 1, hidden_size)
        """
        self.hidden_size = 768 # D
        self.patch_size  = 14  # ps

        self.cls_token           = weights["embeddings.cls_token"] # (1, 1, D)
        self.position_embeddings = weights["embeddings.position_embeddings"] # (1, N+1, D)
        self.patch_embed_w       = weights["embeddings.patch_embeddings.projection.weight"].reshape(768, -1).T
        self.patch_embed_b       = weights["embeddings.patch_embeddings.projection.bias"].reshape(768, 1).T

    def pixel2patches(self, pixel_values): 
        B, C, H, W = pixel_values.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0

        patches = []
        for i in range(0, H, self.patch_size):
            for j in range(0, W, self.patch_size):
                patch = pixel_values[:, :, i:i+self.patch_size, j:j+self.patch_size].reshape(B, -1)
                patches.append(patch)

        patches = np.stack(patches, axis=1)  # shape: (B, num_patches, patch_dim)
        return patches

    def interpolate_pos_encoding(self, embeddings, height, width):
        # 获取输入图像的实际patch数量
        h_patches = height // self.patch_size
        w_patches = width // self.patch_size
        num_patches = h_patches * w_patches
        
        # 获取原始位置编码的维度
        # self.position_embeddings 形状: (1, N+1, D)，其中N是预训练时的patch数量
        orig_num_positions = self.position_embeddings.shape[1] - 1  # 减去cls_token
        
        # 如果当前patch数量与原始数量相同，直接返回原始位置编码
        if num_patches == orig_num_positions:
            return self.position_embeddings
        
        # 分离cls_token和patch位置编码
        cls_pos_embed = self.position_embeddings[:, :1, :]  # (1, 1, D)
        patch_pos_embed = self.position_embeddings[:, 1:, :]  # (1, N, D)
        
        # 计算原始patch网格大小（假设原始是正方形排列）
        orig_size = int(orig_num_positions ** 0.5)
        
        # 重塑为2D网格形式用于插值
        patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, self.hidden_size)
        
        # 计算目标尺寸
        target_size_h = h_patches
        target_size_w = w_patches
        
        # 使用scipy的zoom函数进行双线性插值
        # zoom参数：(z轴缩放, y轴缩放, x轴缩放, 特征维度缩放)
        # 特征维度不需要缩放，所以最后一个参数是1
        zoom_factors = (1, target_size_h / orig_size, target_size_w / orig_size, 1)
        
        # 执行插值
        interpolated_pos_embed = zoom(patch_pos_embed, zoom_factors, order=1)
        
        # 重塑回序列形式
        interpolated_pos_embed = interpolated_pos_embed.reshape(1, target_size_h * target_size_w, self.hidden_size)
        
        # 拼接cls_token位置编码
        final_pos_embed = np.concatenate([cls_pos_embed, interpolated_pos_embed], axis=1)
        
        return final_pos_embed

    def __call__(self, pixel_values):
        B, _, H, W = pixel_values.shape

        patch_values = self.pixel2patches(pixel_values) # (B, C, H, W) -> (B, h*w, C*ps**2), h=H//ps, w=W//ps
        
        # (B, h*w, C*ps**2) @ (C*ps**2, D) + (1, D) -> (B, h*w, D)
        embeddings = patch_values @ self.patch_embed_w + self.patch_embed_b
        
        cls_token  = np.tile(self.cls_token, (B, 1, 1)) # (1, 1, D) -> (B, 1, D)
        embeddings = np.concatenate([cls_token, embeddings], axis=1) # (B, h*w+1, D)

        pos_embed  = self.interpolate_pos_encoding(embeddings, H, W) # (B, N+1, D) -> (B, h*w+1, D)
        
        embeddings = embeddings + pos_embed
        return embeddings

class LayerNorm:
    def __init__(self, weight, bias, eps=1e-6):
        self.weight = weight
        self.bias   = bias
        self.eps    = eps

    def __call__(self, x, ):
        mean = x.mean(-1, keepdims=True)
        var  = x.var(-1, keepdims=True)
        norm = (x - mean) / np.sqrt(var + self.eps)
        return norm * self.weight + self.bias

class LayerScale: 
    def __init__(self, lambda1): 
        self.lambda1 = lambda1.reshape(1, 1, -1)  # 确保维度正确广播

    def __call__(self, x): 
        return x * self.lambda1

class Linear:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias   = bias

    def __call__(self, x):
        return x @ self.weight.T + self.bias

class SingleHeadAttention:
    def __init__(self, config, prefix, weights):
        self.hidden_size = config['hidden_size']
        
        q_w = weights[f"{prefix}.attention.query.weight"]
        q_b = weights[f"{prefix}.attention.query.bias"]
        k_w = weights[f"{prefix}.attention.key.weight"]
        k_b = weights[f"{prefix}.attention.key.bias"]
        v_w = weights[f"{prefix}.attention.value.weight"]
        v_b = weights[f"{prefix}.attention.value.bias"]
        o_w = weights[f"{prefix}.output.dense.weight"]
        o_b = weights[f"{prefix}.output.dense.bias"]

        self.q_proj = Linear(q_w, q_b)
        self.k_proj = Linear(k_w, k_b)
        self.v_proj = Linear(v_w, v_b)
        self.out_proj = Linear(o_w, o_b)

    def __call__(self, x):
        q = self.q_proj(x) # (B, h*w+1, D)
        k = self.k_proj(x) # (B, h*w+1, D)
        v = self.v_proj(x) # (B, h*w+1, D)
        att = np.matmul(q, k.transpose(0,2,1)) / np.sqrt(self.hidden_size) # (B, h*w+1, h*w+1)
        att = softmax(att)
        out = np.matmul(att, v) # (B, h*w+1, D)
        return self.out_proj(out)

class MultiHeadAttention:
    def __init__(self, config, prefix, weights):
        self.num_heads = config['num_heads']
        self.head_dim = config['hidden_size'] // self.num_heads

        q_w = weights[f"{prefix}.attention.query.weight"]
        q_b = weights[f"{prefix}.attention.query.bias"]
        k_w = weights[f"{prefix}.attention.key.weight"]
        k_b = weights[f"{prefix}.attention.key.bias"]
        v_w = weights[f"{prefix}.attention.value.weight"]
        v_b = weights[f"{prefix}.attention.value.bias"]
        o_w = weights[f"{prefix}.output.dense.weight"]
        o_b = weights[f"{prefix}.output.dense.bias"]

        self.q_proj   = Linear(q_w, q_b)
        self.k_proj   = Linear(k_w, k_b)
        self.v_proj   = Linear(v_w, v_b)
        self.out_proj = Linear(o_w, o_b)

    def __call__(self, x):
        B, seq_len, D = x.shape
        
        # 步骤1: 通过线性投影得到q, k, v
        q = self.q_proj(x)  # (B, seq_len, D)
        k = self.k_proj(x)  # (B, seq_len, D)
        v = self.v_proj(x)  # (B, seq_len, D)
        
        # 步骤2: 拆分多头
        # 将最后一维D拆分为num_heads × head_dim
        q = q.reshape(B, seq_len, self.num_heads, self.head_dim)  # (B, seq_len, num_heads, head_dim)
        k = k.reshape(B, seq_len, self.num_heads, self.head_dim)  # (B, seq_len, num_heads, head_dim)
        v = v.reshape(B, seq_len, self.num_heads, self.head_dim)  # (B, seq_len, num_heads, head_dim)
        
        # 调整维度顺序以便并行计算: (B, num_heads, seq_len, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # 步骤3: 计算注意力权重
        # att = q @ k.transpose(0,1,3,2) / sqrt(head_dim)
        att = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)  # (B, num_heads, seq_len, seq_len)
        
        # 通过softmax归一化
        att = softmax(att, axis=-1)
        
        # 步骤4: 计算注意力输出
        att_out = np.matmul(att, v)  # (B, num_heads, seq_len, head_dim)
        
        # 步骤5: 拼接多头结果
        # 先调整回原来的维度顺序: (B, seq_len, num_heads, head_dim)
        att_out = att_out.transpose(0, 2, 1, 3)
        
        # 然后重塑回 (B, seq_len, D)
        att_out = att_out.reshape(B, seq_len, D)
        
        # 步骤6: 线性投影输出
        output = self.out_proj(att_out)  # (B, seq_len, D)
        
        return output

class MLP:
    def __init__(self, prefix, weights):
        w1 = weights[f"{prefix}.mlp.fc1.weight"]
        b1 = weights[f"{prefix}.mlp.fc1.bias"]
        w2 = weights[f"{prefix}.mlp.fc2.weight"]
        b2 = weights[f"{prefix}.mlp.fc2.bias"]

        self.fc1 = Linear(w1, b1)
        self.fc2 = Linear(w2, b2)

    def __call__(self, x):
        return self.fc2(gelu(self.fc1(x)))

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    return x_exp / x_sum

class TransformerBlock:
    def __init__(self, config, idx, weights):
        prefix = f"encoder.layer.{idx}"
        
        self.norm1 = LayerNorm(weights[f"{prefix}.norm1.weight"], weights[f"{prefix}.norm1.bias"])
        self.scale1 = LayerScale(weights[f"{prefix}.layer_scale1.lambda1"])
        self.attn = MultiHeadAttention(config, f"{prefix}.attention", weights)

        self.norm2 = LayerNorm(weights[f"{prefix}.norm2.weight"], weights[f"{prefix}.norm2.bias"])
        self.scale2 = LayerScale(weights[f"{prefix}.layer_scale2.lambda1"])
        self.mlp = MLP(f"{prefix}", weights)

    def __call__(self, x):
        x = x + self.scale1(self.attn(self.norm1(x)))
        x = x + self.scale2(self.mlp(self.norm2(x)))
        return x

class Dinov2Numpy:
    def __init__(self, weights, config=None):
        self.weights = weights
        self.config = config or {
            "hidden_size": 768,
            "num_heads": 12,
            "num_layers": 12,
            "patch_size": 14,
        }

        self.embeddings = Embeddings(weights)
        self.blocks     = [TransformerBlock(self.config, i, weights) for i in range(self.config["num_layers"])]
        self.norm       = LayerNorm(weights["layernorm.weight"], weights["layernorm.bias"])

    def __call__(self, pixel_values):
        pos_embed = self.embeddings(pixel_values)
        for blk in self.blocks:
            pos_embed = blk(pos_embed)
        pos_embed = self.norm(pos_embed)
        return pos_embed[:, 0]
