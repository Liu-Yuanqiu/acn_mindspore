import mindspore
from mindspore import nn, ops, Tensor, Parameter, ms_function
import mindspore.numpy as np

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 5.0

clip_grad = ops.MultitypeFuncGraph("clip_grad")
@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = ops.dtype(grad)
    if clip_type == 0:
        new_grad = ops.clip_by_value(grad, ops.cast(ops.tuple_to_array((-clip_value,)), dt),
                                     ops.cast(ops.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, ops.cast(ops.tuple_to_array((clip_value,)), dt))
    return new_grad

def squash(inputs, axis=-1):
    norm = ops.LpNorm(axis=axis, p=2, keep_dims=True)(inputs)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    return scale * inputs

class Capsule(nn.Cell):
    def __init__(self, routings=3):
        super(Capsule, self).__init__()
        self.batch_size = 4
        self.in_num_caps = 36
        self.out_num_caps = 1
        self.in_dim_caps = 1024
        self.out_dim_caps = 1024
        self.routings = routings
        self.standardnormal = ops.StandardNormal(seed=0)
        self.weight = self.standardnormal((self.out_num_caps, self.in_num_caps, self.in_dim_caps, self.out_dim_caps))
        self.b = ops.Zeros()((self.batch_size, self.out_num_caps, self.in_num_caps), mindspore.float32)
        self.softmax_1 = ops.Softmax(1)
        self.reducesum = ops.ReduceSum(keep_dims=True)
        self.reducesum_1 = ops.ReduceSum()
    def construct(self, x):
        x_hat = ops.matmul(self.weight, x[:, None, :, :, None]).squeeze(-1)
        x_hat_detached = x_hat.copy()
        bb = self.b
        outputs = None
        for i in range(self.routings):
            c = self.softmax_1(bb)
            if i == self.routings - 1:
                outputs = squash(self.reducesum(c[:, :, :, None] * x_hat, -2))
            else: 
                outputs = squash(self.reducesum(c[:, :, :, None] * x_hat_detached, -2))
                bb = bb + self.reducesum_1(outputs * x_hat_detached, -1)
        attn = outputs.squeeze(-2)
        return attn
    
class SCAtt(nn.Cell):
    def __init__(self, mid_dims, mid_dropout):
        super(SCAtt, self).__init__()
        sequential = []
        for i in range(1, len(mid_dims)-1):
            sequential.append(nn.Dense(mid_dims[i-1], mid_dims[i]))
            sequential.append(nn.ReLU())
            if mid_dropout > 0:
                sequential.append(nn.Dropout(mid_dropout))
        self.attention_basic = nn.SequentialCell(*sequential) if len(sequential) > 0 else None
        self.attention_last = nn.Dense(mid_dims[-2], mid_dims[-1])

        self.attention_last = nn.Dense(mid_dims[-2], 1)
        self.attention_last2 = nn.Dense(mid_dims[-2], mid_dims[-1])

        self.expand_dims = ops.ExpandDims()
    def construct(self, att_map, att_mask, value1, value2):
        att_map = self.attention_basic(att_map)

        if att_mask is not None:
            att_mask = self.expand_dims(att_mask, 1)
            att_mask_ext = self.expand_dims(att_mask, -1)
            att_map_pool = ops.ReduceSum()(att_map * att_mask_ext, -2) / ops.ReduceSum()(att_mask_ext, -2)
        else:
            att_map_pool = att_map.mean(-2)

        alpha_spatial = self.attention_last(att_map)
        alpha_channel = self.attention_last2(att_map_pool)
        alpha_channel = ops.Sigmoid()(alpha_channel)

        alpha_spatial = alpha_spatial.squeeze(-1)
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask == 0, -1e9)
        alpha_spatial = ops.Softmax()(alpha_spatial)

        if len(alpha_spatial.shape) == 4:
            value2 = ops.matmul(alpha_spatial, value2)
        else:
            value2 = ops.matmul(self.expand_dims(alpha_spatial, -2), value2).squeeze(-2)
        attn = value1 * value2 * alpha_channel
        return attn
    
class CapsuleLowRankLayer(nn.Cell):
    def __init__(self, embed_dim, att_heads, att_mid_dim, att_mid_drop):
        super(CapsuleLowRankLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.output_dim = embed_dim
        self.head_dim = self.output_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.in_proj_q = nn.SequentialCell([
            nn.Dense(self.embed_dim, self.output_dim),
            nn.ReLU()
        ])
        self.in_proj_k = nn.SequentialCell([
            nn.Dense(self.embed_dim, self.output_dim),
            nn.ReLU()
        ])
        self.in_proj_v1 = nn.SequentialCell([
            nn.Dense(self.embed_dim, self.output_dim),
            nn.ReLU()
        ])
        self.in_proj_v2 = nn.SequentialCell([
            nn.Dense(self.embed_dim, self.output_dim),
            nn.ReLU()
        ])
        self.attn_net = Capsule()
        
        sequential = []         
        for i in range(1, len(att_mid_dim) - 1):        
            sequential.append(nn.Dense(att_mid_dim[i - 1], 256))             
            sequential.append(nn.ReLU())           
            sequential.append(nn.Dropout(att_mid_drop))         
        self.attention_basic = nn.SequentialCell(*sequential) if len(sequential) > 0 else None
        
        self.expand_dims = ops.ExpandDims()
        self.reshape = ops.Reshape()
        self.reducesum = ops.ReduceSum()
        
    def construct(self, query, key, mask, value1, value2, precompute=False):
        batch_size = query.shape[0]
        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)
        q = q.view(batch_size, self.num_heads, self.head_dim)
        v1 = v1.view(batch_size, self.num_heads, self.head_dim)
        
        key = key.view(-1, key.shape[-1])
        value2 = value2.view(-1, value2.shape[-1])
        k = self.in_proj_k(key)
        v2 = self.in_proj_v2(value2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim)
        
        attn_map = self.expand_dims(q, 1) * k
        att_map1 = self.attention_basic(attn_map)
        att_mask = mask
        if att_mask is not None:
            att_mask = self.expand_dims(att_mask, 1)
            att_mask_ext = self.expand_dims(att_mask, -1).transpose((0, 2, 1, 3))
            att_map_pool = self.reducesum(att_map1 * att_mask_ext, -1) / self.reducesum(att_mask_ext, -1)
        else:
            att_map_pool = att_map1.mean(-2)
        alpha_channel = ops.Sigmoid()(att_map_pool)
        alpha_channel = self.expand_dims(alpha_channel, -1)
        attn_map = attn_map * alpha_channel
        attn_map = self.reshape(attn_map, (batch_size, -1, self.embed_dim))
        attn= self.attn_net(attn_map)
        attn = attn.squeeze(1)
        return attn
    
class LowRankLayer(nn.Cell):
    def __init__(self, embed_dim, att_heads, att_mid_dim, att_mid_drop):
        super(LowRankLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.output_dim = embed_dim
        self.in_proj_q = nn.SequentialCell([
            nn.Dense(self.embed_dim, self.output_dim),
            nn.ReLU()
        ])
        self.in_proj_k = nn.SequentialCell([
            nn.Dense(self.embed_dim, self.output_dim),
            nn.ReLU()
        ])
        self.in_proj_v1 = nn.SequentialCell([
            nn.Dense(self.embed_dim, self.output_dim),
            nn.ReLU()
        ])
        self.in_proj_v2 = nn.SequentialCell([
            nn.Dense(self.embed_dim, self.output_dim),
            nn.ReLU()
        ])
        self.attn_net = SCAtt(att_mid_dim, att_mid_drop)
        
    def construct(self, query, key, mask, value1, value2):
        batch_size = query.shape[0]
        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)
        q = q.view(batch_size, self.num_heads, self.head_dim)
        v1 = v1.view(batch_size, self.num_heads, self.head_dim)
        
        key = key.view(-1, key.shape[-1])
        value2 = value2.view(-1, value2.shape[-1])
        k = self.in_proj_k(key)
        v2 = self.in_proj_v2(value2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        k = ops.Transpose()(k, (0, 2, 1, 3))
        v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim)
        v2 = ops.Transpose()(v2, (0, 2, 1, 3))
        
        attn_map = ops.ExpandDims()(q, -2) * k
        attn = self.attn_net(attn_map, mask, v1, v2)
        attn = attn.view(batch_size, self.num_heads * self.head_dim)
        return attn
    
class Encoder(nn.Cell):
    def __init__(self, layer_num, embed_dim, att_heads, att_mid_dim, att_mid_drop):
        super(Encoder, self).__init__()
        self.encoder = nn.CellList([])
        self.bifeat_emb = nn.CellList([])
        self.layer_norms = nn.CellList([])
        for _ in range(layer_num):
            sublayer = CapsuleLowRankLayer(embed_dim=embed_dim, att_heads=8, att_mid_dim=[128, 64, 128], att_mid_drop=0.9)
            self.encoder.append(sublayer)
            self.bifeat_emb.append(nn.SequentialCell(
                nn.Dense(2 * embed_dim, embed_dim),
                nn.Dropout(0.9)
            ))
            self.layer_norms.append(nn.LayerNorm([embed_dim]))
        self.proj = nn.Dense(embed_dim * (layer_num + 1), embed_dim)
        self.layer_norm = nn.LayerNorm([embed_dim])
        self.expand_dims = ops.ExpandDims()
        self.reducesum = ops.ReduceSum()
        self.concat_last = ops.Concat(-1)
        
    def construct(self, att_feats, gv_feat, att_mask):
        # encoder
        if gv_feat.shape[-1] == 1:
            if att_mask is not None:
                gv_feat = self.reducesum(att_feats * self.expand_dims(att_mask, -1), 1) / self.reducesum(self.expand_dims(att_mask, -1), 1)
            else:
                gv_feat = self.reducemean(att_feats, 1)
        feat_arr = [gv_feat]
        for i, encoder_layer in enumerate(self.encoder):
            gv_feat = encoder_layer(gv_feat, att_feats, att_mask, gv_feat, att_feats)
            att_feats_cat = self.concat_last((self.expand_dims(gv_feat, 1).expand_as(att_feats), att_feats))
            att_feats = self.bifeat_emb[i](att_feats_cat) + att_feats
            att_feats = self.layer_norms[i](att_feats)
            feat_arr.append(gv_feat)
        gv_feat = self.concat_last(feat_arr)
        gv_feat = self.proj(gv_feat)
        gv_feat = self.layer_norm(gv_feat)
        return gv_feat, att_feats

class Decoder(nn.Cell):
    def __init__(self, layer_num, embed_dim, att_heads, att_mid_dim, att_mid_drop):
        super(Decoder, self).__init__()
        self.decoder = nn.CellList([])
        for _ in range(layer_num):
            sublayer = LowRankLayer(embed_dim=embed_dim, att_heads=8, att_mid_dim=[128, 64, 128], att_mid_drop=0.9)
            self.decoder.append(sublayer)
        self.proj = nn.Dense(embed_dim * (layer_num + 1), embed_dim)
        self.layer_norm = nn.LayerNorm([embed_dim])
        self.concat_last = ops.Concat(-1)
        
    def construct(self, gv_feat, att_feats, att_mask):
        batch_size =  att_feats.shape[0]
        feat_arr = [gv_feat]
        for i, decoder_layer in enumerate(self.decoder):
            gv_feat = decoder_layer(gv_feat, att_feats, att_mask, gv_feat, att_feats)
            feat_arr.append(gv_feat)
        gv_feat = self.concat_last(feat_arr)
        gv_feat = self.proj(gv_feat)
        gv_feat = self.layer_norm(gv_feat)
        return gv_feat, att_feats

@ms_function
def expand_tensor(tensor, size, dim=1):
    if size==1 or tensor is None:
        return tensor
    tensor = ops.ExpandDims()(tensor, 1)
    x1 = size,
    x2 = -1,
    shape1 = tensor.shape[:dim] + x1 + tensor.shape[dim+1:]
    shape2 = tensor.shape[:dim-1] + x2 + tensor.shape[dim+1:]
    broadcast_to = ops.BroadcastTo(shape1)
    tensor = broadcast_to(tensor)
    tensor = tensor.view(shape2)
    return tensor

class CapsuleXlan(nn.Cell):
    def __init__(self, vocab_size, seq_per_img, encode_layer_num, decode_layer_num):
        super(CapsuleXlan, self).__init__()
        self.num_layers = 2
        self.rnn_size = 1024
        self.embed_dim = 1024
        self.att_heads = 8
        self.att_mid_dim = [128, 64, 128]
        self.att_mid_drop = 0.9
        self.vocab_size = vocab_size +1
        # self.batch_size = batch_size
        self.seq_per_img = seq_per_img
            
        self.word_embed = nn.SequentialCell([
            nn.Embedding(self.vocab_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.9)
        ])
        self.att_embed = nn.SequentialCell([
            nn.Dense(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.9)
        ])
        self.encoder = Encoder(layer_num=encode_layer_num, embed_dim=self.embed_dim, att_heads=self.att_heads, att_mid_dim=self.att_mid_dim, att_mid_drop=self.att_mid_drop)
        self.decoder = Decoder(layer_num=decode_layer_num, embed_dim=self.embed_dim, att_heads=self.att_heads, att_mid_dim=self.att_mid_dim, att_mid_drop=self.att_mid_drop)
        
        self.ctx_drop = nn.Dropout(0.5)
        self.att_lstm = nn.LSTMCell(self.rnn_size+self.embed_dim, self.rnn_size)
        self.att2ctx = nn.Dense(self.rnn_size+self.embed_dim, 2*self.rnn_size)
        self.dropout_lm = nn.Dropout(0.5)
        self.logit = nn.Dense(self.rnn_size, self.vocab_size)
        
        self.concat_1 = ops.Concat(1)
        self.concat_last = ops.Concat(-1)
        self.expand_dims = ops.ExpandDims()
        self.reducesum = ops.ReduceSum()
        self.reducemean = ops.ReduceMean()
        self.logsoftmax = ops.LogSoftmax()
        self.sigmoid = nn.Sigmoid()
        
    
    def construct(self, seq, att_feats):
        seq = seq.view(-1, seq.shape[-1])
        att_feats = self.att_embed(att_feats)
        gv_feat = np.ones((1, 1))
        att_mask = np.ones((att_feats.shape[0], att_feats.shape[1]))
        gv_feat, att_feats = self.encoder(att_feats, gv_feat, att_mask)
        
        gv_feat = expand_tensor(gv_feat, self.seq_per_img)
        att_feats = expand_tensor(att_feats, self.seq_per_img)
        att_mask = expand_tensor(att_mask, self.seq_per_img)
        
        batch_size = gv_feat.shape[0]
        state = [Parameter(np.zeros((self.num_layers, batch_size, self.rnn_size))), Parameter(np.zeros((self.num_layers, batch_size, self.rnn_size)))]
        outputs = np.zeros((batch_size, seq.shape[1], self.vocab_size))
        for t in range(seq.shape[1]):
            wt = seq[:,t].copy()

            if t>=1 and seq[:,t].astype('float32').max() == 0:
                break
            if gv_feat.shape[-1] == 1:
                if att_mask is not None:
                    gv_feat = self.reducesum(att_feats * self.expand_dims(att_mask, -1), 1) / self.reducesum(self.expand_dims(att_mask, -1), 1)
                else:
                    gv_feat = self.reducemean(att_feats, 1)

            xt = self.word_embed(wt)
            a = self.concat_1((xt, gv_feat+self.ctx_drop(state[0][1])))
            h_att, c_att = self.att_lstm(a, (state[0][0], state[1][0]))
            att, _ = self.decoder(h_att, att_feats, att_mask)
            ctx_input = self.concat_1((att, h_att))

            output = self.att2ctx(ctx_input)
            output = output[:, :self.rnn_size] * self.sigmoid(output[:, self.rnn_size:])
            state = [ops.Stack()([h_att, output]), ops.Stack()([c_att, state[1][1]])]
            output = self.dropout_lm(output)
            logit = self.logit(output)
            outputs[:, t] = logit
        outputs = self.logsoftmax(outputs)
        return outputs

class CapsuleXlanWithLoss(nn.Cell):
    def __init__(self, model):
        super(CapsuleXlanWithLoss, self).__init__()
        self.model = model
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        
    def construct(self, indices, input_seq, target_seq, att_feats):
        logit = self.model(input_seq, att_feats)
        logit = logit.view((-1, logit.shape[-1]))
        target_seq = target_seq.view((-1))
        mask = (target_seq > -1).astype("float32")
        loss = self.ce(logit, target_seq)
        loss = ops.ReduceSum(False)(loss * mask) / mask.sum()
        return loss
    
class CapsuleXlanTrainOneStepCell(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, sens=1.0):
        super(CapsuleXlanTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.cast = ops.Cast()
        self.hyper_map = ops.HyperMap()

    def set_sens(self, value):
        self.sens = value
    
    def construct(self, indices, input_seq, target_seq, att_feats):
        weights = self.weights
        loss = self.network(indices, input_seq, target_seq, att_feats)
        grads = self.grad(self.network, weights)(indices, input_seq, target_seq, att_feats, self.cast(ops.tuple_to_array((self.sens,)), mindspore.float32))
        grads = self.hyper_map(ops.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss
