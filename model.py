import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.activation import ReLU
import clip


def get_prompt_list(type):
	if type=="plain":
		return ['{}']
	elif type=="fixed":
		return ['a photo of a {}.']
	elif type=="shuffle":
		return ['a photo of a {}.', 'a photograph of a {}.', 'an image of a {}.', '{}.']
	elif type=="shuffle+":
		return ['a photo of a {}.', 'a photograph of a {}.', 'an image of a {}.', '{}.',
					'a cropped photo of a {}.', 'a good photo of a {}.', 'a photo of one {}.',
					'a bad photo of a {}.', 'a photo of the {}.']

	else:
		raise ValueError("Incorrect Prompt Type")
	

def forward_multihead_attention(inp, mod, atten_mask = None):
	"""
	inp: num_tokens*batch_size*emb_dim
	"""


	attn_in = mod.ln_1(inp)
	q,k,v = F.linear(attn_in, mod.attn.in_proj_weight, mod.attn.in_proj_bias).chunk(3,dim=-1)

	# MultiHead Projection
	num_tokens, batch_size, embed_dim = q.size()
	num_heads = mod.attn.num_heads

	dim = embed_dim//num_heads
	scale = 1.0/(float(dim)**0.5)

	q = q.contiguous().view(-1, batch_size * num_heads, dim).transpose(0, 1)
	k = k.contiguous().view(-1, batch_size * num_heads, dim).transpose(0, 1)
	v = v.contiguous().view(-1, batch_size * num_heads, dim).transpose(0, 1)

	# Q -> (bs*num_heads, num_tokens, dim)

	# (Q.K^T)/sqrt(dim)
	attn_weights = torch.bmm(q, k.transpose(1,2)) * scale

	# bs*num_heads, num_tokens, num_tokens
	attn_weights = torch.softmax(attn_weights, dim=-1)

	# (bs*heads,tokens,tokens) * (bs*heads,tokens,dim) -> (bs*heads,tokens,dim)
	attn_out = torch.bmm(attn_weights, v)
	attn_out = attn_out.transpose(0,1).contiguous().view(-1,batch_size,num_heads*dim)

	interm = inp + mod.attn.out_proj(attn_out)
	out = interm + mod.mlp(mod.ln_2(interm))

	out = mod.ln_2(out)

	return out, attn_weights

class ClipBase(nn.Module):

	def __init__(self, reduce_dim, prompt_type, device='cpu'):
		super().__init__()
		self.prompt_list = get_prompt_list(prompt_type)
		self.device = device
		print("device ",device)
		version = "ViT-B/32"
		self.clip_model, _ = clip.load(version,device = device,jit= False)
		print("crossed")
		self.visual_clip = self.clip_model.visual

		for param in self.clip_model.parameters():
			param.requires_grad_(False)

		self.film_mul = nn.Linear(512, reduce_dim)
		self.film_add = nn.Linear(512, reduce_dim)


	def visual_forward(self, input, extract_layers=()):
		
		with torch.no_grad():

			inp_shape = input.shape[2:]
			x = self.visual_clip.conv1(input)

			# shape = [bs, channel, grid ** 2]
			x = x.reshape(x.shape[0],x.shape[1],-1)
			x = x.permute(0,2,1)

			# shape = [*, grid ** 2 + 1, channel]
			cls_emb = self.visual_clip.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
			x = torch.cat([cls_emb, x], dim=1)  # shape = [*, grid ** 2 + 1, width]

			assert x.shape[1]==50, f" Shape 1 of input should be 50, got: {x}"

			standard_n_tokens = 50
			x = x + self.visual_clip.positional_embedding.to(x.dtype)

			x = self.visual_clip.ln_pre(x)
			# [Token,BS,EMD]
			x = x.permute(1,0,2)
			activations = []
			#print(self.visual_clip)
			for i, block in enumerate(self.visual_clip.transformer.resblocks):
				
				x, _ = forward_multihead_attention(x,block)

				if i in extract_layers:
					activations += [x]

			x = x.permute(1,0,2)
			
			x = self.visual_clip.ln_post(x[:,0,:])

			if self.visual_clip.proj is not None:
				x = torch.matmul(x,self.visual_clip.proj)

			return x, activations

	def get_conditional_vec(self, conditional, batch_size=32):
	
		# Modify based on what is conditional [string]*batchsize etc.
		cond = self.compute_conditional(conditional)
		return cond

	def compute_conditional(self, conditional):
		#print("conditional = ",conditional)
		text_tokens = clip.tokenize(conditional).to(self.device)
		#print("tokens",len(text_tokens))
		#print("tokens.shape",text_tokens[0].shape)
		cond = self.clip_model.encode_text(text_tokens)
		return cond

	def sample_prompts(self, words, prompt_list=None):

			prompt_list = prompt_list if prompt_list is not None else self.prompt_list
			prompt_indices = torch.multinomial(torch.ones(len(prompt_list)), len(words), replacement=True)
			prompts = [prompt_list[i] for i in prompt_indices]
			return [promt.format(w) for promt, w in zip(prompts, words)]

class ClipPred(ClipBase):
	def __init__(self,reduce_dim=128, prompt_type="shuffle+", device="cpu", upsample=False):
		super().__init__(reduce_dim, prompt_type, device)

		self.n_heads = 4
		self.extract_layers = (3,6,9)
		self.cond_layer = 0
		depth = len(self.extract_layers)
		self.upsample_proj = nn.Conv2d(reduce_dim, 1, kernel_size=1) if upsample else None

		# Only vit-b/32 supported rightnow
		self.version = "ViT-B/32"
		self.token_shape = {'ViT-B/32': (7, 7), 'ViT-B/16': (14, 14)}[self.version]

		trans_conv_ks = {'ViT-B/32': (32, 32), 'ViT-B/16': (16, 16)}[self.version]
		self.trans_conv = nn.ConvTranspose2d(reduce_dim, 1, trans_conv_ks, stride=trans_conv_ks)

		self.reduces = nn.ModuleList([nn.Linear(768, reduce_dim) for _ in range(depth)])
		self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=self.n_heads) for _ in range(len(self.extract_layers))])

	def forward(self,inp_image, conditional=None):
			
		inp_image = inp_image.to(self.device)
		cond = self.get_conditional_vec(conditional)
		
		visual_q, _activations = self.visual_forward(inp_image, extract_layers= list(self.extract_layers))

		a = None
		for i, (activation, block, reduce) in enumerate(zip(_activations, self.blocks, self.reduces)):
			
			if a is not None:
				a = reduce(activation) + a
			else:
				a = reduce(activation)

			if i == self.cond_layer:					
				a = self.film_mul(cond) * a + self.film_add(cond)

			a = block(a)
	
		a = a[1:].permute(1, 2, 0) # rm cls token and -> BS, Feats, Tokens
		size = int(math.sqrt(a.shape[2]))
		a = a.view(a.shape[0], a.shape[1], size, size)
		a = self.trans_conv(a)
		
		return a