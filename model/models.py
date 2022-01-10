from torch._C import _jit_clear_class_registry
from helper import *
from model.compgcn_conv import CompGCNConv
from model.compgcn_conv_basis import CompGCNConvBasis

from model.pooling import TopKPooling

from torch_sparse import spspmm

from model.utils import repeat, add_self_loops, remove_self_loops, sort_edge_index

class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p			= params
		self.act		= torch.tanh
		self.bceloss	= torch.nn.BCELoss()

	def loss(self, pred, true_label):
		return self.bceloss(pred, true_label)
		
class CompGCNBase(BaseModel):
	def __init__(self, edge_index, edge_type, num_rel, params=None):
		super(CompGCNBase, self).__init__(params)

		self.edge_index		= edge_index
		self.edge_type		= edge_type
		self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
		self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim))
		self.device			= self.edge_index.device

		if self.p.num_bases > 0:
			self.init_rel  		= get_param((self.p.num_bases,   self.p.init_dim))
		else:
			if self.p.score_func == 'transe': 	
				self.init_rel 	= get_param((num_rel,   self.p.init_dim))
			else: 					
				self.init_rel 	= get_param((num_rel*2, self.p.init_dim))

		if self.p.num_bases > 0:
			self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None
		else:
			self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim, num_rel, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def forward_base(self, sub, rel, drop1, drop2):

		r		= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		x, r	= self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
		x		= drop1(x)
		x, r	= self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) 	if self.p.gcn_layer == 2 else (x, r)
		x		= drop2(x) 							if self.p.gcn_layer == 2 else x

		sub_emb	= torch.index_select(x, 0, sub)
		rel_emb	= torch.index_select(r, 0, rel)

		return sub_emb, rel_emb, x

class UCompGCNBase(BaseModel):
	def __init__(self, edge_index, edge_type, num_rel, params=None):
		super(UCompGCNBase, self).__init__(params)
		assert self.p.depth >= 1
		self.edge_index			= edge_index
		self.edge_type			= edge_type

		self.in_channels		= self.p.init_dim
		self.hidden_channels 	= self.p.gcn_dim
		self.out_channels 		= self.p.embed_dim

		self.depth				= self.p.depth
		self.pool_ratios 		= repeat(self.p.pool_ratios, self.depth)
		self.sum_res 			= self.p.sum_res

		self.init_embed			= get_param((self.p.num_ent, self.p.init_dim))
		if self.p.num_bases > 0:
			self.init_rel  		= get_param((self.p.num_bases, self.p.init_dim))
		else:
			if self.p.score_func == 'transe': 	
				self.init_rel 	= get_param((num_rel,   self.p.init_dim))
			else: 					
				self.init_rel 	= get_param((num_rel*2, self.p.init_dim))

		channels = self.hidden_channels
		
		self.down_convs 		= torch.nn.ModuleList()
		self.pools 				= torch.nn.ModuleList()
		self.down_convs.append(CompGCNConv(self.in_channels, channels, num_rel, act=self.act, params=self.p))
		for i in range(self.depth):
			self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
			self.down_convs.append(CompGCNConv(channels, channels, num_rel, act=self.act, params=self.p))
		
		in_channels = channels if self.sum_res else 2 * channels

		self.up_convs = torch.nn.ModuleList()
		for i in range(self.depth - 1):
			self.up_convs.append(CompGCNConv(in_channels, channels, num_rel, act=self.act, params=self.p))
		self.up_convs.append(CompGCNConv(in_channels, self.out_channels, num_rel, act=self.act, params=self.p))

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def augment_adj(self, edge_index, edge_type, num_nodes):
		edge_type 				= edge_type.type(torch.cuda.DoubleTensor)
		edge_index, edge_type 	= remove_self_loops(edge_index, edge_type)
		edge_index, edge_type 	= add_self_loops(edge_index, edge_type, num_nodes=num_nodes)
		edge_index, edge_type 	= sort_edge_index(edge_index, edge_type, num_nodes)
		edge_index, edge_type 	= spspmm(edge_index, edge_type, edge_index, edge_type, num_nodes, num_nodes, num_nodes)
		edge_index, edge_type 	= remove_self_loops(edge_index, edge_type)
		return edge_index, edge_type
	
	def forward_base(self, sub, rel, drop1, drop2, batch=None):
		edge_index		= self.edge_index
		edge_weight		= self.edge_type

		r				= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		x, r			= self.down_convs[0](self.init_embed, edge_index, edge_weight, rel_embed=r)
		xs 				= [x]
		rs 				= [r]
		edge_indices 	= [edge_index]
		edge_weights	= [edge_weight]
		perms 			= []

		for i in range(1, self.depth + 1):
			#edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
			x, edge_index, edge_weight, batch, perm, _ 	= self.pools[i - 1](x, edge_index, edge_weight)
			x, r 										= self.down_convs[i](x, edge_index, edge_weight, rel_embed=rs[i-1])
			if i < self.depth:
				xs 				+= [x]
				edge_indices 	+= [edge_index]
				edge_weights 	+= [edge_weight]
				rs 				+= [r]
			perms 										+= [perm]

		for i in range(self.depth):
			j 			= self.depth - 1 - i
			res 		= xs[j]
			edge_index 	= edge_indices[j]
			edge_weight = edge_weights[j]
			perm 		= perms[j]

			up 			= torch.zeros_like(res)
			up[perm] 	= x
			x 			= res + up if self.sum_res else torch.cat((res, up), dim=-1)

			x, r 		= self.up_convs[i](x, edge_index, edge_weight, rel_embed=rs[j])
			
			
		sub_emb	= torch.index_select(x, 0, sub)
		rel_emb	= torch.index_select(r, 0, rel)
		return sub_emb, rel_emb, x


class CompGCN_TransE(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb						= sub_emb + rel_emb
		x							= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)		
		score						= torch.sigmoid(x)

		return score

class UCompGCN_TransE(UCompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb						= sub_emb + rel_emb
		x							= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)		
		score						= torch.sigmoid(x)

		return score

class CompGCN_DistMult(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb						= sub_emb * rel_emb
		x 							= torch.mm(obj_emb, all_ent.transpose(1, 0))
		x 							+= self.bias.expand_as(x)
		score 						= torch.sigmoid(x)

		return score

class UCompGCN_DistMult(UCompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb						= sub_emb * rel_emb
		x 							= torch.mm(obj_emb, all_ent.transpose(1, 0))
		x 							+= self.bias.expand_as(x)
		score 						= torch.sigmoid(x)

		return score

class CompGCN_ConvE(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

		self.bn0			= torch.nn.BatchNorm2d(1)
		self.bn1			= torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2			= torch.nn.BatchNorm1d(self.p.embed_dim)
		
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

		flat_sz_h			= int(2 * self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w			= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz		= flat_sz_h * flat_sz_w * self.p.num_filt
		self.fc				= torch.nn.Linear(self.flat_sz, self.p.embed_dim)

	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
		stk_inp						= self.concat(sub_emb, rel_emb)
		x							= self.bn0(stk_inp)
		x							= self.m_conv1(x)
		x							= self.bn1(x)
		x							= F.relu(x)
		x							= self.feature_drop(x)
		x							= x.view(-1, self.flat_sz)
		x							= self.fc(x)
		x							= self.hidden_drop2(x)
		x							= self.bn2(x)
		x							= F.relu(x)
		x							= torch.mm(x, all_ent.transpose(1,0))
		x 							+= self.bias.expand_as(x)
		score 						= torch.sigmoid(x)

		return score

class UCompGCN_ConvE(UCompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

		self.bn0			= torch.nn.BatchNorm2d(1)
		self.bn1			= torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2			= torch.nn.BatchNorm1d(self.p.embed_dim)
		
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

		flat_sz_h			= int(2 * self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w			= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz		= flat_sz_h * flat_sz_w * self.p.num_filt
		self.fc				= torch.nn.Linear(self.flat_sz, self.p.embed_dim)

	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp

	def forward(self, sub, rel):
		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
		stk_inp						= self.concat(sub_emb, rel_emb)
		x							= self.bn0(stk_inp)
		x							= self.m_conv1(x)
		x							= self.bn1(x)
		x							= F.relu(x)
		x							= self.feature_drop(x)
		x							= x.view(-1, self.flat_sz)
		x							= self.fc(x)
		x							= self.hidden_drop2(x)
		x							= self.bn2(x)
		x							= F.relu(x)
		x							= torch.mm(x, all_ent.transpose(1,0))
		x 							+= self.bias.expand_as(x)
		score 						= torch.sigmoid(x)

		return score
