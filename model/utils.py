from typing import Optional, Tuple, Union, List

import torch
from torch_scatter import  scatter, segment_csr, gather_csr

import numbers
import itertools

# Optional Tensor
OptTensor = Optional[torch.Tensor]

def repeat(src, length):
    if src is None:
        return None
    if isinstance(src, numbers.Number):
        return list(itertools.repeat(src, length))
    if (len(src) > length):
        return src[:length]
    if (len(src) < length):
        return src + list(itertools.repeat(src[-1], length - len(src)))
    return src

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, torch.Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))

def remove_self_loops(edge_index: torch.Tensor, edge_attr: OptTensor = None) -> Tuple[torch.Tensor, OptTensor]:
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]

def add_self_loops(
        edge_index: torch.Tensor, edge_attr: OptTensor = None,
        fill_value: Union[float, torch.Tensor, str] = None,
        num_nodes: Optional[int] = None) -> Tuple[torch.Tensor, OptTensor]:
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted or has multi-dimensional edge features
    (:obj:`edge_attr != None`), edge features of self-loops will be added
    according to :obj:`fill_value`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_attr != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    num_node = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0, num_node, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_attr is not None:
        if fill_value is None:
            loop_attr = edge_attr.new_full((num_node, ) + edge_attr.size()[1:], 1.)

        elif isinstance(fill_value, (int, float)):
            loop_attr = edge_attr.new_full((num_node, ) + edge_attr.size()[1:], fill_value)
        elif isinstance(fill_value, torch.Tensor):
            loop_attr   = fill_value.to(edge_attr.device, edge_attr.dtype)
            if edge_attr.dim() != loop_attr.dim():
                loop_attr = loop_attr.unsqueeze(0)

            sizes       = [num_node] + [1] * (loop_attr.dim() - 1)
            loop_attr   = loop_attr.repeat(*sizes)

        elif isinstance(fill_value, str):
            loop_attr = scatter(edge_attr, edge_index[1], dim=0, dim_size=num_node, reduce=fill_value)
        else:
            raise AttributeError("No valid 'fill_value' provided")

        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index, edge_attr

def sort_edge_index(
    edge_index: torch.Tensor,
    edge_attr: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    num_nodes: Optional[int] = None,
    sort_by_row: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    """Row-wise sorts :obj:`edge_index`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will re-shuffle and remove duplicates for all
            its entries. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        sort_by_row (bool, optional): If set to :obj:`False`, will sort
            :obj:`edge_index` column-wise.
    :rtype: :class:`LongTensor` if :attr:`edge_attr` is :obj:`None`, else
        (:class:`LongTensor`, :obj:`Tensor` or :obj:`List[Tensor]]`)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    idx = edge_index[1 - int(sort_by_row)] * num_nodes
    idx += edge_index[int(sort_by_row)]

    perm = idx.argsort()

    edge_index = edge_index[:, perm]

    if edge_attr is None:
        return edge_index
    elif isinstance(edge_attr, torch.Tensor):
        return edge_index, edge_attr[perm]
    else:
        return edge_index, [e[perm] for e in edge_attr]

def softmax(src: torch.Tensor, index: Optional[torch.Tensor] = None,
            ptr: Optional[torch.Tensor] = None, num_nodes: Optional[int] = None,
            dim: int = 0) -> torch.Tensor:
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.
    Args:
        src (Tensor): The source tensor.
        index (LongTensor, optional): The indices of elements for applying the
            softmax. (default: :obj:`None`)
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dim (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)
    :rtype: :class:`Tensor`
    """
    if ptr is not None:
        dim     = dim + src.dim() if dim < 0 else dim
        size    = ([1] * dim) + [-1]
        ptr     = ptr.view(size)
        src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr)
        out     = (src - src_max).exp()
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)

    elif index is not None:
        N       = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src, index, dim, dim_size=N, reduce='max')
        src_max = src_max.index_select(dim, index)
        out     = (src - src_max).exp()
        out_sum = scatter(out, index, dim, dim_size=N, reduce='sum')
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    return out / (out_sum + 1e-16)