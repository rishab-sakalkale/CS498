###Q3: allreduce###
###please implement ring_allreduce method, using  pytorch's dist method is not allowed###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist

def reduce_scatter(chunks, tmp, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    for step in range(world - 1):
        send_idx = (rank - step) % world
        recv_idx = (rank - step - 1) % world
        
        send_req = dist.isend(chunks[send_idx].contiguous(), dst=left)
        recv_req = dist.irecv(tmp, src=right)
        
        send_req.wait()
        recv_req.wait()
        
        chunks[recv_idx] += tmp
        
def all_gather(chunks, tmp, current, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    for step in range(world - 1):
        send_idx = (rank - step - 1) % world
        recv_idx = (rank - step - 2) % world

        send_req = dist.isend(chunks[send_idx].contiguous(), dst=left)
        
        recv_req = dist.irecv(tmp, src=right)
        
        recv_req.wait()
        send_req.wait()
        
        chunks[recv_idx].copy_(tmp)

def ring_allreduce_(tensor: torch.Tensor, world_size = None, rankid = None):
    """In-place ring all-reduce (SUM, optional average) using isend/irecv."""
    world = world_size
    if world == 1: return tensor
    rank = rankid
    left, right = (rank - 1) % world, (rank + 1) % world

    ##following steps try to fill blank to the tensor so that final tensor can be divided to 3 chunks evenly
    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world
    #                                                                   #
    #                                                                   #
    # your code here: we cannot divide flat into 3 pieces evenly as the
    # flat lengh may not be able to divided exactly by 3....
    #
    #                                                                   #
    #                                                                   #
    #So, fill zeros at the end of flat to generate padded_flat
    # total_size = chunk_size * world
    # padding_size = total_size - n

    # padded_flat = None # modify this line and fill correct value into padded_flat
    total_size = chunk * world
    padding_size = total_size - n
    
    # Pad if necessary
    if padding_size > 0:
        padded_flat = torch.cat([flat, torch.zeros(padding_size, dtype=flat.dtype, device=flat.device)])
    else:
        padded_flat = flat.clone()

    chunks = [padded_flat[i*chunk:(i+1)*chunk] for i in range(world)]

    #                                                                   #
    #                                                                   #
    # your code here: call reduce_scatter and all_gather
    #
    #                                                                   #
    #                                                                   #
    #we provide the reduce_scatter and all_gather func prototype for you
    # You may adjust the function signature (input structure) of `reduce_scatter` and `all_gather` if needed.
    tmp = torch.empty_like(chunks[0])
    reduce_scatter(chunks, tmp, world, rank, left, right)
    all_gather(chunks, tmp, None, world, rank, left, right)
    padded_flat.copy_(torch.cat(chunks))

    
    # stitch & unpad  
    flat /= world
    tensor.view(-1).copy_(flat[:n])
    return