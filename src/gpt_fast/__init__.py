import torch

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True
torch._dynamo.config.automatic_dynamic_shapes = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.epilogue_fusion = False
torch._dynamo.config.cache_size_limit = 100000
