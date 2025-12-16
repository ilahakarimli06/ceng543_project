
import inspect
from flash_attn import flash_attn_func

print("Sig:", inspect.signature(flash_attn_func))
print("Doc:", flash_attn_func.__doc__)
