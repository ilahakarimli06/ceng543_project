
try:
    import flash_attn
    print(f"flash_attn version: {flash_attn.__version__}")
    print("\nDir(flash_attn):")
    print([x for x in dir(flash_attn) if "flash" in x])

    try:
        from flash_attn import flash_attn_interface
        print("\nDir(flash_attn.flash_attn_interface):")
        print([x for x in dir(flash_attn_interface) if "flash" in x])
    except ImportError:
        print("\nCould not import flash_attn.flash_attn_interface")

except ImportError:
    print("flash_attn not installed")
