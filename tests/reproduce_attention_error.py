import mlx.core as mx

def reproduce():
    B = 1
    H = 64
    L = 9
    D = 32
    
    # Simulate the shapes from the traceback
    queries = mx.random.uniform(shape=(B, H, L, D))
    keys = mx.random.uniform(shape=(B, H, L, D))
    values = mx.random.uniform(shape=(B, H, L, D))
    
    # Mask shape reported in traceback: (1, 1, 9, 9)
    mask = mx.full((1, 1, L, L), -1e9)
    
    scale = D ** -0.5
    
    print("Calling mx.fast.scaled_dot_product_attention with shapes:")
    print(f"Q: {queries.shape}")
    print(f"K: {keys.shape}")
    print(f"V: {values.shape}")
    print(f"Mask: {mask.shape}")
    
    try:
        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask
        )
        print("Success!")
        print(out.shape)
    except ValueError as e:
        print("Caught expected ValueError:")
        print(e)
    except Exception as e:
        print(f"Caught unexpected {type(e)}:")
        print(e)

    print("\nTesting Float16:")
    queries_fp16 = queries.astype(mx.float16)
    keys_fp16 = keys.astype(mx.float16)
    values_fp16 = values.astype(mx.float16)
    mask_fp16 = mask.astype(mx.float16)
    
    try:
        out = mx.fast.scaled_dot_product_attention(
            queries_fp16, keys_fp16, values_fp16, scale=scale, mask=mask_fp16
        )
        print("Success (Float16)!")
    except Exception as e:
        print(f"Caught error (Float16): {e}")

if __name__ == "__main__":
    reproduce()
