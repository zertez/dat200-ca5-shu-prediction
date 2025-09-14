# %%
import dis
import sys
import traceback

import numpy as np

print(np.__config__.show())
# %%


def analyze_matrix_properties(n):
    """Analyze properties of identity matrix of size n"""
    identity = np.identity(n)

    # Basic properties
    print(f"\nMatrix size: {n}x{n}")
    print(f"Data type: {identity.dtype}")
    print(f"Memory usage: {identity.nbytes} bytes")

    # Check for special values
    has_nan = np.isnan(identity).any()
    has_inf = np.isinf(identity).any()
    print(f"Contains NaN: {has_nan}")
    print(f"Contains Inf: {has_inf}")

    # Try multiplication with different methods
    try:
        result_matmul = identity @ identity
        print("@ operator: Success")
        has_nan_matmul = np.isnan(result_matmul).any()
        has_inf_matmul = np.isinf(result_matmul).any()
        print(f"  Result contains NaN: {has_nan_matmul}")
        print(f"  Result contains Inf: {has_inf_matmul}")
    except Exception as e:
        print(f"@ operator: Failed with {type(e).__name__}: {e}")

    try:
        result_dot = np.dot(identity, identity)
        print("np.dot(): Success")
        has_nan_dot = np.isnan(result_dot).any()
        has_inf_dot = np.isinf(result_dot).any()
        print(f"  Result contains NaN: {has_nan_dot}")
        print(f"  Result contains Inf: {has_inf_dot}")
    except Exception as e:
        print(f"np.dot(): Failed with {type(e).__name__}: {e}")

    return identity


# Test the transition sizes
for size in range(13, 17):
    analyze_matrix_properties(size)


# %%

print("Testing different data types at size 15:")
for dtype in [np.float32, np.float64, np.int32, np.int64]:
    identity = np.identity(16, dtype=dtype)
    print(f"\nData type: {dtype}")
    try:
        with np.errstate(all="raise"):  # Convert warnings to exceptions
            result = identity @ identity
        print("@ operator: Success")
    except Exception as e:
        print(f"@ operator: Failed with {type(e).__name__}: {str(e)}")

    try:
        result = np.dot(identity, identity)
        print("np.dot(): Success")
    except Exception as e:
        print(f"np.dot(): Failed with {type(e).__name__}: {str(e)}")
# %%


def trace_matmul(a, b):
    try:
        # Set up tracing
        old_trace = sys.gettrace()
        sys.settrace(lambda frame, event, arg: print(f"Event: {event}, Function: {frame.f_code.co_name}") or old_trace)

        # Perform the operation
        result = a @ b

        # Restore normal operation
        sys.settrace(old_trace)
        return result
    except Exception as e:
        sys.settrace(old_trace)
        print(f"Error: {e}")
        traceback.print_exc()
        return None


# Test different matrix sizes and types
for dtype in [np.float32, np.float64]:
    for n in [14, 15, 16, 17]:
        print(f"\nTesting {dtype.__name__}, size {n}x{n}")
        a = np.identity(n, dtype=dtype)
        trace_matmul(a, a)


# %%

# Create a function to disassemble


def test_matmul():
    a = np.identity(10, dtype=np.float64)
    return a @ a


# Disassemble to see the bytecode
dis.dis(test_matmul)

# %%

np.__config__.show()  # Should show OpenBLAS instead of Accelerate
