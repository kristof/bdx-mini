import onnxruntime


class OnnxInfer:
    def __init__(self, onnx_model_path, input_name="obs", awd=False):
        self.onnx_model_path = onnx_model_path
        
        # Optimize ONNX session for faster inference
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2  # Tune based on your CPU
        
        self.ort_session = onnxruntime.InferenceSession(
            self.onnx_model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = input_name
        self.awd = awd

    def infer(self, inputs):
        if self.awd:
            outputs = self.ort_session.run(None, {self.input_name: [inputs]})
            return outputs[0][0]
        else:
            outputs = self.ort_session.run(
                None, {self.input_name: inputs.astype("float32")}
            )
            return outputs[0]


if __name__ == "__main__":
    import argparse
    import numpy as np
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument("-n", "--num_iters", type=int, default=1000)
    args = parser.parse_args()

    oi = OnnxInfer(args.onnx_model_path, awd=True)
    
    # Auto-detect input size from model
    input_shape = oi.ort_session.get_inputs()[0].shape
    input_size = input_shape[-1]  # Last dimension is the feature count
    print(f"Model expects input size: {input_size}")
    
    inputs = np.random.uniform(size=input_size).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        oi.infer(inputs)
    
    # Benchmark
    times = []
    for i in range(args.num_iters):
        start = time.time()
        output = oi.infer(inputs)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    print(f"\nBenchmark results ({args.num_iters} iterations):")
    print(f"  Average time: {avg_time*1000:.3f} ms")
    print(f"  Average FPS:  {1/avg_time:.1f}")
    print(f"  Min time:     {min(times)*1000:.3f} ms")
    print(f"  Max time:     {max(times)*1000:.3f} ms")
