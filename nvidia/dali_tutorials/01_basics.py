from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import numpy as np

print("DALI Basics Tutorial")
print("====================")

# 1. Simple DALI Pipeline
print("\n1. Simple DALI Pipeline:")

class SimplePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(SimplePipeline, self).__init__(batch_size, num_threads, device_id)

    def define_graph(self):
        self.input = fn.random.uniform(range=(0.0, 1.0))
        return self.input

pipe = SimplePipeline(batch_size=1, num_threads=1, device_id=0)
pipe.build()

output = pipe.run()
print("Pipeline output:", output)

print("\nNote: This tutorial demonstrates a basic DALI pipeline. Ensure DALI is installed with CUDA support.")
