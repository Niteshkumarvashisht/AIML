from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn

print("DALI Advanced Tutorial")
print("=========================")

# 1. Advanced DALI Pipeline with Custom Operators
print("\n1. Advanced DALI Pipeline with Custom Operators:")
class CustomPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(CustomPipeline, self).__init__(batch_size, num_threads, device_id)

    def define_graph(self):
        images = fn.readers.file(file_root='path/to/images')
        return fn.color_twist(images, brightness=0.5)

pipe = CustomPipeline(batch_size=32, num_threads=1, device_id=0)
pipe.build()
print("Custom pipeline built.")
