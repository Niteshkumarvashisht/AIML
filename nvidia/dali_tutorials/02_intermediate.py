from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn

print("DALI Intermediate Tutorial")
print("==========================")

# 1. DALI Pipeline with Image Augmentation
print("\n1. DALI Pipeline with Image Augmentation:")
class AugmentationPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(AugmentationPipeline, self).__init__(batch_size, num_threads, device_id)

    def define_graph(self):
        images = fn.readers.file(file_root='path/to/images')
        return fn.flip(images, horizontal=True)

pipe = AugmentationPipeline(batch_size=32, num_threads=1, device_id=0)
pipe.build()
print("Pipeline built.")
