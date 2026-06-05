from PIL import Image
import torch
import time
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration

class SceneDescriber:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading scene description model... This might take a few seconds...")


        from transformers import BlipProcessor, BlipForConditionalGeneration

        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.to(self.device)


        if torch.cuda.is_available():
            self.model = torch.compile(self.model)
        self.model.eval()  # Set to evaluation mode
        print("Scene description model loaded successfully!")

        # Initialize cache for recent descriptions
        self.cache = {}
        self.cache_size = 5
        self.last_process_time = 0
        self.process_interval = 1.0  # Process every 1 second

    def get_frame_hash(self, frame):
        """Generate a simple hash for the frame to use as cache key"""

        small_frame = cv2.resize(frame, (32, 32))
        return hash(small_frame.tobytes())

    @torch.no_grad()
    def get_scene_description(self, frame):
        current_time = time.time()


        if current_time - self.last_process_time < self.process_interval:

            if self.cache:
                return list(self.cache.values())[-1]
            return "Processing scene..."


        frame_hash = self.get_frame_hash(frame)


        if frame_hash in self.cache:
            return self.cache[frame_hash]

        try:

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)


            inputs = self.processor(images=image, return_tensors="pt").to(self.device)


            outputs = self.model.generate(
                **inputs,
                max_length=30,
                num_beams=3,
                length_penalty=1.0,
                num_return_sequences=1,
                temperature=0.7,
            )


            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)


            self.cache[frame_hash] = generated_text
            if len(self.cache) > self.cache_size:
                self.cache.pop(next(iter(self.cache)))

            self.last_process_time = current_time
            return generated_text

        except Exception as e:
            print(f"Error in scene description: {str(e)}")
            return "Unable to describe the scene." 