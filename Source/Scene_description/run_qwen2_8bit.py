import cv2
import torch
import torch.ao.quantization
import time
import os
import psutil
import threading
import warnings
from PIL import Image

# Suppress warnings for a clean report
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# ============================================================
# TECHNICAL PATCHES (Required for Python 3.13 stability)
# ============================================================
import transformers
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
import transformers.utils.import_utils as import_utils

if not hasattr(import_utils, "is_torch_fx_available"):
    import_utils.is_torch_fx_available = lambda: False

if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
    transformers.PreTrainedModel.all_tied_weights_keys = []

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
# ============================================================

def smart_word_cap(text, max_chars=50):
    """Cuts at the last full word before 50 characters."""
    if len(text) <= max_chars: return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    return truncated[:last_space].strip() if last_space != -1 else truncated.strip()

latest_description = "Memory-Safe Ready"
is_thinking = False

class Qwen8BitSafeTester:
    def __init__(self):
        self.model_id = "Qwen/Qwen2-VL-2B-Instruct"
        self.device = torch.device("cpu")
        
        print(f"\n--- LOADING QWEN2-VL: SELECTIVE 8-BIT MODE ---")
        
        try:
            # 1. Load weights using low_cpu_mem_usage to prevent the 16GB crash
            print("Loading weights into RAM (Low Memory Mode)...")
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float32, 
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            # 2. Untie weights to prevent "attribute weight already exists"
            print("Untying weights for conversion...")
            base_model.lm_head.weight = torch.nn.Parameter(base_model.lm_head.weight.clone())

            # 3. SELECTIVE QUANTIZATION
            # We only quantize the internal language layers (base_model.model)
            # This skips the vision tower, saving massive amounts of peak RAM during conversion
            print("Applying INT8 quantization to language layers only...")
            base_model.model = torch.ao.quantization.quantize_dynamic(
                base_model.model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            
            self.model = base_model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            print(f"✅ SUCCESS: Qwen2-VL loaded safely.")
        except Exception as e:
            print(f"❌ LOAD FAILED: {e}"); exit()

    def get_system_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**2), psutil.cpu_percent(interval=None)

def ai_thread_worker(tester, frame):
    global latest_description, is_thinking
    try:
        start_time = time.time()
        
        # Prepare image (Resizing to help CPU speed)
        frame_resized = cv2.resize(frame, (640, 480))
        image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in one short sentence."},
        ]}]

        text = tester.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = tester.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(tester.device)

        # Inference
        with torch.no_grad():
            generated_ids = tester.model.generate(**inputs, max_new_tokens=30)
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = tester.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        description = smart_word_cap(output_text.strip(), 50)
        
        dur = time.time() - start_time
        ram, cpu = tester.get_system_usage()
        
        latest_description = description
        timestamp = time.strftime('%H:%M:%S')
        log_entry = (f"[{timestamp}] {description}\n"
                     f" > Latency: {dur:.2f}s | RAM: {ram:.1f}MB | CPU: {cpu}% | Device: cpu-qwen-safe8\n")
        
        print(log_entry)
        with open("performance_log.txt", "a") as f: f.write(log_entry)

    except Exception as e:
        latest_description = f"Error: {str(e)[:30]}"
    finally:
        is_thinking = False

def main():
    tester = Qwen8BitSafeTester()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with open("performance_log.txt", "a") as log_file:
        log_file.write(f"\n{'='*60}\nSESSION: {time.ctime()} | QWEN2-VL-SAFE-8BIT\n{'='*60}\n")

    prev_time = time.time()
    global is_thinking

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        fps = 1 / (curr_time - prev_time) if (curr_time := time.time()) - prev_time > 0 else 0
        prev_time = curr_time

        # UI: ONLY FPS AND MODEL (Green)
        color = (0, 0, 255) if is_thinking else (0, 255, 0)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 1, 2, color, 2)
        cv2.putText(frame, "QWEN-2-VL-8BIT", (20, 80), 1, 1.5, (0, 255, 0), 2)
        
        cv2.imshow('Safe Benchmark', frame)

        key = cv2.waitKey(1) & 0xFF
        if is_thinking: time.sleep(0.01)
        if key == ord('d') and not is_thinking:
            is_thinking = True
            threading.Thread(target=ai_thread_worker, args=(tester, frame.copy()), daemon=True).start()
        elif key == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()