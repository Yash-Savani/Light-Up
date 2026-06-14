import cv2
import torch
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

def patch_tokenizer():
    old_init = RobertaTokenizer.__init__
    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        if not hasattr(self, "additional_special_tokens"): self.additional_special_tokens = []
        if not hasattr(self, "image_token"): self.image_token = "<image>"
    RobertaTokenizer.__init__ = new_init
patch_tokenizer()

transformers.configuration_utils.PretrainedConfig.forced_bos_token_id = None
transformers.PreTrainedModel._supports_sdpa = False

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
# ============================================================

def smart_word_cap(text, max_chars=50):
    """Cuts at the last full word before 50 characters."""
    if len(text) <= max_chars: return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    return truncated[:last_space].strip() if last_space != -1 else truncated.strip()

latest_description = "Ready"
is_thinking = False

class QwenTester:
    def __init__(self):
        self.model_id = "Qwen/Qwen2-VL-2B-Instruct"
        self.device = torch.device("cpu")
        
        print(f"\n--- LOADING QWEN2-VL (2.8 BILLION PARAMETERS) ---")
        print("Note: This model is heavy. Loading may take 1-2 minutes...")
        
        try:
            # We use bfloat16 for the best balance of speed and RAM on modern CPUs
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id, 
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            print(f"✅ SUCCESS: Qwen2-VL is active on CPU.")
        except Exception as e:
            print(f"❌ LOAD FAILED: {e}"); exit()

    def get_system_usage(self):
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)
        cpu_p = psutil.cpu_percent(interval=None)
        return ram_mb, cpu_p

def ai_thread_worker(tester, frame):
    global latest_description, is_thinking
    try:
        start_time = time.time()
        
        # 1. Prepare image for Qwen's unique processor
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        
        # 2. Qwen-VL Chat Template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image in one short sentence."},
                ],
            }
        ]

        # 3. Processing vision info
        text = tester.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = tester.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(tester.device)

        # 4. Inference (Greedy Search)
        generated_ids = tester.model.generate(**inputs, max_new_tokens=40)
        
        # Trim the output to get only the answer
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = tester.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        description = smart_word_cap(output_text.strip(), 50)
        
        # 5. Capture Metrics
        dur = time.time() - start_time
        ram, cpu = tester.get_system_usage()
        
        latest_description = description
        
        # 6. LOGGING (Exact format as requested)
        timestamp = time.strftime('%H:%M:%S')
        log_entry = (f"[{timestamp}] {description}\n"
                     f" > Latency: {dur:.2f}s | RAM: {ram:.1f}MB | CPU: {cpu}% | Device: cpu-qwen\n")
        
        print(log_entry)
        with open("performance_log.txt", "a") as f:
            f.write(log_entry)

    except Exception as e:
        latest_description = f"Error: {str(e)[:30]}"
    finally:
        is_thinking = False

def main():
    try:
        tester = QwenTester()
    except: return

    # Camera: 1280x720 HD
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with open("performance_log.txt", "a") as log_file:
        log_file.write(f"\n{'='*60}\nSESSION: {time.ctime()} | MODEL: Qwen2-VL-2B\n{'='*60}\n")

    prev_time = time.time()
    global is_thinking

    while True:
        curr_time = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # --- UI DISPLAY: GREEN FPS AND MODEL ---
        status_color = (0, 0, 255) if is_thinking else (0, 255, 0)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 1, 2, status_color, 2)
        cv2.putText(frame, "QWEN-2-VL-2B", (20, 80), 1, 1.5, (0, 255, 0), 2)
        
        cv2.imshow('Qwen Benchmarking', frame)

        key = cv2.waitKey(1) & 0xFF
        if is_thinking: time.sleep(0.01)

        if key == ord('d') and not is_thinking:
            is_thinking = True
            thread = threading.Thread(target=ai_thread_worker, args=(tester, frame.copy()), daemon=True)
            thread.start()

        elif key == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()