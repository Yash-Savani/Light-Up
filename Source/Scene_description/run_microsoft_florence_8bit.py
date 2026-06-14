import cv2
import torch
import torch.ao.quantization
import time
import os
import psutil
import threading
from PIL import Image

# ============================================================
# TECHNICAL PATCHES
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
transformers.PreTrainedModel._supports_flash_attn_2 = False

from transformers import AutoProcessor, AutoModelForCausalLM
# ============================================================

latest_description = "Ready"
is_thinking = False

class Florence8BitTester:
    def __init__(self):
        self.model_id = "microsoft/Florence-2-base"
        self.device = torch.device("cpu")
        print(f"--- LOADING & QUANTIZING: {self.model_id} ---")
        
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager"
            )
            if hasattr(base_model, "language_model"):
                base_model.language_model.lm_head.weight = torch.nn.Parameter(base_model.language_model.lm_head.weight.clone())
            
            # Applying quantization
            self.model = torch.ao.quantization.quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False)
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            print(f"✅ SUCCESS: 8-bit model ready.")
        except Exception as e:
            print(f"❌ FAILED: {e}"); exit()

def ai_thread_worker(tester, frame):
    global latest_description, is_thinking
    try:
        start_time = time.time()
        # Pre-process
        frame_resized = cv2.resize(frame, (768, 768))
        image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        inputs = tester.processor(text="<DETAILED_CAPTION>", images=image, return_tensors="pt")
        
        # INFERENCE MODE: The fastest way to run PyTorch
        with torch.inference_mode():
            generated_ids = tester.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=40,
                num_beams=1,
                use_cache=False
            )
        
        generated_text = tester.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = tester.processor.post_process_generation(generated_text, task="<DETAILED_CAPTION>", image_size=(image.width, image.height))
        
        # 50 Character limit
        description = parsed_answer["<DETAILED_CAPTION>"].strip()[:50]
        
        dur = time.time() - start_time
        ram = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        cpu = psutil.cpu_percent(interval=None)
        
        latest_description = description
        log_entry = f"[{time.strftime('%H:%M:%S')}] {description}\n > Latency: {dur:.2f}s | RAM: {ram:.1f}MB | CPU: {cpu}% | Device: cpu (8-bit)\n"
        print(log_entry)
        with open("performance_log.txt", "a") as f: f.write(log_entry)

    except Exception as e:
        latest_description = f"Error: {str(e)}"
    finally:
        is_thinking = False

def main():
    tester = Florence8BitTester()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with open("performance_log.txt", "a") as log_file:
        log_file.write(f"\n{'='*60}\nSESSION: {time.ctime()} | MODEL: Florence-2 (8-bit)\n{'='*60}\n")

    prev_time = time.time()
    global is_thinking

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        fps = 1 / (time.time() - prev_time) if (time.time() - prev_time) > 0 else 0
        prev_time = time.time()

        # UI: ONLY FPS AND MODEL (Green)
        color = (0, 0, 255) if is_thinking else (0, 255, 0)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 1, 2, color, 2)
        cv2.putText(frame, "FLORENCE-2-8BIT", (20, 80), 1, 1.5, (0, 255, 0), 2)
        
        cv2.imshow('Benchmark', frame)

        key = cv2.waitKey(1) & 0xFF
        if is_thinking: time.sleep(0.01)
        if key == ord('d') and not is_thinking:
            is_thinking = True
            threading.Thread(target=ai_thread_worker, args=(tester, frame.copy()), daemon=True).start()
        elif key == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()