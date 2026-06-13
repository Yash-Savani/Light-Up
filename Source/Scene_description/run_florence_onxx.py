import cv2
import torch
import time
import os
import psutil
import threading
import warnings
from PIL import Image

# Suppress all library warnings for a professional terminal output
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("optimum").setLevel(logging.ERROR)

# ============================================================
# TECHNICAL PATCHES (Required for Florence-2 + ONNX + Python 3.13)
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

# Global library attribute fixes
transformers.configuration_utils.PretrainedConfig.forced_bos_token_id = None
transformers.PreTrainedModel._supports_sdpa = False
transformers.PreTrainedModel._supports_flash_attn_2 = False

from transformers import AutoProcessor, AutoConfig
from optimum.onnxruntime import ORTModelForVision2Seq
# ============================================================

def smart_word_cap(text, max_chars=50):
    """Cuts text at the last full word before the 50-character limit."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space != -1:
        return truncated[:last_space].strip()
    return truncated.strip()

latest_description = "Ready"
is_thinking = False

class FlorenceONNXTester:
    def __init__(self):
        self.model_id = "microsoft/Florence-2-base"
        print(f"--- INITIALIZING ONNX ENGINE (Applying Dictionary Injection) ---")
        
        try:
            # 1. Load config
            config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
            
            # 2. THE ONNX FIX: Injecting specific dimension keys directly into config dict
            if hasattr(config, "vision_config"):
                v_cfg = config.vision_config
                v_cfg.__dict__["embed_dim"] = 768
                v_cfg.__dict__["hidden_size"] = 768
                v_cfg.__dict__["model_type"] = "davit"
            
            # 3. EXPORT TO ONNX (Creates an 'onnx' folder in your directory)
            self.model = ORTModelForVision2Seq.from_pretrained(
                self.model_id,
                export=True,
                config=config,
                trust_remote_code=True,
                provider="CPUExecutionProvider", # Optimized for your CPU math
                use_cache=False
            )
            
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            print(f"✅ SUCCESS: ONNX Engine Active.")
        except Exception as e:
            print(f"❌ ENGINE LOAD FAILED: {e}")
            exit()

    def get_system_usage(self):
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024) 
        cpu_p = psutil.cpu_percent(interval=None) 
        return ram_mb, cpu_p

def ai_thread_worker(tester, frame):
    global latest_description, is_thinking
    try:
        start_time = time.time()
        
        # 1. Pre-process (Force Square for Florence Vision Tower)
        frame_resized = cv2.resize(frame, (768, 768))
        image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        
        # 2. Prepare inputs
        inputs = tester.processor(text="<CAPTION>", images=image, return_tensors="pt")
        
        # 3. ONNX Inference (Greedy Search, 1 beam)
        generated_ids = tester.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=40,
            num_beams=1,
            use_cache=False
        )
        
        # 4. Decode and Parse
        generated_text = tester.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = tester.processor.post_process_generation(
            generated_text, task="<CAPTION>", image_size=(image.width, image.height)
        )
        
        # 5. Format to 50 characters with Smart Truncation
        description = smart_word_cap(parsed_answer["<CAPTION>"].strip(), 50)
        
        # 6. Capture Metrics
        dur = time.time() - start_time
        ram, cpu = tester.get_system_usage()
        
        latest_description = description
        
        # LOGGING (Exactly matching your previous successful format)
        timestamp = time.strftime('%H:%M:%S')
        log_entry = (f"[{timestamp}] {description}\n"
                     f" > Latency: {dur:.2f}s | RAM: {ram:.1f}MB | CPU: {cpu}% | Device: ONNX-Engine\n")
        
        print(log_entry)
        with open("performance_log.txt", "a") as f:
            f.write(log_entry)

    except Exception as e:
        latest_description = f"Error: {str(e)[:30]}"
    finally:
        is_thinking = False

def main():
    try:
        tester = FlorenceONNXTester()
    except: return

    # --- CAMERA SETUP: 1280x720 HD ---
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    res_label = "1280x720"

    # --- INITIAL LOG ENTRY ---
    with open("performance_log.txt", "a") as log_file:
        log_file.write(f"\n{'='*60}\nSESSION: {time.ctime()} | ENGINE: ONNX RUNTIME\n{'='*60}\n")

    prev_time = time.time()
    global is_thinking

    while True:
        curr_time = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        # FPS Calculation
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # --- UI DISPLAY: GREEN FPS AND MODEL ---
        status_color = (0, 0, 255) if is_thinking else (0, 255, 0)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 1, 2, status_color, 2)
        cv2.putText(frame, "FLORENCE-2-ONNX", (20, 80), 1, 1.5, (0, 255, 0), 2)
        
        cv2.imshow('Inference Engine Benchmark', frame)

        key = cv2.waitKey(1) & 0xFF
        
        # Stability sleep during background work to prevent "Not Responding"
        if is_thinking:
            time.sleep(0.01)

        if key == ord('d') and not is_thinking:
            is_thinking = True
            # Pass a frame copy to the background thread
            thread = threading.Thread(target=ai_thread_worker, args=(tester, frame.copy()), daemon=True)
            thread.start()

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()